"""Inference backends with identical stage-timing boundaries.

Every backend's `process(frame_u8, delta)` returns a dict with
  t_pre_ms  : uint8 frame -> float input (+ perturbation) -> on device
  t_inf_ms  : network forward pass
  t_nms_ms  : confidence filter + sort + NMS + max_det (src/nms.py)
  t_d2h_ms  : final detections copied to host memory
plus candidate counts from nms(). GPU segments are synchronised before each timestamp,
so wall-clock (perf_counter) differences are exact stage times.

Backends
  trt  : TensorRT engine on the Jetson (TensorRT 8.x binding API and 10.x tensor API)
  ort  : ONNX Runtime (CPU on the Raspberry Pi / host; CUDA EP if installed)
  torch: Ultralytics PyTorch model (host only; used for attack checks)
"""
import time

import numpy as np
import torch

from common import to_input
from nms import nms

now = time.perf_counter


class _Base(object):
    device = "cpu"

    def __init__(self, nms_cfg):
        self.nms_cfg = dict(nms_cfg)
        self.nms_device = self.nms_cfg.pop("device", None) or self.device

    def _sync(self):
        if self.device.startswith("cuda") or self.nms_device.startswith("cuda"):
            torch.cuda.synchronize()

    def _nms(self, pred):
        if pred.device.type != self.nms_device.split(":")[0]:
            pred = pred.to(self.nms_device)
        kw = dict(conf_thres=self.nms_cfg["conf_thres"], iou_thres=self.nms_cfg["iou_thres"],
                  max_det=self.nms_cfg["max_det"], max_nms=self.nms_cfg["max_nms"],
                  agnostic=self.nms_cfg.get("agnostic", False),
                  time_limit_s=self.nms_cfg.get("time_limit_s", 2.05))
        return nms(pred, **kw)

    def process(self, frame_u8, delta=None):
        t0 = now()
        self.load_input(to_input(frame_u8, delta))
        self._sync()
        t1 = now()
        pred = self.forward()
        self._sync()
        t2 = now()
        det, info = self._nms(pred)
        self._sync()
        t3 = now()
        det_host = det.cpu().numpy()
        t4 = now()
        info.update(t_pre_ms=(t1 - t0) * 1e3, t_inf_ms=(t2 - t1) * 1e3,
                    t_nms_ms=(t3 - t2) * 1e3, t_d2h_ms=(t4 - t3) * 1e3)
        return info, det_host


class ORTBackend(_Base):
    def __init__(self, onnx_path, nms_cfg, threads=None):
        import onnxruntime as ort
        from common import n_pinned_cores
        so = ort.SessionOptions()
        threads = threads or n_pinned_cores()      # never more threads than pinned cores
        if threads:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        prov = [pv for pv in ["CUDAExecutionProvider", "CPUExecutionProvider"] if pv in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, so, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.device = "cpu"
        super(ORTBackend, self).__init__(nms_cfg)
        self.x = None

    def load_input(self, x):
        self.x = x

    def forward(self):
        return torch.from_numpy(self.sess.run(None, {self.inp: self.x})[0])


class TorchBackend(_Base):
    def __init__(self, pt_path, nms_cfg, device=None, half=False):
        from ultralytics import YOLO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(pt_path).model.to(self.device).eval()
        self.half = half and self.device.startswith("cuda")
        if self.half:
            self.model.half()
        super(TorchBackend, self).__init__(nms_cfg)
        self.x = None

    def load_input(self, x):
        t = torch.from_numpy(x).to(self.device)
        self.x = t.half() if self.half else t

    @torch.no_grad()
    def forward(self):
        y = self.model(self.x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y.float()


class TRTBackend(_Base):
    """TensorRT engine with torch-allocated CUDA buffers (no pycuda needed)."""

    def __init__(self, engine_path, nms_cfg):
        import tensorrt as trt
        self.device = "cuda"
        super(TRTBackend, self).__init__(nms_cfg)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.v10 = not hasattr(self.engine, "num_bindings")
        self.bufs, self.names, self.inputs, self.outputs = [], [], [], []
        n = self.engine.num_io_tensors if self.v10 else self.engine.num_bindings
        for i in range(n):
            if self.v10:
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                name = self.engine.get_binding_name(i)
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_in = self.engine.binding_is_input(i)
            tdtype = {np.float32: torch.float32, np.float16: torch.float16, np.int32: torch.int32}[np.dtype(dtype).type]
            buf = torch.empty(shape, dtype=tdtype, device="cuda")
            self.bufs.append(buf)
            self.names.append(name)
            (self.inputs if is_in else self.outputs).append(i)
            if self.v10:
                self.ctx.set_tensor_address(name, buf.data_ptr())
        self.stream = torch.cuda.Stream()
        # the detection head output is the (1, 4+nc, N) tensor
        self.out_idx = max(self.outputs, key=lambda k: self.bufs[k].numel())

    def load_input(self, x):
        b = self.bufs[self.inputs[0]]
        b.copy_(torch.from_numpy(x).to(dtype=b.dtype))

    def forward(self):
        if self.v10:
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
        else:
            self.ctx.execute_v2([b.data_ptr() for b in self.bufs])
        return self.bufs[self.out_idx].float()


def make_backend(kind, path, nms_cfg, **kw):
    kind = kind.lower()
    if kind == "trt":
        return TRTBackend(path, nms_cfg)
    if kind == "ort":
        return ORTBackend(path, nms_cfg, threads=kw.get("threads"))
    if kind == "torch":
        return TorchBackend(path, nms_cfg, device=kw.get("device"))
    raise ValueError("unknown backend " + kind)
