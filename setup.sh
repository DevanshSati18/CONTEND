#!/usr/bin/env bash
# CONTEND - ONE file that sets up everything this project needs, on any of its machines.
#
#   bash setup.sh             # detect the device and set it up (safe to re-run: skips what is done)
#   bash setup.sh --pi        # force a device:  --pi | --nvidia | --mac | --linux
#   bash setup.sh --check     # only print the version report (changes nothing)
#
# What it does, per device
#   Raspberry Pi   64-bit OS check, apt packages, .venv with CPU PyTorch + ONNX Runtime + analysis libs,
#                  performance CPU governor (made permanent), co-runners, version report
#   Jetson Nano    JetPack 4.6 check (stops on Orin), 4 GB swap (needed to build torchvision), apt
#                  packages, NVIDIA PyTorch 1.10 wheel from ~/wheels, torchvision 0.11.1 from source,
#                  MAXN + locked clocks + fan, `python` -> python3.6, co-runners, version report
#   Mac / Linux    .venv with the full host toolchain (ultralytics, ONNX export, attack, analysis)
# Afterwards:  python3 runscript.py --pi | --nvidia | --dev
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd)
LOG="$ROOT/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   OK   %s\n' "$*"; }
warn() { printf '   WARN %s\n' "$*"; }
die()  { printf '\n   STOP %s\n   (full log: %s)\n' "$*" "$LOG"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------ device detection
detect() {
  if [ -f /etc/nv_tegra_release ]; then echo nvidia
  elif grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then echo pi
  elif [ "$(uname -s)" = Darwin ]; then echo mac
  else echo linux; fi
}
MODE=""; CHECK=0
for a in "$@"; do
  case "$a" in
    --pi) MODE=pi ;; --nvidia|--jetson) MODE=nvidia ;; --mac) MODE=mac ;; --linux|--dev) MODE=linux ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) die "unknown option $a" ;;
  esac
done
MODE=${MODE:-$(detect)}

# ------------------------------------------------------------------ version report
report() {
  say "Version report ($MODE)"
  [ -f /etc/nv_tegra_release ] && echo "   L4T:     $(head -1 /etc/nv_tegra_release)"
  [ -f /proc/device-tree/model ] && echo "   Board:   $(tr -d '\0' < /proc/device-tree/model)"
  echo "   OS:      $( (lsb_release -ds 2>/dev/null || sw_vers -productVersion 2>/dev/null || uname -sr) ) $(uname -m)"
  [ -x /usr/local/cuda/bin/nvcc ] && echo "   CUDA:    $(/usr/local/cuda/bin/nvcc --version | tail -1)"
  [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] && \
    echo "   CPU governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  have nvpmodel && echo "   Power:   $(sudo -n nvpmodel -q 2>/dev/null | head -1 || echo 'run with sudo to read')"
  echo "   gcc:     $(gcc -dumpversion 2>/dev/null || echo MISSING)"
  echo "   co-runners: $( [ -x corunners/bwhog ] && [ -x corunners/cpuhog ] && echo built || echo NOT built)"
  "$PY" - <<'EOF'
import importlib, sys
print("   Python:  %s  (%s)" % (sys.version.split()[0], sys.executable))
mods = ["numpy", "cv2", "torch", "torchvision", "onnxruntime", "tensorrt", "pandas", "scipy",
        "yaml", "matplotlib", "statsmodels", "ultralytics"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception:
        v = "-"
    print("   %-12s %s" % (m, v))
try:
    import torch
    print("   torch CUDA: %s" % torch.cuda.is_available())
except Exception:
    pass
EOF
}

pick_python() {
  if [ "$MODE" = nvidia ]; then PY=$(command -v python3)
  elif [ -x .venv/bin/python ]; then PY="$ROOT/.venv/bin/python"
  else PY=$(command -v python3); fi
}

if [ "$CHECK" = 1 ]; then pick_python; report; exit 0; fi

build_corunners() {
  say "Co-runners (memory-bandwidth + CPU control)"
  have gcc || die "gcc missing"
  gcc -O2 -o corunners/bwhog corunners/bwhog.c
  gcc -O2 -o corunners/cpuhog corunners/cpuhog.c
  ok "built corunners/bwhog and corunners/cpuhog"
}

venv_setup() {   # $1 = extra pip args for torch
  say "Python environment (.venv)"
  have python3 || die "python3 missing"
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -q --upgrade pip wheel
  if ! python -c "import torch, torchvision" 2>/dev/null; then
    python -m pip install -q torch torchvision $1 || python -m pip install -q torch torchvision
  fi
  ok "torch $(python -c 'import torch;print(torch.__version__)')"
}

# ------------------------------------------------------------------ Raspberry Pi
setup_pi() {
  say "Raspberry Pi: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
  [ "$(uname -m)" = aarch64 ] || die "needs the 64-bit Raspberry Pi OS (uname -m = $(uname -m))"
  echo raspberry_pi > .device
  sudo apt-get update -q
  sudo apt-get install -y -q python3-venv python3-dev gcc git unzip curl tmux libopenblas-dev
  venv_setup "--index-url https://download.pytorch.org/whl/cpu"
  python -m pip install -q onnxruntime opencv-python-headless numpy pandas scipy statsmodels matplotlib pyyaml
  ok "packages installed"
  say "CPU governor -> performance (permanent)"
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$c" >/dev/null; done
  if [ ! -f /etc/systemd/system/contend-governor.service ]; then
    sudo tee /etc/systemd/system/contend-governor.service >/dev/null <<'EOF'
[Unit]
Description=CONTEND: performance CPU governor (contend-governor)
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $c; done'
[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl enable -q contend-governor.service || true
  fi
  ok "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  if have vcgencmd; then
    t=$(vcgencmd get_throttled | cut -d= -f2)
    [ "$t" = 0x0 ] && ok "no throttling / under-voltage ($t)" || warn "throttled=$t: use the official PSU and a cooler"
  fi
  build_corunners
}

# ------------------------------------------------------------------ Jetson Nano
setup_nvidia() {
  rel=$(head -1 /etc/nv_tegra_release)
  say "Jetson: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null) | $rel"
  case "$rel" in
    *R32*) ok "JetPack 4.x (Jetson Nano)";;
    *) die "this is not JetPack 4.6 (Jetson Orin uses R35/R36). Tell Claude the board: setup differs";;
  esac
  echo jetson_nano > .device
  say "Swap (4 GB; the torchvision build runs out of memory without it)"
  if [ "$(free -m | awk '/Swap/{print $2}')" -lt 3000 ]; then
    sudo fallocate -l 4G /var/swapfile && sudo chmod 600 /var/swapfile && sudo mkswap /var/swapfile >/dev/null
    sudo swapon /var/swapfile
    grep -q /var/swapfile /etc/fstab || echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab >/dev/null
  fi
  ok "swap: $(free -m | awk '/Swap/{print $2}') MB"
  sudo apt-get update -q
  sudo apt-get install -y -q python3-pip python3-dev python3-yaml python3-pandas python3-scipy \
       libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev gcc git tmux
  say "python -> python3.6"
  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 >/dev/null
  ok "$(python --version 2>&1)"
  PY=$(command -v python3)
  say "pip (Ubuntu 18.04 ships pip 9, too old for aarch64 manylinux2014 wheels)"
  $PY -m pip install --user -q --upgrade "pip<22" "setuptools<60" wheel
  ok "$($PY -m pip --version)"
  say "PyTorch 1.10.0 (NVIDIA wheel)"
  if ! $PY -c "import torch" 2>/dev/null; then
    WHL=$(ls "$HOME"/wheels/torch-1.10.0-cp36-cp36m-linux_aarch64.whl 2>/dev/null || true)
    [ -n "$WHL" ] || die "put torch-1.10.0-cp36-cp36m-linux_aarch64.whl in ~/wheels/ (NVIDIA 'PyTorch for Jetson' forum, JetPack 4.6) and re-run"
    $PY -m pip install --user -q "Cython<3" "numpy==1.19.4"
    $PY -m pip install --user -q "$WHL"
  fi
  $PY -c "import torch; assert torch.cuda.is_available()" || die "torch installed but CUDA not available"
  ok "torch $($PY -c 'import torch;print(torch.__version__)') with CUDA"
  say "torchvision 0.11.1 from source (~45-60 min, once)"
  if ! $PY -c "import torchvision" 2>/dev/null; then
    [ -d "$HOME/torchvision" ] || git clone -q --depth 1 --branch v0.11.1 https://github.com/pytorch/vision "$HOME/torchvision"
    (cd "$HOME/torchvision" && BUILD_VERSION=0.11.1 MAX_JOBS=2 $PY setup.py install --user >/dev/null)
  fi
  $PY -c "import torch, torchvision; b=torch.rand(50,4).cuda()*100; b[:,2:]+=b[:,:2]; torchvision.ops.nms(b, torch.rand(50).cuda(), .5)" \
    || die "torchvision CUDA NMS failed"
  ok "torchvision $($PY -c 'import torchvision;print(torchvision.__version__)') (CUDA NMS works)"
  $PY -c "import tensorrt, cv2" || die "TensorRT/OpenCV Python bindings missing: reflash JetPack 4.6.x"
  say "Performance mode: MAXN, locked clocks, fan"
  sudo nvpmodel -m 0 || warn "MAXN needs the 5V/4A barrel supply + J48 jumper"
  sudo jetson_clocks || true
  sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' 2>/dev/null || true
  ok "$(sudo nvpmodel -q | head -1)"
  build_corunners
}

# ------------------------------------------------------------------ Mac / Linux dev host
setup_host() {
  say "Development host ($(uname -s) $(uname -m))"
  [ -f .device ] || echo dev_vm > .device
  if [ "$MODE" = mac ]; then
    have gcc || die "install the Xcode command-line tools: xcode-select --install"
    have gh || warn "no 'gh' (GitHub CLI): brew install gh   (needed to push the repo)"
  fi
  venv_setup ""
  python -m pip install -q -r requirements-host.txt
  ok "host toolchain installed"
  python tests/test_nms.py | tail -1
  build_corunners
}

case "$MODE" in
  pi) setup_pi ;; nvidia) setup_nvidia ;; mac|linux) setup_host ;;
esac
pick_python
report
say "Done. Next:"
case "$MODE" in
  pi)     echo "   tmux new -s contend;  python3 runscript.py --pi --push-each" ;;
  nvidia) echo "   tmux new -s contend;  python3 runscript.py --nvidia --push-each" ;;
  *)      echo "   python3 runscript.py --smoke      (then --compile after the devices push)" ;;
esac
echo "   log: $LOG"
