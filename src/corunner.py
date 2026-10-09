"""Start/stop the background co-runners defined in configs/contend.yaml (contention:).

Each instance is pinned to one of platform.corunner_cores and writes its own throughput
log (GB/s for bwhog, Mit/s for cpuhog), which is the independent validation used in G2.

  python src/corunner.py --name mem_high --seconds 10     # standalone validation
"""
import argparse
import os
import signal
import subprocess
import time

from common import load_config, p

BIN = {"bw": p("corunners/bwhog"), "cpu": p("corunners/cpuhog")}


class CoRunner(object):
    def __init__(self, cfg, name, log_dir):
        self.spec = cfg["contention"][name]
        self.cores = cfg["platform"]["corunner_cores"]
        self.name, self.log_dir, self.procs, self.logs = name, log_dir, [], []

    def start(self):
        if self.spec["kind"] == "none":
            return self
        os.makedirs(self.log_dir, exist_ok=True)
        for i in range(int(self.spec.get("instances", 1))):
            core = self.cores[i % len(self.cores)]
            args = [BIN[self.spec["kind"]]]
            if self.spec["kind"] == "bw":
                args.append(str(self.spec.get("buffer_mb", 256)))
            log = open(os.path.join(self.log_dir, "%s_%d.log" % (self.name, i)), "w")
            self.logs.append(log)
            self.procs.append(subprocess.Popen(["taskset", "-c", str(core)] + args, stdout=log,
                                               stderr=subprocess.STDOUT, preexec_fn=os.setsid))
        time.sleep(3.0)                       # let it reach steady state before timing
        return self

    def stop(self):
        for pr in self.procs:
            try:
                os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
                pr.wait(timeout=5)
            except Exception:
                pass
        for lg in self.logs:
            lg.close()
        self.procs, self.logs = [], []

    @staticmethod
    def summarize(log_dir, name):
        """Mean throughput per instance from its log (skips the first, ramp-up line)."""
        out = []
        i = 0
        while True:
            fn = os.path.join(log_dir, "%s_%d.log" % (name, i))
            if not os.path.exists(fn):
                break
            vals = []
            for line in open(fn):
                try:
                    vals.append(float(line.split()[0]))
                except Exception:
                    pass
            out.append(sum(vals[1:]) / max(len(vals[1:]), 1) if len(vals) > 1 else float("nan"))
            i += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    cfg = load_config()
    d = p("results/corunner_check")
    c = CoRunner(cfg, a.name, d).start()
    time.sleep(a.seconds)
    c.stop()
    unit = "GB/s" if cfg["contention"][a.name]["kind"] == "bw" else "Mit/s"
    print("%s: per-instance mean %s %s" % (a.name, ["%.2f" % v for v in CoRunner.summarize(d, a.name)], unit))


if __name__ == "__main__":
    main()
