'''Design choices:
1.Net energy= NVML energy - (idle power * duration)
2.The hardware energy counter is logged next to the integrated power, so you get a free internal cross-check.
3.CodeCarbon’s GPU energy is logged separately, so you can report all sources.
'''
import time, threading, csv, os
import numpy as np
import torch, pynvml

class PowerSampler:
    """Polls NVML power in a background thread; also reads the hardware energy counter."""
    def __init__(self, gpu=0, interval=0.05):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.interval = interval
        self._stop = threading.Event()
        self.t, self.w = [], []

    def _loop(self):
        while not self._stop.is_set():
            self.t.append(time.perf_counter())
            self.w.append(pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
            time.sleep(self.interval)

    def start(self):
        self.t, self.w = [], []
        self._stop.clear()
        self._e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set(); self._th.join()
        e_counter = (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) - self._e0) / 1000.0  # J
        t, w = np.array(self.t), np.array(self.w)
        e_integral = float(np.trapz(w, t))  # J
        return dict(avg_w=float(w.mean()), peak_w=float(w.max()),
                    e_integral=e_integral, e_counter=e_counter)

def measure_idle(seconds=15, gpu=0):
    s = PowerSampler(gpu); s.start(); time.sleep(seconds); return s.stop()["avg_w"]

def run_measured(fn, n_items, meta, out_csv="results.csv", idle_w=None,
                 warmup_s=5, min_duration=20, trials=5, gpu=0):
    """fn(): does ONE unit of work (e.g. one batch). n_items: items (samples/tokens) per call.
    meta: dict of columns, e.g. {model, dataset, phase, batch_size, seq_len}."""
    from codecarbon import EmissionsTracker
    if idle_w is None:
        idle_w = measure_idle(gpu=gpu)

    t0 = time.time()                       # warm-up: reach steady clocks/temperature
    while time.time() - t0 < warmup_s:
        fn()
    torch.cuda.synchronize()

    rows = []
    for trial in range(trials):
        torch.cuda.synchronize()
        s = PowerSampler(gpu); cc = EmissionsTracker(log_level="error", save_to_file=False)
        s.start(); cc.start()
        start, iters = time.perf_counter(), 0
        while time.perf_counter() - start < min_duration:   # min duration -> stable reading
            fn(); iters += 1
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        cc.stop(); r = s.stop()
        net = r["e_integral"] - idle_w * dur                # idle subtraction
        items = iters * n_items
        row = dict(meta, trial=trial, duration_s=dur, iters=iters, idle_w=idle_w,
                   avg_power_w=r["avg_w"], peak_power_w=r["peak_w"],
                   energy_j_nvml=r["e_integral"], energy_j_counter=r["e_counter"],
                   energy_j_net=net, j_per_item=net / items, items_per_s=items / dur,
                   codecarbon_gpu_kwh=getattr(cc.final_emissions_data, "gpu_energy", None),
                   peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
                   temp_c=pynvml.nvmlDeviceGetTemperature(s.h, 0))
        rows.append(row)
        new = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=row.keys())
            if new: wr.writeheader()
            wr.writerow(row)
    return rows
