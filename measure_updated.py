"""GPU energy measurement harness (NVML + CodeCarbon) for the AFML experiments.

Design choices:
1. Gross energy (integrated NVML power over the measurement window) is the primary
   metric. Net energy = gross - idle_w * duration is secondary and baseline-dependent.
2. The NVML hardware energy counter is logged next to the integrated power as an
   internal cross-check (it is optional: skipped if the GPU does not expose it).
3. CodeCarbon energy is logged separately so every source can be reported.
4. Peak GPU memory is reset before each trial, so peak_mem_mb is per trial and not
   a running maximum over the whole process.

Default protocol (shared by the HAR and text experiments):
    15 s idle reference, 5 s warm-up, >= 20 s per trial, 5 trials.

`fn()` does ONE unit of work (one inference batch, or one training step) and `n_items`
is how many items that unit contains. For training use n_items=1 to get joules per
step, or n_items=batch*seq_len to get joules per token.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pynvml
import torch

META_FIELDS = [
    "model", "dataset", "impl", "phase", "seed", "params",
    "batch_size", "seq_len", "commit",
]
MEASUREMENT_FIELDS = [
    "trial", "duration_s", "iters", "idle_w",
    "avg_power_w", "peak_power_w", "energy_j_nvml", "energy_j_counter",
    "energy_j_net", "j_per_item_gross", "j_per_item_net", "items_per_s",
    "codecarbon_total_kwh", "codecarbon_cpu_kwh", "codecarbon_gpu_kwh",
    "peak_mem_mb", "temp_c", "temp_start_c", "sm_clock_mhz", "throttle_mask",
]
CSV_FIELDS = META_FIELDS + MEASUREMENT_FIELDS

# np.trapz was removed in NumPy 2.x; np.trapezoid is the replacement (NumPy >= 2.0).
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _throttle(handle) -> Optional[int]:
    """NVML throttle-reason bitmask, or None where the driver does not support it."""
    for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                 "nvmlDeviceGetCurrentClocksThrottleReasons"):
        fn = getattr(pynvml, name, None)
        if fn is not None:
            try:
                return int(fn(handle))
            except pynvml.NVMLError:
                continue
    return None


def _energy_counter_j(handle) -> Optional[float]:
    """NVML cumulative energy counter in joules, or None if unavailable."""
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
    except pynvml.NVMLError:
        return None


class PowerSampler:
    """Polls NVML power in a background thread and integrates it over a time window."""

    def __init__(self, gpu: int = 0, interval: float = 0.05):
        if interval <= 0:
            raise ValueError("interval must be positive")
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.interval = interval
        self._stop = threading.Event()
        self._th = None
        self.t, self.w = [], []

    def _loop(self):
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                self.t.append(time.perf_counter())
                self.w.append(w)
            except pynvml.NVMLError:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.t, self.w = [], []
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def abort(self):
        """Stop the polling thread without integrating (used on errors)."""
        self._stop.set()
        if self._th is not None:
            self._th.join()

    def stop(self, t0: Optional[float] = None, t1: Optional[float] = None):
        """Stop sampling and integrate power over [t0, t1] (perf_counter times).

        Defaults to the full sampled range. The window edges are linearly
        interpolated so the integral covers exactly [t0, t1].
        """
        self._stop.set()
        if self._th is not None:
            self._th.join()
        t = np.asarray(self.t, dtype=float)
        w = np.asarray(self.w, dtype=float)
        if len(t) < 2:
            raise RuntimeError("NVML power sampler collected fewer than two samples.")
        t0 = float(t[0]) if t0 is None else t0
        t1 = float(t[-1]) if t1 is None else t1
        if t1 <= t0:
            raise ValueError("Measurement end must be after start.")
        inside = (t > t0) & (t < t1)
        grid = np.concatenate(([t0], t[inside], [t1]))
        e_integral = float(_trapezoid(np.interp(grid, t, w), grid))  # J
        mask = (t >= t0) & (t <= t1)
        peak_w = float(w[mask].max()) if mask.any() else float(w.max())
        return dict(avg_w=e_integral / (t1 - t0), peak_w=peak_w, e_integral=e_integral)


def measure_idle(seconds: float = 15, gpu: int = 0) -> float:
    """Average GPU power (W) over `seconds` with no workload running."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    s = PowerSampler(gpu)
    s.start()
    time.sleep(seconds)
    return s.stop()["avg_w"]


def _append_row(out_csv: str, row: dict):
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_data = path.exists() and path.stat().st_size > 0
    if has_data:
        with path.open("r", newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), [])
        if existing != CSV_FIELDS:
            raise ValueError(
                f"CSV schema mismatch in {out_csv}. Back up or rename the old CSV before rerunning.")
    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not has_data:
            wr.writeheader()
        wr.writerow(row)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_measured(fn: Callable[[], object], n_items: int, meta: dict,
                 out_csv: str = "results.csv", idle_w: Optional[float] = None,
                 warmup_s: float = 5, min_duration: float = 20, trials: int = 5,
                 gpu: int = 0):
    """Measure `trials` repeated runs of `fn` and append one CSV row per trial.

    fn():      ONE unit of work (a batch or a training step).
    n_items:   items per call (samples, tokens, or 1 for per-step energy).
    meta:      dict with any of META_FIELDS (model, dataset, phase, batch_size, seq_len, ...).
    idle_w:    idle reference in W; measured for `measure_idle()` seconds if omitted.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if min_duration <= 0 or warmup_s < 0 or trials < 1:
        raise ValueError("Use min_duration > 0, warmup_s >= 0, and trials >= 1")

    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        EmissionsTracker = None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    if idle_w is None:
        idle_w = measure_idle(gpu=gpu)

    # Warm-up (excluded from all measurement windows): reach steady clocks/temperature.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        fn()
    _sync()

    metadata = {"model": "unknown", "dataset": "unknown", "impl": "default",
                "phase": "infer", "seed": -1, "params": None,
                "batch_size": None, "seq_len": None, "commit": "unknown"}
    metadata.update(meta or {})
    rows = []

    for trial in range(trials):
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # per-trial peak memory
        tracker = (EmissionsTracker(log_level="error", save_to_file=False)
                   if EmissionsTracker else None)
        sampler = PowerSampler(gpu)
        try:
            temp_start = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp_start = None

        sampler.start()
        tracker_started = False
        if tracker:
            try:
                tracker.start()
                tracker_started = True
            except Exception:
                tracker = None
        e0 = _energy_counter_j(handle)
        start = time.perf_counter()
        iters = 0
        try:
            while time.perf_counter() - start < min_duration:
                fn()
                iters += 1
            _sync()
            end = time.perf_counter()
            e1 = _energy_counter_j(handle)
        except BaseException:
            sampler.abort()
            if tracker_started:
                try:
                    tracker.stop()
                except Exception:
                    pass
            raise

        emissions = None
        if tracker_started:
            try:
                tracker.stop()
                emissions = getattr(tracker, "final_emissions_data", None)
            except Exception:
                pass
        sampled = sampler.stop(start, end)

        dur = end - start
        gross = sampled["e_integral"]
        net = gross - float(idle_w) * dur
        items = iters * n_items
        counter_j = e1 - e0 if e0 is not None and e1 is not None else None
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            sm_clock = None
        try:
            temp_end = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp_end = None

        row = dict(metadata)
        row.update(
            trial=trial, duration_s=dur, iters=iters, idle_w=idle_w,
            avg_power_w=sampled["avg_w"], peak_power_w=sampled["peak_w"],
            energy_j_nvml=gross, energy_j_counter=counter_j, energy_j_net=net,
            j_per_item_gross=gross / items if items else None,
            j_per_item_net=net / items if items else None,
            items_per_s=items / dur,
            codecarbon_total_kwh=getattr(emissions, "energy_consumed", None),
            codecarbon_cpu_kwh=getattr(emissions, "cpu_energy", None),
            codecarbon_gpu_kwh=getattr(emissions, "gpu_energy", None),
            peak_mem_mb=(torch.cuda.max_memory_allocated() / 2**20
                         if torch.cuda.is_available() else None),
            temp_c=temp_end, temp_start_c=temp_start, sm_clock_mhz=sm_clock,
            throttle_mask=_throttle(handle),
        )
        _append_row(out_csv, row)
        rows.append(row)
    return rows
