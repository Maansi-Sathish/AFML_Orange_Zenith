"""
bench_seqlen.py  --  Person 2 (Text)
Sequence-length x batch-size sweep (forward pass only, random token ids, no training needed).
Records latency, tokens/s and peak GPU memory for each (batch, seq_len); OOM is recorded, not fatal.

Example:
  !python bench_seqlen.py --arch transformer --out_csv /kaggle/working/results/sweep_transformer.csv
  !python bench_seqlen.py --arch lstm        --out_csv /kaggle/working/results/sweep_lstm.csv
  !python bench_seqlen.py --arch cfc         --out_csv /kaggle/working/results/sweep_cfc.csv

Later, Person 1's measure.py can wrap run_one() to add energy per row.
NOTE: this is the "prefill / full forward" test. A decode test (one token at a time, Transformer
with KV cache vs recurrent state) needs a step() function in Person 3's models - ask for it.
"""
import argparse, csv, os, time
import torch
from train_text import build_model, count_params


def run_one(model, B, T, vocab, device, use_amp, reps):
    x = torch.randint(0, vocab, (B, T), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        for _ in range(3):                      # warm-up
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated()
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return dict(latency_ms=mean * 1e3, latency_std_ms=std * 1e3, tokens_per_s=B * T / mean,
                peak_mem_mb=peak / 2**20, extra_mem_mb=(peak - base) / 2**20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["transformer", "lstm", "cfc"], required=True)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--d_model", type=int, default=320)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=5)
    ap.add_argument("--seq_lens", default="128,256,512,1024,2048,4096")
    ap.add_argument("--batch_sizes", default="1,4,16")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out_csv", default="/kaggle/working/results/sweep.csv")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = "cuda"
    use_amp = args.precision == "fp16"
    model = build_model(args.arch, args.vocab, args.d_model, args.n_layer, args.n_head).to(device).eval()
    n_params = count_params(model)
    print(f"{args.arch}: {n_params/1e6:.2f}M params")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    for B in [int(b) for b in args.batch_sizes.split(",")]:
        for T in [int(t) for t in args.seq_lens.split(",")]:
            row = dict(arch=args.arch, params=n_params, precision=args.precision, batch=B, seq_len=T)
            try:
                row.update(run_one(model, B, T, args.vocab, device, use_amp, args.reps))
                row["status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                row["status"] = "OOM"
                torch.cuda.empty_cache()
            print(row)
            rows.append(row)

    keys = ["arch", "params", "precision", "batch", "seq_len", "status", "latency_ms",
            "latency_std_ms", "tokens_per_s", "peak_mem_mb", "extra_mem_mb"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("saved", args.out_csv)


if __name__ == "__main__":
    main()
