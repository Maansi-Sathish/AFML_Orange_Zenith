"""
train_text.py  --  Person 2 (Text)
Trains a small causal language model on the token files made by prep_data.py.

* Works for any architecture: --arch transformer | lstm | cfc
* If Person 3's models.py exists (with build_model(...)), it is used automatically.
  Otherwise the simple placeholder models below are used so you can start today.
* Logs validation perplexity every N tokens (log.csv) -> this is the "quality vs
  cost" curve that Person 1's measure.py will later attach energy to.
* train(args) can be wrapped by measure.py.

Example (Kaggle, quick timing test):
  !python train_text.py --arch transformer --data_dir /kaggle/working/data \
      --out_dir /kaggle/working/runs/tf_test --max_tokens 2000000 --eval_every_tokens 500000
"""
import argparse, csv, importlib.util, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Placeholder models (REPLACE by Person 3's models.py when ready)
# All three use the same embedding style, tied input/output embeddings.
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = [t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class _LMBase(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=d ** -0.5)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def embed(self, idx):
        return self.emb(idx) * math.sqrt(self.d)


class TinyGPT(_LMBase):
    def __init__(self, vocab, d, n_layer, n_head):
        super().__init__(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])

    def _pos(self, T, device):  # sinusoidal, no parameters, works for any length
        pos = torch.arange(T, device=device).float()[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device).float() * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, idx):
        x = self.embed(idx) + self._pos(idx.shape[1], idx.device).to(self.emb.weight.dtype)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


class LSTMLM(_LMBase):
    def __init__(self, vocab, d, n_layer):
        super().__init__(vocab, d)
        self.rnn = nn.LSTM(d, d, num_layers=n_layer, batch_first=True)

    def forward(self, idx):
        x, _ = self.rnn(self.embed(idx))
        return self.head(self.ln_f(x))


class CfCLM(_LMBase):
    """Placeholder CfC LM built on the `ncps` library (pip install ncps).
    Check the ncps docs if the constructor signature differs in your version."""
    def __init__(self, vocab, d, n_layer):
        super().__init__(vocab, d)
        from ncps.torch import CfC
        self.layers = nn.ModuleList([CfC(d, d, batch_first=True) for _ in range(n_layer)])
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])

    def forward(self, idx):
        x = self.embed(idx)
        for rnn, ln in zip(self.layers, self.lns):
            y, _ = rnn(ln(x))
            x = x + y
        return self.head(self.ln_f(x))


def build_model(arch, vocab, d_model, n_layer, n_head):
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "models.py")):      # teammate's file, next to this script
        import sys, importlib
        sys.path.insert(0, here)
        ext = importlib.import_module("models")
        if hasattr(ext, "build_lm"):                          # Person 3's language-model builder
            print("using build_lm from models.py with ITS default sizes "
                  "(--d_model / --n_layer / --n_head are ignored)")
            return ext.build_lm(arch)
        print("models.py found but it has no build_lm(...) -> using the placeholder LM models")
    if arch == "transformer":
        return TinyGPT(vocab, d_model, n_layer, n_head)
    if arch == "lstm":
        return LSTMLM(vocab, d_model, n_layer)
    if arch == "cfc":
        return CfCLM(vocab, d_model, n_layer)
    raise ValueError(arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters())  # tied weights counted once


# ----------------------------------------------------------------------------
# Data + evaluation
# ----------------------------------------------------------------------------
def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(arr, B, T, rng, device):
    ix = rng.integers(0, len(arr) - T - 1, size=B)
    x = np.stack([arr[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, arr, T, B, device, use_amp, max_tokens):
    """Perplexity over non-overlapping windows of length T (context resets each window).
    Same protocol for every architecture -> comparable numbers."""
    model.eval()
    n_win = min(len(arr) - 1, max_tokens) // T
    total_loss, total_tok = 0.0, 0
    for s in range(0, n_win, B):
        ids = range(s, min(s + B, n_win))
        x = np.stack([arr[i * T:(i + 1) * T] for i in ids]).astype(np.int64)
        y = np.stack([arr[i * T + 1:(i + 1) * T + 1] for i in ids]).astype(np.int64)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tok += y.numel()
    model.train()
    avg = total_loss / total_tok
    return avg, math.exp(avg)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab = meta["vocab_size"]
    train_arr = load_bin(os.path.join(args.data_dir, "train.bin"))
    val_arr = load_bin(os.path.join(args.data_dir, "val.bin"))
    test_arr = load_bin(os.path.join(args.data_dir, "test.bin"))

    model = build_model(args.arch, vocab, args.d_model, args.n_layer, args.n_head).to(device)
    n_params = count_params(model)
    with torch.no_grad():                                     # sanity: logits must be (B, T, vocab)
        probe = model(torch.zeros(1, 8, dtype=torch.long, device=device))
    assert probe.shape == (1, 8, vocab), f"model output {tuple(probe.shape)} but tokenizer vocab is {vocab}"
    print(f"arch={args.arch}  params={n_params/1e6:.2f}M  device={device}  precision={args.precision}")

    use_amp = args.precision == "fp16" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    tok_per_step = args.batch * args.seq_len
    total_steps = max(1, args.max_tokens // tok_per_step)
    eval_every = max(1, args.eval_every_tokens // tok_per_step)
    warmup = max(1, min(200, total_steps // 10))

    def lr_at(step):
        if step <= warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log_path = os.path.join(args.out_dir, "log.csv")
    logf = open(log_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(["step", "tokens_seen", "train_loss", "val_loss", "val_ppl", "train_time_s", "peak_mem_mb"])

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_time, t_last, run_loss, run_n = 0.0, time.time(), 0.0, 0
    best_ppl = float("inf")

    for step in range(1, total_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_arr, args.batch, args.seq_len, rng, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item() if step % 10 == 0 else 0.0
        run_n += 1 if step % 10 == 0 else 0

        if step % eval_every == 0 or step == total_steps:
            if device == "cuda":
                torch.cuda.synchronize()
            train_time += time.time() - t_last          # eval time is NOT counted
            v_loss, v_ppl = evaluate(model, val_arr, args.seq_len, 64, device, use_amp, args.eval_tokens)
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            tr = run_loss / max(1, run_n)
            writer.writerow([step, step * tok_per_step, f"{tr:.4f}", f"{v_loss:.4f}", f"{v_ppl:.3f}",
                             f"{train_time:.1f}", f"{peak:.0f}"])
            logf.flush()
            print(f"step {step}/{total_steps}  tokens {step*tok_per_step/1e6:.1f}M  "
                  f"train {tr:.3f}  val_ppl {v_ppl:.2f}  time {train_time:.0f}s")
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_last.pt"))
            if v_ppl < best_ppl:
                best_ppl = v_ppl
                torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pt"))
            run_loss, run_n = 0.0, 0
            if device == "cuda":
                torch.cuda.synchronize()
            t_last = time.time()

    logf.close()
    t_loss, t_ppl = evaluate(model, test_arr, args.seq_len, 64, device, use_amp, args.eval_tokens)
    result = dict(arch=args.arch, params=n_params, precision=args.precision, seq_len=args.seq_len,
                  batch=args.batch, tokens_trained=total_steps * tok_per_step, train_time_s=train_time,
                  best_val_ppl=best_ppl, test_ppl=t_ppl, seed=args.seed,
                  d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, lr=args.lr)
    json.dump(result, open(os.path.join(args.out_dir, "result.json"), "w"), indent=2)
    print("RESULT", result)
    return result


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["transformer", "lstm", "cfc"], required=True)
    ap.add_argument("--data_dir", default="/kaggle/working/data")
    ap.add_argument("--out_dir", default="/kaggle/working/runs/run")
    ap.add_argument("--d_model", type=int, default=320)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_tokens", type=int, default=50_000_000)
    ap.add_argument("--eval_every_tokens", type=int, default=2_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(get_args())
