"""
prep_data.py  --  Person 2 (Text)
Prepares LMSYS-Chat-1M for training small language models.

What it does:
  1. Downloads LMSYS-Chat-1M from Hugging Face (needs HF token + accepted licence)
  2. Keeps English, non-flagged, non-redacted conversations
  3. Removes duplicate conversations (same first user message)
  4. Splits train / val / test by hash of the first message (no leakage)
  5. Trains ONE shared byte-level BPE tokenizer on the train split only
  6. Writes train.bin / val.bin / test.bin (uint16 token ids) + meta.json

Usage (Kaggle):
  !python prep_data.py --out_dir /kaggle/working/data --sample 20000      # quick dry run
  !python prep_data.py --out_dir /kaggle/working/data                     # full run

Do NOT upload the raw dataset or the token files to public places (licence).
"""
import argparse, hashlib, json, os, random, time
import numpy as np

SPECIALS = ["<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>"]


def get_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:  # Kaggle secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def conversation_to_text(conv):
    """Turn [{'role':..., 'content':...}, ...] into one training string."""
    parts = []
    for m in conv:
        content = (m.get("content") or "").strip()
        if not content:
            return None  # skip conversations with empty turns
        tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
        parts.append(tag + content)
    return "".join(parts) + "<|eos|>"


def filter_and_split(rows, keep_redacted=False, min_chars=20, max_chars=8000):
    """rows: iterable of dicts. Returns dict split -> list of texts, plus stats."""
    seen = set()
    out = {"train": [], "val": [], "test": []}
    stats = dict(total=0, not_english=0, flagged=0, redacted=0, bad_length=0, duplicate=0, kept=0)
    for ex in rows:
        stats["total"] += 1
        if ex["language"] != "English":
            stats["not_english"] += 1
            continue
        mods = ex.get("openai_moderation") or []
        if any(isinstance(m, dict) and m.get("flagged", False) for m in mods):
            stats["flagged"] += 1
            continue
        if ex.get("redacted") and not keep_redacted:
            stats["redacted"] += 1
            continue
        conv = ex["conversation"]
        text = conversation_to_text(conv)
        if text is None or not (min_chars <= len(text) <= max_chars):
            stats["bad_length"] += 1
            continue
        key = hashlib.md5(conv[0]["content"].strip().lower().encode("utf-8")).hexdigest()
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        bucket = int(key, 16) % 100          # same first message -> same split
        split = "val" if bucket == 0 else "test" if bucket == 1 else "train"
        out[split].append(text)
        stats["kept"] += 1
    return out, stats


def train_tokenizer(texts, vocab_size):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def tokenize_to_array(tok, texts, max_tokens, chunk=5000):
    """Tokenize texts in order until max_tokens reached. Returns uint16 array."""
    arrays, total = [], 0
    for i in range(0, len(texts), chunk):
        enc = tok.encode_batch(texts[i:i + chunk])
        for e in enc:
            arrays.append(np.array(e.ids, dtype=np.uint16))
            total += len(e.ids)
        if total >= max_tokens:
            break
    arr = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint16)
    return arr[:max_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/kaggle/working/data")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train_tokens", type=int, default=60_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--sample", type=int, default=0, help="use only first N rows (dry run)")
    ap.add_argument("--keep_redacted", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    t0 = time.time()

    from datasets import load_dataset
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", token=get_hf_token())
    ds = ds.select_columns(["conversation", "language", "openai_moderation", "redacted"])
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))
    print(f"loaded {len(ds):,} rows in {time.time()-t0:.0f}s")

    splits, stats = filter_and_split(ds, keep_redacted=args.keep_redacted)
    print("filter stats:", stats)
    random.shuffle(splits["train"])  # fixed seed -> reproducible

    # tokenizer is trained on (a sample of) the TRAIN split only
    tok = train_tokenizer(splits["train"][:150_000], args.vocab)
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    print("tokenizer vocab size:", tok.get_vocab_size())

    counts = {}
    for name, budget in [("train", args.train_tokens), ("val", args.eval_tokens), ("test", args.eval_tokens)]:
        arr = tokenize_to_array(tok, splits[name], budget)
        arr.tofile(os.path.join(args.out_dir, f"{name}.bin"))
        counts[name] = int(len(arr))
        print(f"{name}: {len(arr):,} tokens")

    meta = dict(vocab_size=tok.get_vocab_size(), token_counts=counts, filter_stats=stats,
                seed=args.seed, specials=SPECIALS, dtype="uint16",
                requested=dict(train=args.train_tokens, eval=args.eval_tokens))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {time.time()-t0:.0f}s ->", args.out_dir)


if __name__ == "__main__":
    main()
