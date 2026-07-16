#!/usr/bin/env python3
"""
Ribo-seq foundation-model SMOKE TEST + throughput benchmark.

Two jobs in one script:
  1. Confirm the ROCm / PyTorch stack works on the MI300X (fwd+bwd, bf16, attention, AdamW).
  2. Measure real training throughput so we can extrapolate the full sweep from a measured number
     instead of a guess.

It trains a representative transformer on SYNTHETIC ribo-seq-shaped data (random tokens at a
realistic vocab / sequence length). The compute shape (params, seq len, batch, fwd+bwd) matches what
the real model will do, so the tokens/sec it reports is what matters. No real data needed.

Dependency: torch only. Runs on ROCm, CUDA, or CPU (auto-detected).

Usage:
  python smoke_test.py                     # default: 'large' (~200M) preset, 200 timed steps
  python smoke_test.py --preset small      # sweep the grid: tiny / small / medium / large
  python smoke_test.py --batch-size 32     # MI300X has 192GB/GPU, you can push this way up
  python smoke_test.py --steps 500
  python smoke_test.py --list-presets

Run each of the four presets once and you have the throughput for every model size in our sweep.
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (d_model, n_layers, n_heads) chosen to land near the four model sizes in the real sweep.
# Actual parameter count is printed at runtime (don't trust the label, trust the number).
PRESETS = {
    "tiny":   dict(d_model=128,  n_layers=4,  n_heads=4),    # ~1M
    "small":  dict(d_model=320,  n_layers=6,  n_heads=5),    # ~10M
    "medium": dict(d_model=640,  n_layers=8,  n_heads=8),    # ~50M
    "large":  dict(d_model=1024, n_layers=12, n_heads=16),   # ~200M
}


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Plain decoder-style transformer. Stand-in for the real masked-denoising model;
    per-token FLOPs are the same, which is all that matters for timing."""
    def __init__(self, vocab, seq_len, d_model, n_layers, n_heads):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="large", choices=list(PRESETS))
    p.add_argument("--seq-len", type=int, default=1024, help="tokens per sample (transcript window)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--vocab", type=int, default=128, help="64 codons + count bins + specials")
    p.add_argument("--steps", type=int, default=200, help="timed steps (one 'epoch' for this test)")
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup steps")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--list-presets", action="store_true")
    args = p.parse_args()

    if args.list_presets:
        for name, cfg in PRESETS.items():
            print(f"{name:8s} {cfg}")
        return

    if torch.cuda.is_available():
        device = "cuda"                       # ROCm reports as 'cuda' in PyTorch too
        dev_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        dev_name = "CPU (no GPU found -- numbers are NOT representative)"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    cfg = PRESETS[args.preset]

    print("=" * 68)
    print("  RIBO-SEQ FM SMOKE TEST + THROUGHPUT BENCHMARK")
    print("=" * 68)
    print(f"  device      : {dev_name}")
    print(f"  torch       : {torch.__version__}")
    print(f"  preset      : {args.preset}  {cfg}")
    print(f"  seq_len     : {args.seq_len}   batch: {args.batch_size}   dtype: {args.dtype}")
    print("=" * 68)

    model = Transformer(args.vocab, args.seq_len, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters  : {human(n_params)}  ({n_params:,})")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tokens_per_step = args.batch_size * args.seq_len

    def one_step():
        idx = torch.randint(0, args.vocab, (args.batch_size, args.seq_len), device=device)
        tgt = torch.randint(0, args.vocab, (args.batch_size, args.seq_len), device=device)
        opt.zero_grad(set_to_none=True)
        use_amp = device == "cuda" and dtype != torch.float32
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            logits = model(idx)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab), tgt.reshape(-1))
        loss.backward()
        opt.step()
        return loss.item()

    print(f"  warmup      : {args.warmup} steps ...")
    for _ in range(args.warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print(f"  timing      : {args.steps} steps ...")
    t0 = time.perf_counter()
    last = 0.0
    for _ in range(args.steps):
        last = one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total_tokens = args.steps * tokens_per_step
    tok_per_s = total_tokens / dt
    ms_per_step = 1000 * dt / args.steps
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    print("-" * 68)
    print(f"  RESULTS")
    print(f"  time for {args.steps} steps : {dt:.1f} s   ({ms_per_step:.1f} ms/step)")
    print(f"  throughput          : {human(tok_per_s)} tokens/sec")
    print(f"  final loss          : {last:.3f}  (random data -> ~ln(vocab)={math.log(args.vocab):.2f}, sanity only)")
    if device == "cuda":
        print(f"  peak GPU memory     : {peak_gb:.1f} GB / ~192 GB per MI300X")
    print("-" * 68)

    # ---- extrapolation: how long is one epoch of the REAL corpus, for THIS model size ----
    # We don't yet know the exact real token count, so show a range. Multiply linearly.
    print("  EXTRAPOLATION (this model size, single GPU):")
    print("  'epoch' = one full pass over the pretraining corpus.")
    print(f"  {'corpus tokens':>16s}  {'1 epoch (1 GPU)':>18s}  {'1 epoch (8 GPU)':>18s}")
    for corpus_tok in [1e9, 1e10, 5e10, 1e11]:
        secs_1 = corpus_tok / tok_per_s
        secs_8 = secs_1 / 8
        print(f"  {human(corpus_tok):>16s}  {secs_1/3600:>15.2f} h  {secs_8/3600:>15.2f} h")
    print("-" * 68)
    print("  To size the whole sweep: run this at tiny/small/medium/large, note tokens/sec")
    print("  for each, and multiply by that size's token budget x number of epochs x")
    print("  number of grid cells. The sweep is ~70-90 runs but almost all are the small")
    print("  presets, which finish in minutes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
