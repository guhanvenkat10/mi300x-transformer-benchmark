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
from torch.nn.attention import SDPBackend, sdpa_kernel

# SDPA backends, best -> worst. FLASH and EFFICIENT never materialize the seq x seq matrix.
BACKENDS = {
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math":      SDPBackend.MATH,
}


def probe_backends(device):
    """Directly test which SDPA backends actually RUN on this stack, rather than guessing from
    memory. On ROCm the auto-dispatcher often silently skips flash, so we check each explicitly."""
    if device != "cuda":
        return {}
    import warnings
    q = torch.randn(1, 4, 512, 64, device=device, dtype=torch.bfloat16)
    avail = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel([backend]):
                    F.scaled_dot_product_attention(q, q, q, is_causal=True)
                torch.cuda.synchronize()
                avail[name] = True
            except Exception:
                avail[name] = False
    return avail

# (d_model, n_layers, n_heads) chosen to land near the four model sizes in the real sweep.
# Actual parameter count is printed at runtime (don't trust the label, trust the number).
PRESETS = {
    "tiny":   dict(d_model=128,  n_layers=4,  n_heads=4),    # ~1M
    "small":  dict(d_model=320,  n_layers=6,  n_heads=5),    # ~10M
    "medium": dict(d_model=640,  n_layers=8,  n_heads=8),    # ~50M
    "large":  dict(d_model=1024, n_layers=12, n_heads=16),   # ~200M
}


class Attention(nn.Module):
    """Attention via scaled_dot_product_attention, with the backend FORCED via sdpa_kernel().

    Why forced: on ROCm, SDPA's auto-dispatcher frequently skips the fused flash/efficient kernel
    and silently falls back to materializing the full (batch, heads, seq, seq) score matrix --
    O(seq^2) memory per layer, memory-bandwidth-bound instead of compute-bound. Forcing the backend
    guarantees the fused path runs (or errors loudly if it genuinely isn't built).
    """
    def __init__(self, d_model, n_heads, backend):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.backend = backend  # an SDPBackend enum

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        with sdpa_kernel([self.backend]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, backend):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, backend)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Plain decoder-style transformer. Stand-in for the real masked-denoising model;
    per-token FLOPs are the same, which is all that matters for timing."""
    def __init__(self, vocab, seq_len, d_model, n_layers, n_heads, backend):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, backend) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
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
    p.add_argument("--attn", default="auto", choices=["auto", "flash", "efficient", "math"],
                   help="which SDPA backend to FORCE. auto = best available (flash>efficient>math).")
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

    # Probe which SDPA backends actually run on this stack, then pick the one to force.
    avail = probe_backends(device)
    if args.attn == "auto":
        chosen = next((n for n in ["flash", "efficient", "math"] if avail.get(n)), "math")
    else:
        chosen = args.attn
        if device == "cuda" and not avail.get(chosen, False):
            print(f"  !! requested backend '{chosen}' is NOT available on this stack; it may error.")

    print("=" * 68)
    print("  RIBO-SEQ FM SMOKE TEST + THROUGHPUT BENCHMARK")
    print("=" * 68)
    print(f"  device      : {dev_name}")
    print(f"  torch       : {torch.__version__}")
    if device == "cuda":
        avail_str = " ".join(f"{n}={'yes' if ok else 'NO'}" for n, ok in avail.items())
        print(f"  sdpa avail  : {avail_str}")
    print(f"  attention   : forcing '{chosen}'" +
          ("  <-- fused, no seq^2 matrix" if chosen in ("flash", "efficient")
           else "  <-- MATERIALIZED, memory-bound (flash unavailable on this build!)"))
    print(f"  preset      : {args.preset}  {cfg}")
    print(f"  seq_len     : {args.seq_len}   batch: {args.batch_size}   dtype: {args.dtype}")
    print("=" * 68)

    model = Transformer(args.vocab, args.seq_len, backend=BACKENDS[chosen], **cfg).to(device)
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
        print(f"  peak GPU memory     : {peak_gb:.1f} GB")

        # We FORCED a backend, so the story is definite -- no need to infer from memory.
        materialized_gb = (args.batch_size * cfg["n_heads"] * args.seq_len * args.seq_len
                           * 2 * cfg["n_layers"]) / 1e9
        if chosen in ("flash", "efficient"):
            print(f"  >> OK: forced '{chosen}' -- fused kernel, no seq^2 matrix. This is the real number.")
            print(f"     (materializing would have needed {materialized_gb:.1f} GB just for scores.)")
        else:
            print(f"  >> forced 'math' -- MATERIALIZED ({materialized_gb:.1f} GB of scores). Memory-bound.")
            if not avail.get("flash") and not avail.get("efficient"):
                print("  >> flash + efficient are BOTH unavailable on this torch build.")
                print("  >> Fix: install a ROCm flash-attention (aotriton/CK) build. See README.")
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
