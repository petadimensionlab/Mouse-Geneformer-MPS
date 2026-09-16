#!/usr/bin/env python3
"""Quantization feasibility report for mouse-Geneformer-L12-E20.

1. Where do the parameters live? (what a Linear-only quantizer can even touch)
2. How much do fp32 / bf16 / int8-dynamic actually shrink, and what do they cost
   in latency?
3. What happens to the hidden states Geneformer's EmbExtractor reads?
"""
import copy, time
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import BertForMaskedLM

MODEL = "/Users/petadimensionlab/workspace/research/Mouse-Geneformer-MPS/mouse-Geneformer-L12-E20"
torch.manual_seed(0)
torch.set_num_threads(torch.get_num_threads())

model = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True, output_attentions=False)
model.eval()
cfg = model.config
print(f"arch={cfg.architectures} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
      f"heads={cfg.num_attention_heads} vocab={cfg.vocab_size} max_pos={cfg.max_position_embeddings}")
print("tie_word_embeddings:", cfg.tie_word_embeddings, "| lm_head shares storage with input emb:",
      model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
print("mps available:", torch.backends.mps.is_available())

# ---------------- 1. parameter breakdown ----------------
def bucket(n):
    if "word_embeddings" in n:
        return "word_embeddings (vocab x hidden)"
    if n.startswith("bert.embeddings"):
        return "position/token_type emb"
    if n.startswith("bert.encoder"):
        return "encoder Linear w (attn/dense)" if n.endswith("weight") else "encoder bias/LayerNorm"
    if n.startswith("cls"):
        return "cls.predictions (MLM head)"
    if n.startswith("bert.pooler"):
        return "pooler"
    return "other"

agg, tot_params = defaultdict(int), 0
for n, p in model.named_parameters():
    agg[bucket(n)] += p.numel()
    tot_params += p.numel()
fp32_bytes = tot_params * 4

print(f"\n=== 1. parameter breakdown (total {tot_params/1e6:.2f}M, fp32 = {fp32_bytes/2**20:.1f} MiB) ===")
linear_params = 0
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {k:30s} {v/1e6:8.3f}M  {100*v/tot_params:5.1f}%  {v*4/2**20:7.1f} MiB fp32")
    if "encoder Linear" in k:
        linear_params = v

print(f"\n  params a Linear-only quantizer can touch : {linear_params/1e6:.3f}M "
      f"({100*linear_params/tot_params:.1f}%)")
print(f"  untouched (embeddings + norms, stay fp16/32): "
      f"{(tot_params-linear_params)/1e6:.3f}M ({100*(tot_params-linear_params)/tot_params:.1f}%)")

def total_bytes(lin_bits, other_bits=32):
    return linear_params * lin_bits / 8 + (tot_params - linear_params) * other_bits / 8

print("\n  best case if ONLY Linear weights are quantized (what bnb / quantize_dynamic do):")
for bits, name in ((8, "int8 Linear"), (4, "nf4 Linear")):
    for ob, oname in ((16, "others bf16"), (32, "others fp32")):
        tb = total_bytes(bits, ob)
        print(f"    {name:11s} + {oname:9s} -> {tb/2**20:5.1f} MiB "
              f"({100*tb/fp32_bytes:4.1f}% of fp32, {fp32_bytes/tb:.2f}x smaller)")
print("  for reference, quantizing EVERYTHING including the embedding table:")
for bits in (8, 4):
    tb = tot_params * bits / 8
    print(f"    all-{bits}-bit -> {tb/2**20:5.1f} MiB ({100*tb/fp32_bytes:4.1f}%, {fp32_bytes/tb:.2f}x)")

# ---------------- 2/3. fidelity + latency ----------------
L, B = 512, 16
ids = torch.randint(1, cfg.vocab_size, (B, L))
mask = torch.ones_like(ids); mask[:, 400:] = 0; ids[:, 400:] = 0

def run(m):
    with torch.no_grad():
        o = m(input_ids=ids, attention_mask=mask)
    return o.hidden_states[-1].float(), o.logits.float()

def timeit(m, n=3):
    run(m)
    t0 = time.time()
    for _ in range(n):
        run(m)
    return (time.time() - t0) / n

ref_h, ref_logits = run(model)
t_fp32 = timeit(model)
print(f"\n=== 2/3. fidelity + latency (batch {B} x {L} tokens, CPU, "
      f"{torch.get_num_threads()} threads) ===")
print(f"  {'variant':26s} {'size MiB':>9s} {'%fp32':>6s} {'s/fwd':>7s} "
      f"{'hidden cos':>11s} {'cell-emb cos':>13s} {'logit maxdiff':>13s}")

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()

def report(tag, h, lg, nbytes, secs):
    print(f"  {tag:26s} {nbytes/2**20:9.1f} {100*nbytes/fp32_bytes:6.1f} {secs:7.2f} "
          f"{cos(ref_h, h):11.6f} {cos(ref_h.mean(1), h.mean(1)):13.6f} "
          f"{(ref_logits-lg).abs().max().item():13.4e}")

report("fp32 (baseline)", ref_h, ref_logits, fp32_bytes, t_fp32)

m_bf16 = copy.deepcopy(model).to(torch.bfloat16)
h, lg = run(m_bf16)
report("bfloat16 cast", h, lg, tot_params * 2, timeit(m_bf16))

try:
    m_fp16 = copy.deepcopy(model).half()
    h, lg = run(m_fp16)
    report("float16 cast", h, lg, tot_params * 2, timeit(m_fp16))
except Exception as e:
    print(f"  float16 on CPU unsupported here: {type(e).__name__}: {str(e)[:80]}")

try:
    from torch.ao.quantization import quantize_dynamic
    qc = quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
    qmods = [m for m in qc.modules() if isinstance(m, nn.quantized.dynamic.Linear)]
    q_params = sum(m.weight().numel() for m in qmods)
    rest = sum(p.numel() * p.element_size() for p in qc.parameters())
    nbytes = rest + q_params * 1 + len(qmods) * 8
    h, lg = run(qc)
    report("dynamic int8 (Linear)", h, lg, nbytes, timeit(qc))
    print(f"     -> {len(qmods)} Linear layers converted, {q_params/1e6:.3f}M params at int8. "
          "Stock torch = no extra dependency, CPU only.")
except Exception as e:
    print(f"  dynamic int8 failed: {type(e).__name__}: {str(e)[:150]}")

try:
    import bitsandbytes as bnb
    print(f"\n  bitsandbytes present: {bnb.__version__} (backend: {getattr(bnb, '__file__', '')})")
except ImportError:
    print("\n  bitsandbytes NOT installed -> 4-bit/8-bit QLoRA path not measured here")

# ---------------- MPS check ----------------
if torch.backends.mps.is_available():
    try:
        mm = copy.deepcopy(model).to("mps")
        t = timeit(mm)
        print(f"\n=== MPS (float32) forward: {t:.2f}s vs CPU {t_fp32:.2f}s "
              f"({t_fp32/t:.2f}x speedup) ===")
        mmh = copy.deepcopy(model).to("mps").to(torch.float16)
        o = mmh(input_ids=ids.to("mps"), attention_mask=mask.to("mps"))
        hh = o.hidden_states[-1].float().cpu()
        print(f"  MPS float16 hidden-state cos vs CPU fp32: {cos(ref_h, hh):.6f}")
        print(f"  MPS float16 fwd: {timeit(mmh):.2f}s")
    except Exception as e:
        print(f"\n  MPS run failed: {type(e).__name__}: {str(e)[:150]}")

print("\n=== embedding-table structure (relevant to quantizing it directly) ===")
ew = model.get_input_embeddings().weight.data.float()
print(f"  shape {tuple(ew.shape)} | absmax {ew.abs().max():.3f} | std {ew.std():.4f} "
      f"| per-row absmax mean {ew.abs().amax(dim=1).mean():.3f}")
sub = torch.nn.functional.normalize(ew[torch.randperm(ew.shape[0])[:400]], dim=1)
sim = sub @ sub.T
off = sim[~torch.eye(400, dtype=torch.bool)]
print(f"  400 random genes: mean pairwise cos {off.mean():.4f}, p95 {off.quantile(0.95):.4f} "
      f"-> {'HIGH' if off.mean() > 0.2 else 'LOW'} redundancy")

print("\n=== low-rank structure of encoder Linear weights (LoRA suitability) ===")
for li in (0, 11):
    w = model.bert.encoder.layer[li].attention.self.query.weight.data.float()
    s = torch.linalg.svdvals(w)
    e = torch.cumsum(s**2, 0) / (s**2).sum()
    print(f"  layer{li:2d} query proj: rank 8={100*e[7]:5.1f}% 16={100*e[15]:5.1f}% "
          f"32={100*e[31]:5.1f}% of spectral energy")
print("\ndone")