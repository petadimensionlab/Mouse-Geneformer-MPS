#!/usr/bin/env python3
"""Simulate realistic quantization options for mouse-Geneformer-L12-E20.

Fake-quantization (quantize -> dequantize in place) so the embedding-space damage of
each scheme is measurable without writing custom kernels. Sizes are exact bit budgets.

Accounting note: BertForMaskedLM ties cls.predictions.decoder.weight to
bert.embeddings.word_embeddings.weight. That single 14.36M-param table is 67% of the
model and must only be counted once.
"""
import copy, os, tempfile, time
import torch
import torch.nn as nn
from transformers import BertForMaskedLM

MODEL = "/Users/petadimensionlab/workspace/research/Mouse-Geneformer-MPS/mouse-Geneformer-L12-E20"
torch.manual_seed(0)

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])

def q_int(w, bits, dim=1):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(w / scale).clamp(-qmax - 1, qmax) * scale

def q_nf4(w, block=64):
    shape = w.shape
    flat = w.reshape(-1, block)
    absmax = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    norm = flat / absmax
    idx = torch.zeros_like(norm, dtype=torch.long)
    best = torch.full_like(norm, float("inf"))
    for i, c in enumerate(NF4):
        d = (norm - c).abs()
        m = d < best
        idx = torch.where(m, torch.full_like(idx, i), idx)
        best = torch.where(m, d, best)
    return (NF4[idx] * absmax).reshape(shape)

model = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True, output_attentions=False)
model.eval()

# ---- deduplicated accounting via named_parameters() ----
params = dict(model.named_parameters())
emb_p  = params["bert.embeddings.word_embeddings.weight"].numel()          # tied with decoder
pos_p  = sum(params[n].numel() for n in params if n.startswith("bert.embeddings.") and "word_emb" not in n)
enc_l  = sum(params[n].numel() for n in params if n.startswith("bert.encoder") and params[n].dim() == 2)
enc_o  = sum(params[n].numel() for n in params if n.startswith("bert.encoder") and params[n].dim() != 2)
mlm    = sum(params[n].numel() for n in params if n.startswith("cls."))
P = sum(p.numel() for p in params.values())
assert P == emb_p + pos_p + enc_l + enc_o + mlm, (P, emb_p + pos_p + enc_l + enc_o + mlm)
fp32 = P * 4
tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
print(f"dedup params {P/1e6:.2f}M -> {fp32/2**20:.1f} MiB fp32 | tied weight groups: "
      f"len(model.state_dict())={len(model.state_dict())} vs {len(params)} unique tensors")
print(f"  word_embeddings(+tied decoder) {emb_p/1e6:7.3f}M {100*emb_p/P:5.1f}%")
print(f"  pos/token-type emb             {pos_p/1e6:7.3f}M {100*pos_p/P:5.1f}%")
print(f"  encoder Linear                 {enc_l/1e6:7.3f}M {100*enc_l/P:5.1f}%")
print(f"  encoder bias/LayerNorm         {enc_o/1e6:7.3f}M {100*enc_o/P:5.1f}%")
print(f"  cls.* (MLM transform+bias)     {mlm/1e6:7.3f}M {100*mlm/P:5.1f}%")

L, B = 512, 16
ids = torch.randint(1, model.config.vocab_size, (B, L))
mask = torch.ones_like(ids); mask[:, 400:] = 0; ids[:, 400:] = 0

def run(m):
    with torch.no_grad():
        o = m(input_ids=ids, attention_mask=mask)
    return o.hidden_states[-1].float(), o.logits.float()

ref_h, ref_lg = run(model)
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), 0).item()

def apply(m, emb_scheme, lin_scheme, untie=False):
    m = copy.deepcopy(m)
    tie_kept = m.get_output_embeddings().weight is m.get_input_embeddings().weight
    if emb_scheme:
        w = m.get_input_embeddings().weight.data
        m.get_input_embeddings().weight.data = (q_int(w, 8, 1) if emb_scheme == 8
                                                else q_nf4(w, 64))
    if untie:                                     # bitsandbytes converts the decoder to its own Linear
        m.get_output_embeddings().weight = nn.Parameter(
            m.get_input_embeddings().weight.data.clone()).to(m.get_input_embeddings().weight.dtype)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            tied_to_emb = (name.endswith("decoder") and not untie)
            if tied_to_emb:
                continue                          # same tensor as the embedding; already handled
            if lin_scheme == 8:
                mod.weight.data = q_int(mod.weight.data, 8, 1)
            elif lin_scheme == 4:
                mod.weight.data = q_nf4(mod.weight.data, 64)
    return m, tie_kept

def budget(emb_bits, lin_bits, other_bits=16, untie_decoder=False, emb_block=None):
    b = emb_bits / 8 if emb_bits else 2
    n = emb_p * b
    n += (emb_p / emb_block) * 4 if emb_block else (model.config.vocab_size * 4)   # per-block / per-row scales
    if untie_decoder:                            # separate int4/int8 decoder Linear, not sharing the table
        n += emb_p * (lin_bits / 8) + (emb_p / 64) * 8
    if lin_bits == 4:
        n += enc_l * 4 / 8 + (enc_l / 64) * 8
    else:
        n += enc_l * lin_bits / 8 + (enc_l / 256) * 4
    n += (pos_p + enc_o + mlm) * other_bits / 8
    return n

print("\n=== simulated schemes (fidelity) ===")
print(f"  {'scheme':40s} {'size':>10s} {'%fp32':>7s} {'hidden':>10s} {'cell-emb':>10s} {'logit maxdiff':>13s}")

def ev(tag, m, nbytes, tie_kept=None):
    h, lg = run(m)
    print(f"  {tag:40s} {nbytes/2**20:6.1f}MiB {100*nbytes/fp32:6.1f}% "
          f"{cos(ref_h,h):10.6f} {cos(ref_h.mean(1),h.mean(1)):10.6f} "
          f"{(ref_lg-lg).abs().max().item():13.4e}"
          + ("" if tie_kept is None else f"  tie_kept={tie_kept}"))

print(f"  {'fp32 baseline':40s} {fp32/2**20:6.1f}MiB {100.0:6.1f}% {'1.000000':>10s} {'1.000000':>10s} {'0':>13s}")

# A) what a Linear-only quantizer does when it also unties the decoder (bnb behaviour)
m, tk = apply(model, None, 8, untie=True)
ev("A bnb int8: fp16 emb + int8 decoder+enc", m, budget(None, 8, 32, True), tk)
m, tk = apply(model, None, 4, untie=True)
ev("A bnb nf4 : fp16 emb + nf4 decoder+enc", m, budget(None, 4, 32, True), tk)

# B) full-stack: embed table quantized too
m, tk = apply(model, 8, 8)
ev("B int8 emb + int8 encoder", m, budget(8, 8))
m, tk = apply(model, 8, 4)
ev("B int8 emb + nf4 encoder", m, budget(8, 4))
m, tk = apply(model, 4, 4)
ev("B nf4 emb + nf4 encoder (all 4-bit)", m, budget(4, 4, emb_block=64))
m, tk = apply(model, 4, 8)
ev("B nf4 emb + int8 encoder", m, budget(4, 8, emb_block=64))

print("\n=== real on-disk checkpoints ===")
with tempfile.TemporaryDirectory() as td:
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        p = os.path.join(td, name)
        copy.deepcopy(model).to(dt).save_pretrained(p)
        tot = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
        print(f"  {name} save_pretrained: {tot/2**20:6.1f} MiB")
print(f"  original pytorch_model.bin: {os.path.getsize(MODEL+'/pytorch_model.bin')/2**20:.1f} MiB")

print("\n=== throughput ===")
m_cpu = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True); m_cpu.eval()
def bench(fn, n=5):
    fn(); t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n
cpu_t = bench(lambda: m_cpu(input_ids=ids, attention_mask=mask))
print(f"  CPU fp32 : {cpu_t:6.3f} s/fwd")
if torch.backends.mps.is_available():
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            m = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True)
            m.eval(); m = m.to("mps").to(dt)
            a = {"input_ids": ids.to("mps"), "attention_mask": mask.to("mps")}
            t = bench(lambda: m(**a))
            h = m(**a).hidden_states[-1].float().cpu()
            print(f"  MPS {str(dt).replace('torch.',''):8s}: {t:6.3f} s/fwd ({cpu_t/t:.1f}x vs CPU) "
                  f"hidden cos {cos(ref_h,h):.6f}")
            del m; torch.mps.empty_cache()
        except Exception as e:
            print(f"  MPS {dt}: FAILED {type(e).__name__}: {str(e)[:100]}")
print("\ndone")