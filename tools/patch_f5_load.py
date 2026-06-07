"""Patch F5-TTS's load_checkpoint() to stage safetensors through CPU.

Tegra's CUDA caching allocator asserts when safetensors tries to allocate
the full 1.4 GB checkpoint as one batched CUDA op. Loading on CPU first
and moving per-tensor avoids the assert.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

old = '        checkpoint = load_file(ckpt_path, device=device)'
new = ('        # patched: load on CPU then move per-tensor to avoid the\n'
       '        # Tegra NVML caching-allocator assert on big safetensors\n'
       '        checkpoint = load_file(ckpt_path, device="cpu")\n'
       '        for _k, _v in list(checkpoint.items()):\n'
       '            checkpoint[_k] = _v.to(device)')

if old not in s:
    print("could not find target line — already patched?")
    sys.exit(0)

p.write_text(s.replace(old, new, 1))
print("patched", p)
