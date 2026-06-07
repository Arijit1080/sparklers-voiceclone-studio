"""v4: cleanest path — build & load model entirely on CPU, convert to
fp16 on CPU, then ONE .to(cuda) call on the small fp16 model.

Tegra's NVML allocator hates many small CUDA allocations during init.
By keeping everything on CPU until the model is fully built AND in
half-precision (~700 MB), we hit the allocator with a single move
that we've proven works.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()

# Revert v2 patch — keep the CFM construction WITHOUT immediate .to(device).
# Then we let load_checkpoint do everything on CPU and the final move at end.
old_v2 = """    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    )
    # patched: move parameter-by-parameter to avoid the Tegra NVML
    # caching-allocator assert on the original single-call .to(device).
    if str(device).startswith("cuda"):
        import gc
        for _name, _p in list(model.named_parameters()):
            _p.data = _p.data.to(device)
            if _p.grad is not None:
                _p.grad.data = _p.grad.data.to(device)
        for _name, _b in list(model.named_buffers()):
            _b.data = _b.data.to(device) if hasattr(_b, "data") else _b.to(device)
        gc.collect(); torch.cuda.empty_cache()
    else:
        model = model.to(device)"""

new_v2 = """    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    )
    # patched: build on CPU; let load_checkpoint move the final fp16
    # model to CUDA in one shot (single allocation of ~700 MB which Tegra
    # can handle, unlike 366 small allocations that fragment the heap)."""

if old_v2 in s:
    s = s.replace(old_v2, new_v2, 1)
    print("reverted v2 patch (param-by-param)")

# Now patch load_checkpoint to do the final .to(device) move at end
old_lc = """    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()"""

new_lc = """    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    import gc; gc.collect()
    if "cuda" in str(device):
        # patched: NOW move the fully-loaded fp16 model to CUDA — one big
        # allocation instead of many. Tegra's allocator handles this cleanly.
        torch.cuda.empty_cache()
        model = model.to(device)
        torch.cuda.empty_cache()"""

if old_lc in s:
    s = s.replace(old_lc, new_lc, 1)
    print("patched load_checkpoint to move at end")

p.write_text(s)
print("done")
