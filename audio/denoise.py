"""
Reference-clip cleanup for enrollment.

F5-TTS conditions on the reference clip at every denoising step, so it
faithfully reproduces whatever is in that clip — including steady
background noise AND the artifacts of an over-aggressive denoiser. So
the policy here is deliberately conservative:

  1. high-pass at ~75 Hz on EVERY reference — removes mains hum (50/60 Hz),
     DC offset, rumble and mic-handling thumps. Near-zero artifact risk.
  2. a GENTLE spectral gate, engaged ONLY when the clip is actually noisy
     (measured SNR below a threshold). The noise profile is estimated from
     the quietest frames of the clip itself, the gain floor is shallow
     (~-8 dB, not a full kill), and the mask is smoothed in time+frequency
     to avoid the "musical noise" that aggressive subtraction leaves behind.

Everything runs at ENROLL time on a ~3 s clip, so it never touches the
synth path or the 2 s time-to-first-audio.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft
from scipy.ndimage import uniform_filter

log = logging.getLogger("audio.denoise")

# Tunables — chosen to be gentle. See module docstring for rationale.
HP_CUTOFF_HZ   = 75.0    # high-pass corner (always applied)
SNR_GATE_DB    = 18.0    # only spectral-gate clips noisier than this
PROP_DECREASE  = 0.60    # noise-bin attenuation: gain floor = 1-this (~-8 dB)
N_STD          = 1.5     # threshold = noise_mean + N_STD * noise_std (per bin)
_NFFT          = 1024
_HOP           = 256
_NOISE_PCT     = 15.0    # quietest X% of frames form the noise estimate


def _to_int16(y: np.ndarray, target_len: int | None = None) -> np.ndarray:
    if target_len is not None:
        if y.size < target_len:
            y = np.pad(y, (0, target_len - y.size))
        else:
            y = y[:target_len]
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def measure_snr_db(samples: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """Rough SNR = (90th-pctile frame level) - (10th-pctile frame level)."""
    x = samples.astype(np.float32) / 32768.0
    win = max(1, int(sr * frame_ms / 1000))
    nf = (x.size // win) * win
    if nf < win:
        return 0.0
    fr = x[:nf].reshape(-1, win)
    e = 20.0 * np.log10(np.sqrt((fr ** 2).mean(1) + 1e-12) + 1e-12)
    return float(np.percentile(e, 90) - np.percentile(e, 10))


def highpass(samples: np.ndarray, sr: int, cutoff_hz: float = HP_CUTOFF_HZ,
             order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass. Kills hum / DC / rumble."""
    if samples.size == 0:
        return samples
    x = samples.astype(np.float32) / 32768.0
    nyq = sr * 0.5
    wc = min(max(cutoff_hz / nyq, 1e-4), 0.99)
    sos = butter(order, wc, btype="highpass", output="sos")
    # padlen must be < signal length; filtfilt needs a bit of runway
    if x.size <= order * 4:
        return samples
    y = sosfiltfilt(sos, x)
    return _to_int16(y, x.size)


def gentle_denoise(samples: np.ndarray, sr: int,
                   prop_decrease: float = PROP_DECREASE,
                   n_std: float = N_STD) -> np.ndarray:
    """Conservative stationary spectral gate.

    Noise spectrum is estimated from the quietest frames of the clip; bins
    that don't clear (noise_mean + n_std*noise_std) are attenuated toward a
    shallow floor, with the gain mask smoothed in time+frequency.
    """
    x = samples.astype(np.float32) / 32768.0
    if x.size < _NFFT:
        return samples
    noverlap = _NFFT - _HOP
    _, _, Z = stft(x, fs=sr, nperseg=_NFFT, noverlap=noverlap,
                   window="hann", boundary="zeros", padded=True)
    mag = np.abs(Z)
    phase = np.angle(Z)
    mag_db = 20.0 * np.log10(mag + 1e-12)

    frame_energy = mag.sum(axis=0)
    thr_e = np.percentile(frame_energy, _NOISE_PCT)
    quiet = mag_db[:, frame_energy <= thr_e]
    if quiet.shape[1] < 3:
        quiet = mag_db                       # degenerate → estimate from all
    noise_mean = quiet.mean(axis=1, keepdims=True)
    noise_std = quiet.std(axis=1, keepdims=True)
    thresh = noise_mean + n_std * noise_std

    mask = (mag_db > thresh).astype(np.float32)
    mask = uniform_filter(mask, size=(3, 5), mode="nearest")   # freq×time smooth
    gain = mask + (1.0 - mask) * (1.0 - prop_decrease)         # floor = 1-prop

    Z2 = mag * gain * np.exp(1j * phase)
    _, y = istft(Z2, fs=sr, nperseg=_NFFT, noverlap=noverlap, window="hann")
    return _to_int16(np.asarray(y), x.size)


def clean_reference(samples: np.ndarray, sr: int,
                    snr_gate_db: float = SNR_GATE_DB,
                    hp_cutoff: float = HP_CUTOFF_HZ) -> tuple[np.ndarray, dict]:
    """High-pass always; gentle spectral gate only when SNR < snr_gate_db.

    Returns (cleaned_int16, info) where info records what ran — handy for
    progress messages and the devlog.
    """
    info: dict = {"highpass_hz": hp_cutoff, "denoised": False}
    if samples.size == 0:
        return samples, info
    try:
        s = highpass(samples, sr, cutoff_hz=hp_cutoff)
    except Exception:
        log.exception("highpass failed — using raw clip")
        s = samples
    snr = measure_snr_db(s, sr)
    info["snr_db"] = round(snr, 1)
    if snr < snr_gate_db:
        try:
            s = gentle_denoise(s, sr)
            info["denoised"] = True
        except Exception:
            log.exception("gentle_denoise failed — keeping high-passed clip")
    return s, info
