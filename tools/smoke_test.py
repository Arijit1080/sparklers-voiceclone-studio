#!/usr/bin/env python3
"""
End-to-end smoke test for the voiceclone studio.

Usage:
  python tools/smoke_test.py enroll  --name "Arijit" --seconds 30
  python tools/smoke_test.py speak   --voice arijit --text "Hello from Jetson"
  python tools/smoke_test.py list

Run from the project root with the venv active.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# allow `python tools/smoke_test.py …` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import (
    enroll_from_mic,
    list_voices,
    delete_voice,
    new_voice_key,
    speak,
)
from audio import play_wav


def _print_progress(stage: str, ratio: float) -> None:
    bar = "#" * int(ratio * 30)
    pad = " " * (30 - len(bar))
    print(f"\r[{bar}{pad}] {ratio*100:5.1f}%  {stage:<40}", end="", flush=True)


def _meter(t: float, db: float) -> None:
    bar = "#" * max(0, int((db + 60) / 2))   # -60..0 dBFS → 0..30 bars
    print(f"\r{t:5.1f}s | {db:6.1f} dBFS | {bar:<30}", end="", flush=True)


def cmd_enroll(args: argparse.Namespace) -> int:
    key = new_voice_key(args.name)
    print(f"enrolling voice key={key} display='{args.name}' seconds={args.seconds}")
    print("speak continuously after the beep:")
    res = enroll_from_mic(
        key, args.name,
        seconds=args.seconds,
        on_progress=_print_progress,
        on_meter=_meter,
    )
    print()
    print(f"OK: kept {res.seconds_kept:.1f}s of speech")
    print(f"     embedding → {res.embedding_path}")
    return 0


def cmd_speak(args: argparse.Namespace) -> int:
    print(f"synthesizing in voice='{args.voice}' nfe={args.nfe} cfg={args.cfg}")
    print(f"text: {args.text!r}")
    res = speak(args.voice, args.text, speed=args.speed,
                tau=args.cfg, nfe_step=args.nfe)
    print(f"  audio: {res.seconds:.2f}s, wall: {res.elapsed:.2f}s, rtf={res.rtf:.2f}")
    print(f"  wav:   {res.wav_path}")
    if args.play:
        print("  playing…")
        play_wav(res.wav_path)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    voices = list_voices()
    if not voices:
        print("(no voices enrolled)")
        return 0
    print(f"{'KEY':<24} {'DISPLAY':<24} {'SEC':>6} {'RMS':>7} {'VOICED':>7}")
    for v in voices:
        print(f"{v.key:<24} {v.display_name:<24} "
              f"{v.seconds:>6.1f} {v.rms_dbfs:>7.1f} {v.voiced_ratio:>7.2f}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if delete_voice(args.voice):
        print(f"deleted: {args.voice}")
        return 0
    print(f"no such voice: {args.voice}", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="record from mic + extract voice embedding")
    e.add_argument("--name", required=True)
    e.add_argument("--seconds", type=float, default=30.0)
    e.set_defaults(fn=cmd_enroll)

    s = sub.add_parser("speak", help="synthesize text in an enrolled voice")
    s.add_argument("--voice", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--nfe", type=int, default=16, help="F5-TTS flow steps (8 fast, 32 slow)")
    s.add_argument("--cfg", type=float, default=3.0, help="F5-TTS guidance strength")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--play", action="store_true", help="play the result through aplay")
    s.set_defaults(fn=cmd_speak)

    l = sub.add_parser("list", help="list enrolled voices")
    l.set_defaults(fn=cmd_list)

    d = sub.add_parser("delete", help="remove an enrolled voice")
    d.add_argument("--voice", required=True)
    d.set_defaults(fn=cmd_delete)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
