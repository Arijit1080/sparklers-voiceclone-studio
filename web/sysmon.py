"""Background system-resource monitor.

Spawns `tegrastats --interval 1000` as a long-lived subprocess and parses
each line into a dict.  Invokes `on_update(stats)` once per second.

Stops cleanly: `stop()` terminates the subprocess and joins the thread.
"""

from __future__ import annotations

import re
import subprocess
import threading
from typing import Callable, Optional

# Regexes pre-compiled (this runs in a hot loop)
_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"CPU \[([^\]]+)\]")
_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_T_CPU = re.compile(r"cpu@([\d.]+)C", re.IGNORECASE)
_T_GPU = re.compile(r"gpu@([\d.]+)C", re.IGNORECASE)
_T_TJ = re.compile(r"tj@([\d.]+)C", re.IGNORECASE)
_PWR_IN = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")
_PWR_CG = re.compile(r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW")


def _parse(line: str) -> dict:
    out: dict = {}
    m = _RAM.search(line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))
    m = _SWAP.search(line)
    if m:
        out["swap_used_mb"] = int(m.group(1))
        out["swap_total_mb"] = int(m.group(2))
    m = _CPU.search(line)
    if m:
        cores: list[int] = []
        for part in m.group(1).split(","):
            pm = re.match(r"\s*(\d+)%", part)
            if pm:
                cores.append(int(pm.group(1)))
        if cores:
            out["cpu_per_core"] = cores
            out["cpu_avg_pct"] = round(sum(cores) / len(cores), 1)
    m = _GPU.search(line)
    if m:
        out["gpu_pct"] = int(m.group(1))
    m = _T_CPU.search(line)
    if m:
        out["cpu_temp_c"] = float(m.group(1))
    m = _T_GPU.search(line)
    if m:
        out["gpu_temp_c"] = float(m.group(1))
    m = _T_TJ.search(line)
    if m:
        out["tj_temp_c"] = float(m.group(1))
    m = _PWR_IN.search(line)
    if m:
        out["power_in_mw"] = int(m.group(1))
        out["power_in_avg_mw"] = int(m.group(2))
    m = _PWR_CG.search(line)
    if m:
        out["power_cpu_gpu_mw"] = int(m.group(1))
    return out


class SystemMonitor:
    """Tegrastats reader.  on_update is called from the parser thread
    every ~1 s with a dict of fields (see `_parse`).  Missing fields just
    aren't included (e.g. older JetPack might not expose every stat)."""

    def __init__(self, on_update: Callable[[dict], None],
                 interval_ms: int = 1000) -> None:
        self._on_update = on_update
        self._interval_ms = interval_ms
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> bool:
        if self._thread is not None:
            return True
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="SysMon", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try: self._proc.terminate()
            except Exception: pass    # noqa: BLE001, E701
            try: self._proc.wait(timeout=2.0)
            except Exception: pass    # noqa: BLE001, E701
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            stats = _parse(line)
            if stats:
                try:
                    self._on_update(stats)
                except Exception:    # noqa: BLE001
                    pass
