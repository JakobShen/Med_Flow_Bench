from __future__ import annotations

import subprocess
from pathlib import Path


def open_with_executable(executable_path: Path, targets: list[Path] | None = None) -> None:
    cmd = [str(executable_path)]
    if targets:
        cmd.extend(str(target) for target in targets)
    subprocess.Popen(cmd)


def activate_process(process_name: str) -> None:
    powershell = (
        "$wshell = New-Object -ComObject WScript.Shell;"
        f"[void]$wshell.AppActivate('{process_name}')"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", powershell], check=False)
