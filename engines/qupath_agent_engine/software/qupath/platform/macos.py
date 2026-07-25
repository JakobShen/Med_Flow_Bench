from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterable


def _escape_applescript_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def derive_app_bundle(executable_path: Path) -> Path:
    executable = executable_path.expanduser().resolve()
    for parent in executable.parents:
        if parent.suffix.lower() == ".app":
            return parent
    raise ValueError(f"Could not derive .app bundle from executable path: {executable}")


def run_osascript(lines: Iterable[str], *, timeout: float = 5) -> str:
    args: list[str] = []
    for line in lines:
        args.extend(["-e", line])
    completed = subprocess.run(
        ["osascript", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def activate_process(process_name: str) -> None:
    escaped = _escape_applescript_string(process_name)
    run_osascript(
        [
            'tell application "System Events"',
            f'  tell process "{escaped}" to set frontmost to true',
            'end tell',
        ]
    )


def open_with_app(app_bundle: Path, targets: list[Path] | None = None) -> None:
    cmd = ["open", "-a", str(app_bundle)]
    if targets:
        cmd.extend(str(target) for target in targets)
    _run_open_command(cmd)


def open_with_app_args(app_bundle: Path, args: list[str]) -> None:
    cmd = ["open", "-a", str(app_bundle), "--args", *args]
    _run_open_command(cmd)


def _run_open_command(cmd: list[str], *, retries: int = 4, retry_sleep_seconds: float = 1.0) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(retries):
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return
        stderr_text = f"{completed.stderr or ''}\n{completed.stdout or ''}".strip()
        retryable = "error -600" in stderr_text or "_LSOpenURLsWithCompletionHandler() failed" in stderr_text
        last_error = subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )
        if not retryable or attempt >= retries - 1:
            raise last_error
        time.sleep(retry_sleep_seconds * float(attempt + 1))
    if last_error is not None:
        raise last_error


def copy_current_viewer_to_clipboard(process_name: str) -> None:
    escaped = _escape_applescript_string(process_name)
    # Preferred path: explicit QuPath menu command, which maps to
    # Edit -> Copy to clipboard… -> Current viewer.
    for menu_name in ("Copy to clipboard", "Copy to clipboard…"):
        escaped_menu_name = _escape_applescript_string(menu_name)
        try:
            run_osascript(
                [
                    'tell application "System Events"',
                    f'  tell process "{escaped}"',
                    '    set frontmost to true',
                    f'    click menu item "Current viewer" of menu 1 of menu item "{escaped_menu_name}" of menu "Edit" of menu bar item "Edit" of menu bar 1',
                    '  end tell',
                    'end tell',
                ],
                timeout=15,
            )
            return
        except subprocess.TimeoutExpired:
            return
        except Exception:
            continue

    # Fallback: send standard copy shortcut after focusing QuPath.
    try:
        run_osascript(
            [
                'tell application "System Events"',
                f'  tell process "{escaped}" to set frontmost to true',
                '  keystroke "c" using command down',
                'end tell',
            ],
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return


def app_quit_without_save(process_name: str) -> None:
    escaped = _escape_applescript_string(process_name)

    for command in (
        f'tell application "{escaped}" to quit',
        f'tell application "{escaped}" to quit saving no',
    ):
        try:
            run_osascript([command])
            return
        except Exception:
            pass

    try:
        run_osascript(
            [
                'tell application "System Events"',
                f'  tell process "{escaped}" to set frontmost to true',
                '  keystroke "q" using command down',
                'end tell',
            ]
        )
    except Exception:
        return

    button_labels = ("Don't Save", "Don’t Save", "不保存")
    try:
        script_lines = [
            'repeat 20 times',
            '  delay 0.2',
            '  tell application "System Events"',
            f'    tell process "{escaped}"',
            '      set frontmost to true',
        ]
        for label in button_labels:
            script_lines.extend(
                [
                    f'      if exists (button "{label}" of sheet 1 of window 1) then',
                    f'        click button "{label}" of sheet 1 of window 1',
                    '        return',
                    '      end if',
                    f'      if exists (button "{label}" of window 1) then',
                    f'        click button "{label}" of window 1',
                    '        return',
                    '      end if',
                ]
            )
        script_lines.extend(
            [
                '    end tell',
                '  end tell',
                'end repeat',
            ]
        )
        run_osascript(script_lines, timeout=10)
        return
    except Exception:
        pass

    try:
        run_osascript(
            [
                'delay 0.6',
                'tell application "System Events"',
                f'  tell process "{escaped}" to keystroke "d" using command down',
                'end tell',
            ],
            timeout=10,
        )
    except Exception:
        pass
