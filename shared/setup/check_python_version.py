from __future__ import annotations

import sys


SUPPORTED_VERSION = (3, 10)


def main() -> None:
    detected = sys.version_info[:2]
    if detected != SUPPORTED_VERSION:
        expected = ".".join(str(part) for part in SUPPORTED_VERSION)
        actual = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(
            f"MedFlowBench requires Python {expected}.x; "
            f"{sys.executable} is Python {actual}. "
            "Set PYTHON_BOOTSTRAP or PYTHON_BIN to a Python 3.10 executable."
        )


if __name__ == "__main__":
    main()
