from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        "/usr/bin/python3",
        "-c",
        (
            "import sys, site; "
            "print('executable=', sys.executable); "
            "print('user_site=', site.getusersitepackages()); "
            "import wan; "
            "print('wan_file=', wan.__file__)"
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"returncode={result.returncode}")
    print("stdout:")
    print(result.stdout.rstrip() or "<empty>")
    print("stderr:")
    print(result.stderr.rstrip() or "<empty>")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
