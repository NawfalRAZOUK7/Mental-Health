#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Streamlit app for a version.")
    parser.add_argument("--version", default="v1", help="Version folder (v0, v1, v2, v3)")
    parser.add_argument("--port", type=int, default=None, help="Optional port override")
    args = parser.parse_args()

    env = os.environ.copy()
    env["MHP_VERSION"] = args.version

    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py"]
    if args.port:
        cmd += ["--server.port", str(args.port)]

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
