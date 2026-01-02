#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    env = os.environ.copy()
    env["MHP_VERSION"] = "v3"

    subprocess.run([sys.executable, "src/v3_prepare_features.py"], check=True, env=env)


if __name__ == "__main__":
    main()
