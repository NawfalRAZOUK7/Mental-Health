#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    env = os.environ.copy()
    env["MHP_VERSION"] = "v2"

    subprocess.run([sys.executable, "src/v2_generate_synth.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_validity_report.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_analytics.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_quantile.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_explainability.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_dl_forecast.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_graph_cluster.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_assoc_rules.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_ge_validate.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_kpi_benchmarks.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_trajectory.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_dtw_clusters.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_backtest.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_sensitivity.py"], check=True, env=env)
    subprocess.run([sys.executable, "src/v2_changepoints.py"], check=True, env=env)


if __name__ == "__main__":
    main()
