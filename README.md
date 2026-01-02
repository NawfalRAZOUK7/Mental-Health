# Mental Health Viz

Global mental health analytics and storytelling dashboards using WHO (2021) and IHME GBD (2023) data. The project is structured as four versions that scale from static visuals to advanced analytics and an interactive risk estimator.

## Why this repo
- Turn complex public health indicators into clear, decision-friendly visuals.
- Combine BI structure, ML baselines, and data mining techniques in one cohesive workflow.
- Provide reproducible pipelines and a polished report.

## Project versions
- v0: Static visual gallery (PNG/HTML) from raw data. High variety, minimal transforms.
- v1: Main dashboard on real data (WHO + GBD) + ML baseline and BI documentation.
- v2: Advanced analytics on synthetic data (clustering, forecasting, explainability).
- v3: Risk estimator with calibration, what-if scenarios, and user inputs.

## Quick start
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard (choose a version):

```bash
python scripts/run_app.py --version v1
# or v0, v2, v3
```

Once the app is running, you can switch versions directly in the sidebar (no terminal needed).

## Build pipelines
v1 (real data):
```bash
python scripts/run_v1_pipeline.py
```

v2 (synthetic advanced analytics):
```bash
pip install -r requirements-v2.txt
python scripts/run_v2_pipeline.py
```

v3 (risk estimator features):
```bash
python scripts/run_v3_pipeline.py
```

v0 (static visual assets):
```bash
MHP_VERSION=v0 python src/v0_visuals.py
```

## Report
LaTeX report sources are in `report_latex/`.

Build the PDF:
```bash
cd report_latex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Output: `report_latex/main.pdf`

## Deploy (Streamlit Cloud)
1) Push this repository to GitHub.
2) In Streamlit Community Cloud, click **New app** → select repo/branch → main file `src/app.py`.
3) Click **Deploy**.

Notes:
- Default version is v1; users can switch in the sidebar or via `?version=v2`.
- Streamlit Cloud installs `requirements.txt` only. To enable v2/v3 advanced features, merge `requirements-v2.txt` into `requirements.txt` before deploy.
- `.streamlit/config.toml` is already included for recommended settings.

## Repository layout
- `data_raw/`: raw source data.
- `src/`: ETL, analytics, and app code.
- `scripts/`: one-command pipeline runners.
- `v0/`, `v1/`, `v2/`, `v3/`: versioned outputs (data, assets, report, notebooks).
- `report_latex/`: final report (French).

## Notes on data
- WHO and IHME GBD datasets are used for educational/academic purposes.
- v2 and v3 include synthetic data for advanced analytics demonstrations.

## Credits
- WHO suicide statistics (2021)
- IHME Global Burden of Disease (GBD 2023)

## License
- Code: MIT License (`LICENSE`).
- Report, figures, and written content: CC BY 4.0 (`LICENSE-CC-BY-4.0`).

## Citation
Plain text:
Razouk, N. (2025). *Mental Health Viz: WHO/IHME GBD dashboards and analytics*. GitHub repository. https://github.com/NawfalRAZOUK7/Mental-Health

BibTeX:
```bibtex
@misc{razouk2025mentalhealthviz,
  author = {Nawfal Razouk},
  title = {Mental Health Viz: WHO/IHME GBD dashboards and analytics},
  year = {2025},
  howpublished = {\\url{https://github.com/NawfalRAZOUK7/Mental-Health}},
  note = {Versioned dashboards (v0-v3), report, and pipelines}
}
```
