# Contributing

Thanks for your interest in improving Mental Health Viz. This is an educational
project; contributions that improve clarity, reproducibility, or analysis quality
are welcome.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest          # dev tools
```

For the advanced analytics and the website, see the README
(`requirements-advanced.txt` and `web/`).

## Before opening a pull request

```bash
ruff check src            # lint
pytest -q                 # tests
```

- Keep changes focused and described in the PR template.
- Match the existing style (ruff-clean, type hints where practical).
- Do not commit data dumps, notebooks output, build artifacts, or secrets
  (see `.gitignore`).
- If you change a design token, update all four sources listed in `DESIGN.md`.

## Reporting issues

Use the issue templates (bug report / feature request). For anything
security-related, see `SECURITY.md`.

## Scope & ethics

This project is educational and **not** a clinical tool. Please keep contributions
consistent with that framing and avoid presenting outputs as medical advice.
