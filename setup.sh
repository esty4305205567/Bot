#!/usr/bin/env bash
# התקנה (פעם אחת): סביבה וירטואלית + חבילות + קובץ .env
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "הותקן. ערכו את .env (ANTHROPIC_API_KEY), ואז:"
echo "  source .venv/bin/activate && python build_index.py && streamlit run app.py"
