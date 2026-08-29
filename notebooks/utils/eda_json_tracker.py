import json
from pathlib import Path

RESULTS_PATH = Path("../reports/eda/Final_Data_first_data_only/summary_states.json")


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_section(section, data):
    results = load_results()
    results[section] = data
    results["last_updated_section"] = section
    save_results(results)


def add_finding(finding):
    results = load_results()
    results.setdefault("findings", []).append(finding)
    save_results(results)
