"""Small release checks: saved evidence, UI imports and cross-file integrity.

These checks load the packaged snapshot. They do not rerun the large studies.
"""
from hashlib import sha256
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from exercise1.ui_exercise1 import load_saved_results
from exercise2.ui_exercise2 import load_exercise2_results, load_exercise2_gamma_results
from exercise3.ui_exercise3 import load_exercise3_results
from exercise4.ui_exercise4 import load_exercise4_results, storage_replay_html
from exercise4.gas_storage_benchmark import load_storage_benchmark
from exercise4.storage_extensions import load_storage_extensions
from results_integrity import verified_result_bytes

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
PAGES = ("Project overview", "Exercise 1 · American valuation", "Exercise 2 · Control variate & gamma",
         "Exercise 3 · Euler & Milstein", "Exercise 4 · Gas storage")


@pytest.mark.parametrize("page", PAGES)
def test_packaged_results_and_ui_pages_load(page):
    # Here each view is exercised with its real published inputs. A handled
    # loader error must fail too, rather than count as a successful page render.
    if page in PAGES[:2]:
        studies, summary = load_saved_results()
        assert len(studies["table1"]) == summary["case_counts"]["table1"]
    elif page == PAGES[2]:
        assert load_exercise2_results(RESULTS / "exercise2_control_variates.json")["pricing_results"]
        assert load_exercise2_gamma_results(RESULTS / "exercise2_gamma.json")["gamma_results"]["rows"]
    elif page == PAGES[3]:
        assert load_exercise3_results(RESULTS / "exercise3_results.json")["sv_prices"]
    else:
        document, arrays = load_exercise4_results()
        assert load_storage_benchmark()["cases"]
        assert load_storage_extensions()["sampling"]
        replay = storage_replay_html(document, arrays)
        assert "/*__STORAGE_DATA__*/" not in replay
        assert all(name in replay for name in ("high_finer", "low_finer", "high_bridge"))
    app = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=30).run()
    app.radio(key="project_navigation").set_value(page).run()
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]
    assert not app.warning, [item.value for item in app.warning]
    if page == PAGES[2]:
        assert any("complete report" in item.label for item in app.get("download_button"))


@pytest.mark.parametrize("loader", [load_storage_benchmark, load_storage_extensions])
@pytest.mark.parametrize("changed_input", ["baseline", "arrays"])
def test_storage_saved_dependencies_reject_mismatched_inputs(tmp_path, loader, changed_input):
    # Here the result and companion-array relationship is checked even for a
    # new run outside the published directory and independently of the manifest.
    arrays = tmp_path / "arrays.npz"
    arrays.write_bytes(b"original saved arrays")
    arrays_hash = sha256(arrays.read_bytes()).hexdigest()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"exercise": 4, "arrays_file": arrays.name, "arrays_sha256": arrays_hash}))
    baseline_hash = sha256(baseline.read_bytes()).hexdigest()
    record = {"exercise": 4, "code_sha256": {"model.py": "0" * 64},
        "cases": [{"case": "high"}], "baseline_results_file": baseline.name,
        "baseline_results_sha256": baseline_hash, "baseline_sha256": baseline_hash,
        "baseline_arrays_sha256": arrays_hash, "grid": [{}], "sampling": [{}],
        "observations": [{}], "sobol_bridge_replay": {}}
    result = tmp_path / "study.json"
    result.write_text(json.dumps(record))
    assert loader(result, baseline) == record
    changed = baseline if changed_input == "baseline" else arrays
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="dependency integrity mismatch"):
        loader(result, baseline)


def test_all_published_result_files_are_manifested():
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    files = {p.name for p in RESULTS.iterdir() if p.suffix in (".json", ".npz") and p.name != "manifest.json"}
    assert files == set(manifest["files_sha256"])
    for name in files:
        verified_result_bytes(RESULTS / name)
