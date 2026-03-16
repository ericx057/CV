from pathlib import Path


def test_collect_result_payload_returns_new_or_changed_files(tmp_path):
    from latent_rollback_modal.modal_app import collect_result_payload

    root = tmp_path / "results"
    root.mkdir()
    before = {}
    out = root / "repobench_results" / "run1.json"
    out.parent.mkdir()
    out.write_text('{"ok": true}\n')

    payload = collect_result_payload(root, before)
    assert payload["files"]
    assert payload["files"]["repobench_results/run1.json"] == '{"ok": true}\n'


def test_write_local_result_payload_mirrors_remote_files(tmp_path):
    from latent_rollback_modal.modal_cli import write_local_result_payload

    payload = {
        "files": {
            "repobench_results/run1.json": '{"ok": true}\n',
            "integration_results/model_a_run.jsonl": '{"row": 1}\n',
        }
    }

    write_local_result_payload(payload, tmp_path)

    assert (tmp_path / "repobench_results" / "run1.json").read_text() == '{"ok": true}\n'
    assert (tmp_path / "integration_results" / "model_a_run.jsonl").read_text() == '{"row": 1}\n'
