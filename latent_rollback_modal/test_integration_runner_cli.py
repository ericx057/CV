from __future__ import annotations


def test_integration_runner_parser_accepts_person2_shape():
    from latent_rollback_modal.benchmark_integration_runner import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--models", "deepseek-14b", "mistral-24b",
            "--tasks", "rename_3file", "add_param_chain",
            "--passes", "5",
        ]
    )

    assert args.models == ["deepseek-14b", "mistral-24b"]
    assert args.tasks == ["rename_3file", "add_param_chain"]
    assert args.passes == 5


def test_integration_runner_has_expected_defaults():
    from latent_rollback_modal.benchmark_integration_runner import build_parser

    args = build_parser().parse_args([])
    assert args.passes == 5
    assert args.injection == "all"
    assert args.fblock == "all"
    assert args.task_type == "all"
