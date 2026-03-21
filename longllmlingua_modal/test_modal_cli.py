from longllmlingua_modal.modal_cli import wants_modern_transformers_image


def test_modern_transformers_image_routing_matches_supported_models():
    assert wants_modern_transformers_image("longbench", ["--models", "llama3.1-8b"])
    assert wants_modern_transformers_image("repobench", ["--models", "llama3.1-70b"])
    assert wants_modern_transformers_image("longbench", ["--models", "mistral-24b"])
    assert not wants_modern_transformers_image("longbench", ["--models", "qwen25-7b"])
    assert not wants_modern_transformers_image("suite", ["--models", "mistral-24b"])
