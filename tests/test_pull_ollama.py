from ouros.pull_ollama import list_ollama_model_tags


def test_list_ollama_model_tags_dedupes_and_sorts() -> None:
    tags = list_ollama_model_tags(
        {
            "a": "ollama/llama3.1:8b",
            "b": "ollama/mistral:7b",
            "c": "ollama/llama3.1:8b",
            "d": "claude-sonnet-4",
            "e": "notollama/foo",
        }
    )

    assert tags == ["llama3.1:8b", "mistral:7b"]
