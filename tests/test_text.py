from rag_eval_lab.text import tokenize


def test_tokenize_lowercases_and_drops_stopwords_and_punctuation() -> None:
    assert tokenize("How do I list the Tools? Use tools/list!") == [
        "list",
        "tools",
        "tools",
        "list",
    ]


def test_tokenize_keeps_identifiers_and_numbers() -> None:
    assert tokenize("notifications/cancelled after 30 seconds (HTTP 404)") == [
        "notifications",
        "cancelled",
        "after",
        "30",
        "seconds",
        "http",
        "404",
    ]
