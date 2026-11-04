from pathlib import Path

import numpy as np

from rag_eval_lab.embedders import LsaEmbedder, embed_corpus


class CountingEmbedder:
    """A fake pretrained embedder that records how often it is called."""

    name = "fake"
    cacheable = True

    def __init__(self) -> None:
        self.calls = 0

    def fit(self, corpus: list[str]) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_second_call_loads_from_disk_instead_of_embedding(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    first = embed_corpus(embedder, ["a", "bb"], cache_dir=tmp_path)
    second = embed_corpus(embedder, ["a", "bb"], cache_dir=tmp_path)
    assert embedder.calls == 1
    np.testing.assert_array_equal(first, second)
    assert len(list(tmp_path.glob("fake-*.npy"))) == 1


def test_changed_texts_get_a_new_cache_entry(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    embed_corpus(embedder, ["a", "bb"], cache_dir=tmp_path)
    changed = embed_corpus(embedder, ["a", "ccc"], cache_dir=tmp_path)
    assert embedder.calls == 2
    assert changed[1, 0] == 3


def test_no_cache_dir_means_no_files(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    embed_corpus(embedder, ["a"], cache_dir=None)
    embed_corpus(embedder, ["a"], cache_dir=None)
    assert embedder.calls == 2


def test_fitted_embedders_are_never_cached(tmp_path: Path) -> None:
    # LSA learns from the corpus, so its query vectors depend on in-memory fit state;
    # a cached corpus matrix without that state would be useless.
    embedder = LsaEmbedder(dimensions=2)
    embedder.fit(["alpha beta", "beta gamma", "gamma delta"])
    embed_corpus(embedder, ["alpha beta", "beta gamma", "gamma delta"], cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
