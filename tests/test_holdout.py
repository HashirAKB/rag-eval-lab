from evals.holdout_eval import HOLDOUT_PATH, render
from evals.run_eval import run
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.gold import load_gold
from rag_eval_lab.llm import MockProvider
from rag_eval_lab.pipeline import Config


def test_holdout_set_is_valid_and_disjoint_from_the_main_gold_set() -> None:
    holdout = load_gold(HOLDOUT_PATH)
    assert len(holdout) == 20 and sum(not h.answerable for h in holdout) == 10
    docs = {d.path: d.text for d in load_corpus()}
    for item in holdout:
        for quote in item.evidence:
            assert item.doc_path is not None and quote in docs[item.doc_path], item.id
    main_questions = {g.question for g in load_gold()}
    assert not main_questions & {h.question for h in holdout}


def test_render_reports_every_config() -> None:
    results = run([Config("heading", "bm25")], MockProvider(), gold=load_gold(HOLDOUT_PATH))
    # render() always reports the served config; reuse the bm25 results under its name here.
    from rag_eval_lab.pipeline import DEFAULT_CONFIG

    results["configs"][DEFAULT_CONFIG.name] = results["configs"]["heading/bm25"]
    text = render(results)
    assert "| heading/bm25 |" in text and "| h20 |" in text
