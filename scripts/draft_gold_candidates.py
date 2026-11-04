"""Step 1 of building the gold set: propose candidate evidence sentences.

Heuristic: from every document, take prose sentences that look like facts worth asking
about (normative MUST/SHOULD statements, definitions, numbers), at most a few per
document so the gold set covers the whole corpus. A person then writes a paraphrased
question for the good candidates and discards the rest (step 2, manual curation).

Usage: python scripts/draft_gold_candidates.py > candidates.txt
"""

import re

from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.llm import _sentences

SIGNALS = re.compile(r"\*\*(MUST|SHOULD|MAY)|\bis an?\b|\bmeans\b|\b\d+\b")
PER_DOC = 6


def main() -> None:
    for doc in load_corpus():
        candidates = [s for s in _sentences(doc.text) if SIGNALS.search(s) and len(s) < 300]
        print(f"\n## {doc.path}")
        for sentence in candidates[:PER_DOC]:
            print(f"- {sentence}")


if __name__ == "__main__":
    main()
