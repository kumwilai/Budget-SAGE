"""Build complete train-derived gloss plans from evaluation source text.

The output is a planning artifact for the clean compositional source branch.
Only ``id`` and source ``text`` are projected from the query trace.  TF-IDF is
fit on training captions, exact-caption donors are excluded, and every emitted
gloss token comes from the selected training donor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_POLICY = (
    "Unicode NFKC; lowercase; replace Unicode punctuation with spaces; "
    "collapse whitespace"
)


def normalize_caption(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_queries(rows: list[dict]) -> list[dict[str, str]]:
    """Discard every query field except the public inference inputs."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get("id", ""))
        text = str(row.get("text", ""))
        if not sid or sid in seen:
            raise ValueError(f"invalid or duplicate query id: {sid!r}")
        if not normalize_caption(text):
            raise ValueError(f"empty query text for {sid}")
        seen.add(sid)
        out.append({"id": sid, "text": text})
    if not out:
        raise ValueError("query trace is empty")
    return out


def project_training(rows: list[dict]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get("id", ""))
        text = str(row.get("text", ""))
        gloss = str(row.get("gloss", ""))
        if not sid or sid in seen:
            raise ValueError(f"invalid or duplicate training id: {sid!r}")
        seen.add(sid)
        if normalize_caption(text) and gloss.split():
            out.append({"id": sid, "text": text, "gloss": gloss})
    if not out:
        raise ValueError("no usable training rows")
    return out


def tfidf_features(train_texts: list[str], query_texts: list[str]):
    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    train_word = word.fit_transform(train_texts)
    query_word = word.transform(query_texts)
    train_char = char.fit_transform(train_texts)
    query_char = char.transform(query_texts)
    train = normalize(hstack([0.60 * train_word, 0.40 * train_char]).tocsr(), norm="l2")
    query = normalize(hstack([0.60 * query_word, 0.40 * query_char]).tocsr(), norm="l2")
    return train, query


def build_plans(query_rows: list[dict], train_rows: list[dict]) -> tuple[dict, list[dict]]:
    queries = project_queries(query_rows)
    training = project_training(train_rows)
    query_ids = {row["id"] for row in queries}
    train_ids = {row["id"] for row in training}
    overlap = sorted(query_ids.intersection(train_ids))
    if overlap:
        raise ValueError(f"query/training id overlap, first: {overlap[:5]}")

    train_texts = [row["text"] for row in training]
    query_texts = [row["text"] for row in queries]
    train_features, query_features = tfidf_features(train_texts, query_texts)
    plans: dict[str, list[str]] = {}
    trace: list[dict] = []
    for index, query in enumerate(queries):
        scores = (query_features.getrow(index) @ train_features.T).toarray().ravel()
        query_norm = normalize_caption(query["text"])
        exact_caption_source_ids = sorted(
            row["id"] for row in training
            if normalize_caption(row["text"]) == query_norm
        )
        order = sorted(
            range(len(training)),
            key=lambda j: (
                -float(scores[j]),
                normalize_caption(training[j]["text"]),
                training[j]["id"],
            ),
        )
        donor_index = next(
            (j for j in order if normalize_caption(training[j]["text"]) != query_norm),
            None,
        )
        if donor_index is None:
            raise ValueError(f"no non-exact training donor for {query['id']}")
        donor = training[donor_index]
        glosses = donor["gloss"].split()
        if not glosses:
            raise AssertionError(f"selected donor has no glosses: {donor['id']}")
        plans[query["id"]] = glosses
        forbidden_source_ids = sorted(
            set(exact_caption_source_ids).union({donor["id"]})
        )
        trace.append({
            "id": query["id"],
            "source": "retrieval",
            "retrieved_id": donor["id"],
            "similarity": float(scores[donor_index]),
            "query_text_sha256": hashlib.sha256(query["text"].encode()).hexdigest(),
            "donor_text_sha256": hashlib.sha256(donor["text"].encode()).hexdigest(),
            "n_gloss": len(glosses),
            "exact_caption": False,
            "exact_caption_source_ids": exact_caption_source_ids,
            "forbidden_source_ids": forbidden_source_ids,
            "forbidden_source_ids_sha256": hashlib.sha256(
                ("\n".join(forbidden_source_ids) + "\n").encode("utf-8")
            ).hexdigest(),
            "normalization_policy": NORMALIZATION_POLICY,
        })
    return plans, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_trace", required=True)
    parser.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    parser.add_argument("--plans_out", required=True)
    parser.add_argument("--query_out", required=True)
    parser.add_argument("--trace_out", required=True)
    parser.add_argument("--provenance_out", required=True)
    args = parser.parse_args()

    query_path = ROOT / args.query_trace
    train_path = ROOT / args.train_manifest
    query_rows = json.loads(query_path.read_text())
    train_rows = json.loads(train_path.read_text())
    plans, trace = build_plans(query_rows, train_rows)

    plans_path = ROOT / args.plans_out
    projected_query_path = ROOT / args.query_out
    trace_path = ROOT / args.trace_out
    provenance_path = ROOT / args.provenance_out
    for path in (plans_path, projected_query_path, trace_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    plans_path.write_text(json.dumps(plans, indent=2, ensure_ascii=False) + "\n")
    projected_query_path.write_text(
        json.dumps(project_queries(query_rows), indent=2, ensure_ascii=False) + "\n"
    )
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    provenance = {
        "method": "train-only word-and-character TF-IDF donor gloss planning",
        "query_fields_used": ["id", "text"],
        "training_fields_used": ["id", "text", "gloss"],
        "exact_caption_donors_allowed": False,
        "all_exact_caption_sources_listed_for_exclusion": True,
        "normalization_policy": NORMALIZATION_POLICY,
        "n_queries": len(plans),
        "n_training_rows": len(project_training(train_rows)),
        "empty_plans": sum(not value for value in plans.values()),
        "exact_caption_matches": sum(bool(row["exact_caption"]) for row in trace),
        "queries_with_exact_caption_training_sources": sum(
            bool(row["exact_caption_source_ids"]) for row in trace
        ),
        "exact_caption_training_source_links": sum(
            len(row["exact_caption_source_ids"]) for row in trace
        ),
        "query_trace": str(query_path),
        "query_trace_sha256": sha256(query_path),
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256(train_path),
        "plans": str(plans_path),
        "plans_sha256": sha256(plans_path),
        "projected_query": str(projected_query_path),
        "projected_query_sha256": sha256(projected_query_path),
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn_version,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "queries": len(plans),
        "empty_plans": provenance["empty_plans"],
        "exact_caption_matches": provenance["exact_caption_matches"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
