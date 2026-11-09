"""Graded relevance eval: judge the top-5 results of each query 0/1/2 with an LLM.

Compares search configurations on the same queries (eval/queries.txt). The judge is the
Claude Code CLI (`claude -p`) by default, falling back to a local Ollama model with --judge ollama.

    uv run python eval/relevance.py --configs baseline,quality
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

from seeker import search as engine
from seeker.config import get_settings

HERE = Path(__file__).parent
K = 5

CONFIGS = {
    "baseline": {"spam_filter": False, "prior_weight": 0.0},
    "spam_only": {"spam_filter": True, "prior_weight": 0.0},
    "quality": {"spam_filter": True, "prior_weight": 0.003},
    "quality_strong": {"spam_filter": True, "prior_weight": 0.008},
}

JUDGE_PROMPT = """You are grading web search results. For each query, grade each numbered result:
2 = highly relevant (directly answers / is about the query), 1 = partially relevant, 0 = not relevant or spam.
Judge only from the title, URL and excerpt. Respond with ONLY a JSON object mapping query id to a list of
{k} integers, e.g. {{"q1": [2,0,1,1,0], ...}}.

{blocks}"""


def run_config(name: str, queries: list[str]) -> dict[str, list[dict]]:
    s = get_settings()
    for key, val in CONFIGS[name].items():
        setattr(s, key, val)
    out = {}
    for i, q in enumerate(queries, 1):
        hits = engine.search(q, K, "auto")
        out[f"q{i}"] = [{"title": h.title, "url": h.url, "excerpt": h.text[:300].replace("\n", " ")} for h in hits]
    return out


def judge(queries: list[str], results: dict[str, list[dict]], backend: str) -> dict[str, list[int]]:
    blocks = []
    for i, q in enumerate(queries, 1):
        lines = [f"### q{i}: {q}"]
        for j, r in enumerate(results[f"q{i}"], 1):
            lines.append(f"{j}. {r['title']} | {r['url']}\n   {r['excerpt']}")
        blocks.append("\n".join(lines))
    prompt = JUDGE_PROMPT.format(k=K, blocks="\n\n".join(blocks))
    if backend == "claude":
        raw = subprocess.run(
            ["claude", "-p", "--output-format", "text", "--tools", ""],
            input=prompt, capture_output=True, text=True, timeout=900, check=True,
        ).stdout
    else:
        import httpx

        raw = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False, "format": "json"},
            timeout=900,
        ).json()["response"]
    grades = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return {qid: [int(g) for g in grades.get(qid, [])][: len(results[qid])] for qid in results}


def ndcg(gr: list[int]) -> float:
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gr))
    ideal = sorted(gr, reverse=True)
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal)) or 1.0
    return dcg / idcg if any(gr) else 0.0


def summarize(grades: dict[str, list[int]]) -> dict:
    all_g = [g for gs in grades.values() for g in gs]
    return {
        "P@5 (grade>=1)": round(sum(g >= 1 for g in all_g) / len(all_g), 3),
        "highly_relevant@5": round(sum(g == 2 for g in all_g) / len(all_g), 3),
        "mean_grade": round(sum(all_g) / len(all_g), 3),
        "top1_relevant": round(sum(gs[0] >= 1 for gs in grades.values() if gs) / len(grades), 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline,quality")
    ap.add_argument("--judge", default="claude", choices=["claude", "ollama"])
    a = ap.parse_args()
    queries = [q.strip() for q in (HERE / "queries.txt").read_text().splitlines() if q.strip()]
    engine.search("warm up", 1)
    report = {}
    for name in a.configs.split(","):
        results = run_config(name, queries)
        grades = judge(queries, results, a.judge)
        report[name] = summarize(grades)
        print(f"{name:15} {report[name]}", flush=True)
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"relevance_{name}_{int(time.time())}.json").write_text(
            json.dumps({"pages": engine.table().count_rows(), "summary": report[name], "results": results, "grades": grades}, indent=2)
        )
