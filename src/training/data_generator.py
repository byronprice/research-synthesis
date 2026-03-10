"""
Generate instruction-tuning data pairs from paper summaries using Claude API.

Categories:
  A: paper text → structured JSON summary  (~1 pair per paper)
  B: paper text → claim extraction          (~1-2 pairs per paper)
  C: paper pair → comparison/conflict       (~0.2 pairs per paper)
  D: cluster summaries → literature review  (~1 pair per cluster)

Output: data/training_data/*.jsonl in the Alpaca instruction format:
  {"instruction": "...", "input": "...", "output": "..."}

Usage:
  python scripts/06_generate_training_data.py
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger


SYSTEM = "You are a neuroscience research assistant generating training examples."

# ── Category A: summarization ─────────────────────────────────────────────

CAT_A_INSTRUCTION = (
    "Summarize this neuroscience paper as a structured JSON object with fields: "
    "key_findings (list), methods (model_system, techniques, analysis), "
    "claims (list of claim/direction/quantitative/conditions), "
    "limitations (list), open_questions (list), keywords (list)."
)

# ── Category B: claim extraction ──────────────────────────────────────────

CAT_B_INSTRUCTION = (
    "Extract the primary empirical claims from this neuroscience paper. "
    "For each claim, note its direction (increase/decrease/no_effect/bidirectional/unclear), "
    "any quantitative values, and the experimental conditions."
)

# ── Category C: comparison ────────────────────────────────────────────────

CAT_C_INSTRUCTION = (
    "Compare these two neuroscience papers. What do they agree on? "
    "What findings are inconsistent or contradictory? "
    "What different methods might explain any discrepancies?"
)

# ── Category D: literature review ─────────────────────────────────────────

CAT_D_INSTRUCTION = (
    "Write a concise literature review paragraph (200-350 words) synthesizing "
    "the findings from these neuroscience papers. Use academic prose and cite "
    "papers as (Author, Year)."
)


def _paper_context(meta: dict, summary: dict, fulltext: Optional[dict] = None) -> str:
    abstract = meta.get("abstract", "")
    results = ""
    if fulltext:
        results = fulltext.get("sections", {}).get("results", "")[:2000]
    lines = [
        f"Title: {meta.get('title', '')}",
        f"Journal: {meta.get('journal', '')} ({meta.get('year', '')})",
        f"Abstract: {abstract[:1500]}",
    ]
    if results:
        lines.append(f"Results excerpt: {results}")
    return "\n".join(lines)


class TrainingDataGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        delay: float = 0.1,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.delay = delay

    def generate_category_a(self, meta: dict, fulltext: Optional[dict], summary: dict) -> Optional[dict]:
        """Generate a summarization instruction pair."""
        context = _paper_context(meta, summary, fulltext)
        output_json = json.dumps({
            "key_findings": summary.get("key_findings", []),
            "methods": summary.get("methods", {}),
            "claims": summary.get("claims", []),
            "limitations": summary.get("limitations", []),
            "open_questions": summary.get("open_questions", []),
            "keywords": summary.get("keywords", []),
        }, indent=2)
        return {
            "instruction": CAT_A_INSTRUCTION,
            "input": context,
            "output": output_json,
            "category": "A",
        }

    def generate_category_b(self, meta: dict, summary: dict) -> Optional[dict]:
        """Generate a claim extraction pair (direct from existing summary)."""
        claims = summary.get("claims", [])
        if not claims:
            return None
        output_lines = []
        for c in claims:
            line = f"- [{c.get('direction','?').upper()}] {c.get('claim','')}"
            if c.get("quantitative"):
                line += f" ({c['quantitative']})"
            if c.get("conditions"):
                line += f" | Conditions: {c['conditions']}"
            output_lines.append(line)

        context = _paper_context(meta, summary)
        return {
            "instruction": CAT_B_INSTRUCTION,
            "input": context,
            "output": "\n".join(output_lines),
            "category": "B",
        }

    def generate_category_c(
        self, meta_a: dict, summary_a: dict, meta_b: dict, summary_b: dict
    ) -> Optional[dict]:
        """Generate a comparison pair via API (requires novel synthesis)."""
        input_text = (
            f"PAPER A: {meta_a.get('title','')} ({meta_a.get('year','')})\n"
            f"Findings: {'; '.join(summary_a.get('key_findings',[])[:4])}\n\n"
            f"PAPER B: {meta_b.get('title','')} ({meta_b.get('year','')})\n"
            f"Findings: {'; '.join(summary_b.get('key_findings',[])[:4])}"
        )

        output = self._call_api(CAT_C_INSTRUCTION, input_text, max_tokens=600)
        if not output:
            return None
        return {
            "instruction": CAT_C_INSTRUCTION,
            "input": input_text,
            "output": output,
            "category": "C",
        }

    def generate_category_d(self, cluster_meta: dict) -> Optional[dict]:
        """Generate a literature review paragraph for a cluster."""
        topic = cluster_meta.get("topic_label", "neuroscience")
        consensus = cluster_meta.get("consensus", [])
        trends = cluster_meta.get("trends", [])

        input_text = (
            f"Topic: {topic}\n"
            f"Consensus findings: {'; '.join(consensus[:5])}\n"
            f"Trends: {'; '.join(trends[:3])}\n"
            f"Paper count: {cluster_meta.get('paper_count', '?')}"
        )

        output = self._call_api(CAT_D_INSTRUCTION, input_text, max_tokens=500)
        if not output:
            return None
        return {
            "instruction": CAT_D_INSTRUCTION,
            "input": input_text,
            "output": output,
            "category": "D",
        }

    def _call_api(self, instruction: str, context: str, max_tokens: int = 600) -> Optional[str]:
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": f"{instruction}\n\n{context}",
                    }],
                )
                time.sleep(self.delay)
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                time.sleep(5 * (2 ** attempt))
            except Exception as e:
                logger.warning(f"API error: {e}")
                time.sleep(2)
        return None
