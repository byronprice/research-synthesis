"""
LLM verification of contradiction candidates.

Takes each (claim_a, claim_b) pair and asks Claude to assess whether
it's a TRUE_CONTRADICTION, APPARENT_CONTRADICTION (different conditions),
or NOT_CONTRADICTORY.

Usage:
    verifier = ContradictionVerifier(api_key="...")
    results = verifier.verify_batch(candidates)
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import anthropic
from loguru import logger

from src.contradiction.candidate_finder import ContraCandidate


SYSTEM_PROMPT = """\
You are a neuroscience expert. Assess whether two empirical claims from
different papers are genuinely contradictory. Consider that apparent
contradictions may arise from different model systems, techniques,
experimental conditions, or brain regions. Be conservative — only call
something a TRUE_CONTRADICTION if both claims concern the same phenomenon
under comparable conditions and reach incompatible conclusions."""

VERIFY_PROMPT = """\
Are these two neuroscience claims contradictory?

CLAIM A (from "{title_a}", {year_a}):
"{claim_a}"
Direction: {dir_a}
Details: {quant_a}
Conditions: {cond_a}

CLAIM B (from "{title_b}", {year_b}):
"{claim_b}"
Direction: {dir_b}
Details: {quant_b}
Conditions: {cond_b}

Verdict options:
- TRUE_CONTRADICTION: same phenomenon, comparable conditions, incompatible conclusions
- APPARENT_CONTRADICTION: different model systems, brain regions, or conditions explain the difference
- NOT_CONTRADICTORY: claims are actually compatible or one does not negate the other

Return JSON:
{{
  "verdict": "TRUE_CONTRADICTION|APPARENT_CONTRADICTION|NOT_CONTRADICTORY",
  "explanation": "brief explanation (2-3 sentences)",
  "reason_for_difference": "what might explain the discrepancy (if any)"
}}"""


@dataclass
class VerifiedContradiction:
    candidate: ContraCandidate
    verdict: str          # TRUE_CONTRADICTION | APPARENT_CONTRADICTION | NOT_CONTRADICTORY
    explanation: str
    reason_for_difference: str

    def to_dict(self) -> dict:
        d = self.candidate.to_dict()
        d["verdict"] = self.verdict
        d["explanation"] = self.explanation
        d["reason_for_difference"] = self.reason_for_difference
        return d


class ContradictionVerifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        delay: float = 0.1,
        max_retries: int = 3,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.delay = delay
        self.max_retries = max_retries

    def verify_batch(
        self,
        candidates: list[ContraCandidate],
        max_candidates: int = 500,
    ) -> list[VerifiedContradiction]:
        """
        Verify up to max_candidates contradiction candidates.
        Returns only TRUE_CONTRADICTION and APPARENT_CONTRADICTION results.
        """
        if len(candidates) > max_candidates:
            logger.info(f"Limiting to top {max_candidates} candidates by similarity")
            candidates = candidates[:max_candidates]

        results = []
        for i, cand in enumerate(candidates):
            result = self._verify_one(cand)
            if result and result.verdict != "NOT_CONTRADICTORY":
                results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"Verified {i+1}/{len(candidates)} candidates, {len(results)} contradictions found")
            time.sleep(self.delay)

        logger.info(f"Verification complete: {len(results)} contradictions from {len(candidates)} candidates")
        return results

    def _verify_one(self, cand: ContraCandidate) -> Optional[VerifiedContradiction]:
        a, b = cand.claim_a, cand.claim_b
        user_content = VERIFY_PROMPT.format(
            title_a=a.title, year_a=a.year,
            claim_a=a.claim_text, dir_a=a.direction,
            quant_a=a.quantitative or "not specified",
            cond_a=a.conditions or "not specified",
            title_b=b.title, year_b=b.year,
            claim_b=b.claim_text, dir_b=b.direction,
            quant_b=b.quantitative or "not specified",
            cond_b=b.conditions or "not specified",
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                data = json.loads(raw)
                return VerifiedContradiction(
                    candidate=cand,
                    verdict=data.get("verdict", "NOT_CONTRADICTORY"),
                    explanation=data.get("explanation", ""),
                    reason_for_difference=data.get("reason_for_difference", ""),
                )
            except anthropic.RateLimitError:
                time.sleep(5 * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Error verifying candidate: {e}")
                time.sleep(2)
        return None
