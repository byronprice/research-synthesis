"""
Per-paper structured summarization using the Claude API.

Produces a structured JSON summary for each paper with:
  key_findings, methods, claims (with direction), limitations, keywords.

The structured output format propagates through clustering, contradiction
detection, and training data generation — keep the schema stable.

Usage:
    summarizer = PaperSummarizer(api_key="...")
    summary = summarizer.summarize(paper_meta, fulltext_rec)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ── Output schema (validated via Pydantic) ──────────────────────────────────

class MethodsInfo(BaseModel):
    model_system: str = Field(
        description="Experimental model (mouse, rat, human, non-human_primate, in_vitro, computational, other)"
    )
    techniques: list[str] = Field(description="Recording/imaging/stimulation techniques used")
    analysis: list[str] = Field(description="Analysis methods (e.g. GLM, PCA, decoding)")


class Claim(BaseModel):
    claim: str = Field(description="A single specific empirical claim made by the paper")
    direction: str = Field(
        description="Direction: increase | decrease | no_effect | bidirectional | unclear"
    )
    quantitative: str = Field(
        default="", description="Quantitative details if available (e.g. '37%, p<0.01')"
    )
    conditions: str = Field(
        default="", description="Experimental conditions or constraints on the claim"
    )

    @field_validator("quantitative", "conditions", mode="before")
    @classmethod
    def coerce_none(cls, v):
        return v if v is not None else ""


class PaperSummary(BaseModel):
    paper_id: int
    title: str
    year: str
    journal: str
    key_findings: list[str] = Field(description="3-7 key empirical findings")
    methods: MethodsInfo
    claims: list[Claim] = Field(description="2-6 specific directional claims")
    limitations: list[str] = Field(description="1-4 limitations stated or implied")
    open_questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(description="5-10 domain keywords")


# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert neuroscience research assistant. Your task is to extract \
structured information from a neuroscience paper. Be precise and factual. \
Do not infer findings not stated in the text. Use neuroscience field terminology.

Return ONLY a JSON object matching the requested schema — no preamble, \
no explanation, no markdown fences."""

USER_PROMPT_TEMPLATE = """\
Extract structured information from this neuroscience paper.

TITLE: {title}
JOURNAL: {journal} ({year})
COLLECTIONS: {collections}

ABSTRACT:
{abstract}

RESULTS (excerpt):
{results}

DISCUSSION (excerpt):
{discussion}

Return a JSON object with this exact structure:
{{
  "key_findings": ["finding 1", "finding 2", ...],
  "methods": {{
    "model_system": "mouse|rat|human|non-human_primate|in_vitro|computational|other",
    "techniques": ["calcium imaging", "patch clamp", ...],
    "analysis": ["dimensionality reduction", "GLM", ...]
  }},
  "claims": [
    {{
      "claim": "specific empirical claim",
      "direction": "increase|decrease|no_effect|bidirectional|unclear",
      "quantitative": "37% increase (p<0.01)",
      "conditions": "head-fixed mice, visual stimulation"
    }}
  ],
  "limitations": ["limitation 1", ...],
  "open_questions": ["question 1", ...],
  "keywords": ["keyword1", "keyword2", ...]
}}"""


# ── Summarizer ───────────────────────────────────────────────────────────────

class PaperSummarizer:
    """
    Calls Claude API to produce a structured summary for each paper.

    Uses claude-haiku for cost efficiency (~$0.004/paper).
    Switch to claude-sonnet for higher quality at ~8x the cost.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def summarize(self, paper_meta: dict, fulltext_rec: dict) -> Optional[PaperSummary]:
        """
        Summarize one paper. Returns a PaperSummary or None on failure.

        paper_meta: dict from papers_metadata.jsonl
        fulltext_rec: dict from papers_fulltext.jsonl
        """
        pid = paper_meta["paper_id"]
        sections = fulltext_rec.get("sections", {})

        # Prefer Zotero abstract; fall back to extracted
        abstract = paper_meta.get("abstract", "") or sections.get("abstract", "")
        results = sections.get("results", "")[:3000]
        discussion = sections.get("discussion", "")[:2000]

        # If we have neither results nor discussion, try full_text excerpt
        if not results and not discussion:
            full_text = fulltext_rec.get("full_text", "")
            results = full_text[len(abstract):len(abstract) + 3000]

        if not abstract and not results:
            logger.warning(f"Paper {pid}: no usable text, skipping")
            return None

        user_content = USER_PROMPT_TEMPLATE.format(
            title=paper_meta.get("title", "Unknown"),
            journal=paper_meta.get("journal", "Unknown"),
            year=paper_meta.get("year", "Unknown"),
            collections=", ".join(paper_meta.get("collections", [])) or "Uncategorized",
            abstract=abstract[:2000],
            results=results,
            discussion=discussion,
        )

        raw_json = self._call_api(user_content, pid)
        if raw_json is None:
            return None

        return self._parse_response(raw_json, paper_meta)

    def _call_api(self, user_content: str, paper_id: int) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit for paper {paper_id}. Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.warning(f"API error for paper {paper_id} (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay)
        logger.error(f"Failed to summarize paper {paper_id} after {self.max_retries} attempts")
        return None

    def _parse_response(self, raw: str, meta: dict) -> Optional[PaperSummary]:
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for paper {meta['paper_id']}: {e}")
            return None

        try:
            return PaperSummary(
                paper_id=meta["paper_id"],
                title=meta.get("title", ""),
                year=meta.get("year", ""),
                journal=meta.get("journal", ""),
                key_findings=data.get("key_findings", []),
                methods=MethodsInfo(**data.get("methods", {"model_system": "other", "techniques": [], "analysis": []})),
                claims=[Claim(**c) for c in data.get("claims", [])],
                limitations=data.get("limitations", []),
                open_questions=data.get("open_questions", []),
                keywords=data.get("keywords", []),
            )
        except Exception as e:
            logger.warning(f"Schema validation error for paper {meta['paper_id']}: {e}")
            return None
