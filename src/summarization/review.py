"""
Per-paper critical review generation using the Claude API.

For each paper, identifies the top three claims and evaluates whether
the actual methods and results justify them, instructing authors on
additional experiments needed where they do not.

Reviews are stored separately from structured summaries and are used
as fine-tuning targets alongside the structured summary task.

Usage:
    reviewer = PaperReviewer(api_key="...")
    review = reviewer.review(paper_meta, fulltext_rec)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from loguru import logger
from pydantic import BaseModel, Field


# ── Output schema ────────────────────────────────────────────────────────────

class ClaimReview(BaseModel):
    claim: str = Field(description="The claim as stated or implied by the paper")
    ideal_experiments: str = Field(
        description="What experiments would be necessary to convincingly prove this claim"
    )
    paper_justification: str = Field(
        description="What the paper actually does — which methods and results bear on this claim"
    )
    verdict: str = Field(
        description="justified | partially_justified | not_justified"
    )
    required_experiments: str = Field(
        default="",
        description="Additional experiments the authors would need to run to fully justify the claim. "
                    "Leave empty if verdict is 'justified'."
    )


class PaperReview(BaseModel):
    paper_id: int
    title: str
    year: str
    journal: str
    claims: list[ClaimReview] = Field(description="Reviews of the top three claims")
    overall_assessment: str = Field(
        description="2-4 sentence synthesis: overall rigor of the paper, "
                    "most critical gaps, and the strongest contribution"
    )
    quality_score: int = Field(
        description="Overall methodological quality: 1=seriously flawed, "
                    "2=significant gaps, 3=average, 4=good, 5=excellent"
    )
    quality_rationale: str = Field(
        description="1-2 sentences justifying the quality score"
    )

    def to_dict(self) -> dict:
        return self.model_dump()


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a neuroscience expert. Write a systematic and rigorous review of this \
research article. Be precise and logically consistent. Look through the abstract, \
introduction, and discussion to identify the top three claims of the paper. Then ask \
the hypothetical: what experiments would be necessary to convincingly prove each of \
those three claims? Next, in fact, do the methods used and results provided in the \
actual paper justify the claims? If they do not, your task is to instruct the authors \
what additional experiments would be necessary to prove their claims."""

USER_PROMPT_TEMPLATE = """\
TITLE: {title}
JOURNAL: {journal} ({year})

ABSTRACT:
{abstract}

INTRODUCTION:
{introduction}

METHODS:
{methods}

RESULTS:
{results}

DISCUSSION:
{discussion}"""

REVIEW_TOOL = {
    "name": "extract_review",
    "description": "Structured critical review of a neuroscience paper",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "Reviews of the top three claims of the paper",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The claim as stated or implied by the paper",
                        },
                        "ideal_experiments": {
                            "type": "string",
                            "description": "What experiments would be necessary to convincingly prove this claim",
                        },
                        "paper_justification": {
                            "type": "string",
                            "description": "What the paper actually does — which methods and results bear on this claim",
                        },
                        "verdict": {
                            "type": "string",
                            "description": "justified | partially_justified | not_justified",
                        },
                        "required_experiments": {
                            "type": "string",
                            "description": "Additional experiments needed to fully justify the claim. "
                                           "Empty string if verdict is 'justified'.",
                        },
                    },
                    "required": ["claim", "ideal_experiments", "paper_justification", "verdict", "required_experiments"],
                },
            },
            "overall_assessment": {
                "type": "string",
                "description": "2-4 sentence synthesis: overall rigor, most critical gaps, strongest contribution",
            },
            "quality_score": {
                "type": "integer",
                "description": "1=seriously flawed (claims unsupported, major errors), "
                               "2=significant gaps (notable weaknesses), "
                               "3=average (typical published paper), "
                               "4=good (claims well-supported, minor gaps), "
                               "5=excellent (rigorous, claims fully justified)",
            },
            "quality_rationale": {
                "type": "string",
                "description": "1-2 sentences justifying the quality score",
            },
        },
        "required": ["claims", "overall_assessment", "quality_score", "quality_rationale"],
    },
}


# ── Reviewer ─────────────────────────────────────────────────────────────────

class PaperReviewer:
    """
    Calls Claude API to produce a structured critical review for each paper.
    Uses tool_use to guarantee valid JSON output.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def review(self, paper_meta: dict, fulltext_rec: dict) -> Optional[PaperReview]:
        """
        Review one paper. Returns a PaperReview or None on failure.

        paper_meta: dict from papers_metadata.jsonl
        fulltext_rec: dict from papers_fulltext.jsonl
        """
        pid = paper_meta["paper_id"]
        sections = fulltext_rec.get("sections", {})

        abstract = paper_meta.get("abstract", "") or sections.get("abstract", "")
        introduction = sections.get("introduction", sections.get("intro", ""))
        methods = sections.get("methods", "")
        results = sections.get("results", "")
        discussion = sections.get("discussion", "")

        # Fall back to full_text if sections are sparse
        if not results and not discussion:
            full_text = fulltext_rec.get("full_text", "")
            results = full_text[len(abstract):]

        if not abstract and not results:
            logger.warning(f"Paper {pid}: no usable text for review, skipping")
            return None

        user_content = USER_PROMPT_TEMPLATE.format(
            title=paper_meta.get("title", "Unknown"),
            journal=paper_meta.get("journal", "Unknown"),
            year=paper_meta.get("year", "Unknown"),
            abstract=abstract,
            introduction=introduction,
            methods=methods,
            results=results,
            discussion=discussion,
        )

        data = self._call_api(user_content, pid)
        if data is None:
            return None

        try:
            return PaperReview(
                paper_id=pid,
                title=paper_meta.get("title", ""),
                year=paper_meta.get("year", ""),
                journal=paper_meta.get("journal", ""),
                claims=[ClaimReview(**c) for c in data.get("claims", [])],
                overall_assessment=data.get("overall_assessment", ""),
                quality_score=int(data.get("quality_score", 3)),
                quality_rationale=data.get("quality_rationale", ""),
            )
        except Exception as e:
            logger.warning(f"Schema validation error for review of paper {pid}: {e}")
            return None

    def _call_api(self, user_content: str, paper_id: int) -> Optional[dict]:
        """Call Claude with tool use to get guaranteed-valid structured JSON."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=[REVIEW_TOOL],
                    tool_choice={"type": "tool", "name": "extract_review"},
                    messages=[{"role": "user", "content": user_content}],
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return block.input
                logger.warning(f"Paper {paper_id}: no tool_use block in review response")
                return None
            except anthropic.RateLimitError:
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit for paper {paper_id}. Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.warning(f"API error for paper {paper_id} (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay)
        logger.error(f"Failed to review paper {paper_id} after {self.max_retries} attempts")
        return None
