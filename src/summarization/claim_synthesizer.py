"""
Per-cluster claim synthesis using the Claude API.

For each claim cluster, synthesizes the evidence across papers to produce:
  - A consensus position
  - Boundary conditions (when/where the effect holds)
  - What is contested
  - Assessment of overall evidence quality
  - Emergent hypotheses: new claims suggested by the pattern but not stated in any paper
  - Cross-domain insight (when the cluster spans multiple Zotero collections)

Outputs are stored in data/clusters/{cluster_id}.json and are indexed
into Qdrant so emergent hypotheses are retrievable in Q&A.

Usage:
    synthesizer = ClaimSynthesizer(api_key="...")
    synthesis = synthesizer.synthesize(cluster_id, claims)
"""

import time
from typing import Optional

import anthropic
from loguru import logger
from pydantic import BaseModel, Field


# ── Output schema ─────────────────────────────────────────────────────────────

class ClusterSynthesis(BaseModel):
    cluster_id: int
    n_claims: int
    n_papers: int
    collections_spanned: list[str]
    consensus: str = Field(
        description="What this cluster of claims collectively establishes — the shared finding"
    )
    boundary_conditions: str = Field(
        description="Conditions under which the effect holds or breaks down: "
                    "species, brain region, technique, behavioral state, etc."
    )
    contested: str = Field(
        description="What is disputed or inconsistent within this cluster, "
                    "and what might explain the disagreements"
    )
    evidence_quality_note: str = Field(
        description="Assessment of the overall evidential strength: "
                    "are claims well-supported, partially justified, or mostly speculative?"
    )
    emergent_hypotheses: list[str] = Field(
        description="1-3 new claims suggested by the collective pattern but not explicitly "
                    "stated in any individual paper. Clearly inferential, not established fact."
    )
    cross_domain_insight: str = Field(
        default="",
        description="If the cluster spans multiple brain regions or research domains, "
                    "what structural parallel or shared mechanism does this suggest?"
    )
    keywords: list[str] = Field(description="5-8 keywords characterizing this cluster")

    def to_dict(self) -> dict:
        return self.model_dump()


# ── Tool schema ───────────────────────────────────────────────────────────────

SYNTHESIS_TOOL = {
    "name": "synthesize_claim_cluster",
    "description": "Synthesize a cluster of related neuroscience claims across papers",
    "input_schema": {
        "type": "object",
        "properties": {
            "consensus": {
                "type": "string",
                "description": "What this cluster of claims collectively establishes",
            },
            "boundary_conditions": {
                "type": "string",
                "description": "Conditions under which the finding holds or breaks down "
                               "(species, brain region, technique, behavioral state, etc.)",
            },
            "contested": {
                "type": "string",
                "description": "What is disputed or inconsistent within the cluster, "
                               "and what might explain disagreements",
            },
            "evidence_quality_note": {
                "type": "string",
                "description": "Overall evidential strength: well-supported, partially justified, or speculative",
            },
            "emergent_hypotheses": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 new claims suggested by the collective pattern but not "
                               "stated in any individual paper. Must be clearly inferential.",
            },
            "cross_domain_insight": {
                "type": "string",
                "description": "If the cluster spans multiple domains, what structural parallel "
                               "or shared mechanism does this suggest? Empty string if not applicable.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "5-8 keywords characterizing this cluster",
            },
        },
        "required": [
            "consensus", "boundary_conditions", "contested",
            "evidence_quality_note", "emergent_hypotheses",
            "cross_domain_insight", "keywords",
        ],
    },
}

SYSTEM_PROMPT = """\
You are an expert neuroscience researcher performing cross-paper synthesis. \
You will be given a set of related claims drawn from multiple research papers. \
Your task is to synthesize these claims into a coherent picture of what the \
collective evidence establishes, where the boundaries are, what remains contested, \
and — most importantly — what new hypotheses the pattern of evidence suggests that \
no individual paper has explicitly stated. Be scientifically rigorous: clearly \
distinguish established consensus from speculative inference. Emergent hypotheses \
must be grounded in the evidence presented and clearly flagged as inferential."""

MAX_CLAIMS_PER_CALL = 40


def _format_claim_block(claim) -> str:
    """Format a single UnifiedClaim for the synthesis prompt."""
    parts = [f"  Claim: {claim.claim_text}"]
    if claim.source == "summary":
        if claim.direction:
            parts.append(f"  Direction: {claim.direction}")
        if claim.quantitative:
            parts.append(f"  Quantitative: {claim.quantitative}")
        if claim.conditions:
            parts.append(f"  Conditions: {claim.conditions}")
    elif claim.source == "review":
        if claim.verdict:
            parts.append(f"  Evidential verdict: {claim.verdict}")
        if claim.paper_justification:
            parts.append(f"  What the paper does: {claim.paper_justification}")
        if claim.required_experiments:
            parts.append(f"  Still needed: {claim.required_experiments}")
    return "\n".join(parts)


def _build_user_prompt(cluster_id: int, claims) -> str:
    # Sort claims by paper for readability
    by_paper: dict = {}
    for c in claims:
        by_paper.setdefault(c.paper_id, []).append(c)

    collections_all: set = set()
    for c in claims:
        collections_all.update(c.collections)

    lines = [
        f"CLAIM CLUSTER {cluster_id}",
        f"Total claims: {len(claims)} from {len(by_paper)} papers",
        f"Collections spanned: {', '.join(sorted(collections_all)) or 'Unknown'}",
        "",
    ]

    for pid, pclaims in sorted(by_paper.items()):
        c0 = pclaims[0]
        coll_str = ", ".join(c0.collections) if c0.collections else "Uncategorized"
        lines.append(f"--- {c0.title} ({c0.year}) [{c0.journal}] | {coll_str} ---")
        for claim in pclaims:
            lines.append(_format_claim_block(claim))
            lines.append("")

    return "\n".join(lines)


# ── Synthesizer ───────────────────────────────────────────────────────────────

class ClaimSynthesizer:
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

    def synthesize(self, cluster_id: int, claims) -> Optional[ClusterSynthesis]:
        """
        Synthesize a cluster. Returns ClusterSynthesis or None on failure.
        If the cluster has more than MAX_CLAIMS_PER_CALL claims, samples evenly.
        """
        if len(claims) > MAX_CLAIMS_PER_CALL:
            import random
            sampled = random.sample(claims, MAX_CLAIMS_PER_CALL)
            logger.info(f"Cluster {cluster_id}: sampling {MAX_CLAIMS_PER_CALL}/{len(claims)} claims")
        else:
            sampled = claims

        # Need claims from at least 2 different papers to be meaningful
        paper_ids = {c.paper_id for c in sampled}
        if len(paper_ids) < 2:
            logger.debug(f"Cluster {cluster_id}: only 1 paper, skipping synthesis")
            return None

        user_content = _build_user_prompt(cluster_id, sampled)
        data = self._call_api(user_content, cluster_id)
        if data is None:
            return None

        collections_spanned = sorted({
            col for c in claims for col in c.collections
        })

        try:
            return ClusterSynthesis(
                cluster_id=cluster_id,
                n_claims=len(claims),
                n_papers=len({c.paper_id for c in claims}),
                collections_spanned=collections_spanned,
                consensus=data.get("consensus", ""),
                boundary_conditions=data.get("boundary_conditions", ""),
                contested=data.get("contested", ""),
                evidence_quality_note=data.get("evidence_quality_note", ""),
                emergent_hypotheses=data.get("emergent_hypotheses", []),
                cross_domain_insight=data.get("cross_domain_insight", ""),
                keywords=data.get("keywords", []),
            )
        except Exception as e:
            logger.warning(f"Schema validation error for cluster {cluster_id}: {e}")
            return None

    def _call_api(self, user_content: str, cluster_id: int) -> Optional[dict]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=3000,
                    system=SYSTEM_PROMPT,
                    tools=[SYNTHESIS_TOOL],
                    tool_choice={"type": "tool", "name": "synthesize_claim_cluster"},
                    messages=[{"role": "user", "content": user_content}],
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return block.input
                logger.warning(f"Cluster {cluster_id}: no tool_use block in response")
                return None
            except anthropic.RateLimitError:
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit for cluster {cluster_id}. Waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as e:
                logger.warning(f"API error for cluster {cluster_id} (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay)
        logger.error(f"Failed to synthesize cluster {cluster_id} after {self.max_retries} attempts")
        return None
