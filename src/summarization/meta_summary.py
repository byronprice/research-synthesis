"""
Generate meta-summaries for each cluster of papers.

For each cluster: synthesizes consensus findings, temporal trends,
contradictions (coarse), open questions, and representative papers.

Usage:
    meta = MetaSummarizer(api_key="...", model="claude-haiku-4-5-20251001")
    cluster_summaries = meta.summarize_clusters(
        summaries_dir="data/summaries",
        assignments={paper_id: cluster_id, ...},
        output_dir="data/clusters/meta_summaries",
    )
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger


SYSTEM_PROMPT = """\
You are a neuroscience literature analyst. Synthesize a cluster of related papers
to identify consensus, trends, disagreements, and open questions.
Be precise, use field terminology, and cite papers by (Author, Year) when relevant.
Return only the JSON object requested — no preamble or markdown fences."""

META_PROMPT_TEMPLATE = """\
Synthesize this cluster of {n} neuroscience papers on the topic of {topic}.

PAPER SUMMARIES:
{summaries_text}

Return a JSON object:
{{
  "topic_label": "concise label for this cluster (5-10 words)",
  "consensus": ["agreed-upon finding 1", "agreed-upon finding 2", ...],
  "trends": ["how understanding has evolved chronologically"],
  "contradictions": [
    {{
      "description": "brief description of the contradiction",
      "papers_for": ["Author Year", ...],
      "papers_against": ["Author Year", ...]
    }}
  ],
  "open_questions": ["unresolved question 1", ...],
  "methodology_notes": "common methods; any method-specific disagreements",
  "representative_papers": ["most central/impactful titles in this cluster"],
  "keywords": ["keyword1", ...]
}}"""


class MetaSummarizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_papers_per_call: int = 20,
        retry_delay: float = 5.0,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_papers_per_call = max_papers_per_call
        self.retry_delay = retry_delay

    def summarize_clusters(
        self,
        summaries_dir: str,
        assignments: dict[int, int],
        output_dir: str,
        delay: float = 0.2,
    ) -> dict[int, dict]:
        """
        For each cluster, generate a meta-summary.
        Writes cluster_{id}.json files to output_dir.
        Returns {cluster_id: meta_summary_dict}.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Load summaries
        summaries_by_id: dict[int, dict] = {}
        for fpath in Path(summaries_dir).glob("*.json"):
            try:
                with open(fpath) as f:
                    s = json.load(f)
                    summaries_by_id[s["paper_id"]] = s
            except Exception as e:
                logger.warning(f"Could not load {fpath}: {e}")

        # Group by cluster
        clusters: dict[int, list[dict]] = defaultdict(list)
        for pid, cluster_id in assignments.items():
            if cluster_id == -1:
                continue  # skip noise
            if pid in summaries_by_id:
                clusters[cluster_id].append(summaries_by_id[pid])

        logger.info(f"Generating meta-summaries for {len(clusters)} clusters")
        results: dict[int, dict] = {}

        for cluster_id, papers in sorted(clusters.items()):
            logger.info(f"Cluster {cluster_id}: {len(papers)} papers")

            meta = self._summarize_cluster(cluster_id, papers)
            if meta:
                results[cluster_id] = meta
                out_file = out_path / f"cluster_{cluster_id}.json"
                with open(out_file, "w") as f:
                    json.dump(meta, f, indent=2)

            time.sleep(delay)

        logger.info(f"Meta-summaries written to {output_dir}")
        return results

    def _summarize_cluster(self, cluster_id: int, papers: list[dict]) -> Optional[dict]:
        """
        Summarize one cluster. Uses two-level hierarchy if > max_papers_per_call.
        """
        if len(papers) <= self.max_papers_per_call:
            return self._call_api(cluster_id, papers)
        else:
            # Two-level: summarize batches, then summarize those summaries
            logger.info(f"  Cluster {cluster_id} has {len(papers)} papers — using two-level summarization")
            batch_metas = []
            for i in range(0, len(papers), self.max_papers_per_call):
                batch = papers[i : i + self.max_papers_per_call]
                meta = self._call_api(cluster_id, batch)
                if meta:
                    batch_metas.append(meta)
                time.sleep(self.retry_delay / 2)

            if not batch_metas:
                return None

            # Synthesize batch metas (treat them as "papers" in a second pass)
            # Convert each batch meta into a pseudo-summary for the second call
            pseudo_papers = []
            for m in batch_metas:
                pseudo_papers.append({
                    "title": m.get("topic_label", "Batch summary"),
                    "year": "",
                    "key_findings": m.get("consensus", []) + m.get("trends", []),
                    "keywords": m.get("keywords", []),
                    "claims": [],
                    "limitations": [],
                })
            return self._call_api(cluster_id, pseudo_papers, topic_hint="(synthesized from batch summaries)")

    def _call_api(self, cluster_id: int, papers: list[dict], topic_hint: str = "") -> Optional[dict]:
        # Infer topic from keywords
        all_keywords = []
        for p in papers:
            all_keywords.extend(p.get("keywords", []))
        from collections import Counter
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]
        topic = ", ".join(top_keywords) + (f" {topic_hint}" if topic_hint else "")

        summaries_text = self._format_papers(papers)
        user_content = META_PROMPT_TEMPLATE.format(
            n=len(papers),
            topic=topic or "neuroscience",
            summaries_text=summaries_text,
        )

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                result = json.loads(raw)
                result["cluster_id"] = cluster_id
                result["paper_count"] = len(papers)
                result["paper_ids"] = [p["paper_id"] for p in papers]
                return result
            except anthropic.RateLimitError:
                time.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Error on cluster {cluster_id} attempt {attempt+1}: {e}")
                time.sleep(self.retry_delay)

        logger.error(f"Failed to generate meta-summary for cluster {cluster_id}")
        return None

    def _format_papers(self, papers: list[dict]) -> str:
        lines = []
        for p in papers:
            lines.append(f"--- {p.get('title','?')} ({p.get('year','?')}) ---")
            findings = "; ".join(p.get("key_findings", [])[:4])
            lines.append(f"Findings: {findings}")
            keywords = ", ".join(p.get("keywords", [])[:6])
            lines.append(f"Keywords: {keywords}")
            lines.append("")
        return "\n".join(lines)
