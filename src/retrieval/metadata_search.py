"""
Metadata-first paper search for summary mode.

Searches papers_metadata.jsonl by author name, year, and title keywords
extracted from a freeform query string. Much more reliable than vector
search for specific paper lookup (e.g. "schoonover 2021").
"""

import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path


def find_paper(query: str, metadata_path: str = "data/extracted/papers_metadata.jsonl") -> list[dict]:
    """
    Search metadata for papers matching a freeform query.

    Returns a ranked list of matching paper dicts (best match first).
    Scoring:
      +4  year matches exactly
      +3  author token found in pdf_path filename (case-insensitive)
      +1  per query token found in title (case-insensitive)
    """
    if not Path(metadata_path).exists():
        return []

    # Extract 4-digit year from query
    year_match = re.search(r"\b(19|20)\d{2}\b", query)
    year = year_match.group(0) if year_match else None

    # Remaining tokens are potential author/title terms (ignore short stop words)
    tokens = re.sub(r"\b(19|20)\d{2}\b", "", query).lower().split()
    tokens = [t for t in tokens if len(t) > 2]

    papers = []
    with open(metadata_path) as f:
        for line in f:
            papers.append(json.loads(line))

    scored = []
    for paper in papers:
        score = 0

        if year and paper.get("year") == year:
            score += 4

        # Author names live in the pdf filename: "LastName et al._2021_Title.pdf"
        filename = Path(paper.get("pdf_path", "")).name.lower()
        title_lower = paper.get("title", "").lower()

        for token in tokens:
            # Author match: token appears before the first underscore (author segment)
            author_segment = filename.split("_")[0] if "_" in filename else filename
            if token in author_segment:
                score += 3
            elif token in filename:
                score += 1
            if token in title_lower:
                score += 1

        if score > 0:
            scored.append((score, paper))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:5]]


def get_authors(paper_id: int, zotero_db: str = "~/Zotero/zotero.sqlite") -> list[str]:
    """
    Return ordered list of author last names for a paper by querying Zotero SQLite.
    Falls back to empty list if the DB is unavailable.
    """
    db_path = Path(zotero_db).expanduser()
    if not db_path.exists():
        return []
    try:
        # Copy to avoid locking the live Zotero DB
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.close()
        shutil.copy2(db_path, tmp.name)
        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.lastName, c.firstName
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            WHERE ic.itemID = ?
            ORDER BY ic.orderIndex
            """,
            (paper_id,),
        ).fetchall()
        conn.close()
        Path(tmp.name).unlink(missing_ok=True)
        return [dict(r) for r in rows]  # [{"lastName": ..., "firstName": ...}, ...]
    except Exception:
        return []


def format_authors(authors: list[dict]) -> str:
    """
    Format author list for display.
    1 author  → "Smith"
    2 authors → "Smith & Jones"
    3+        → "Smith, Jones & Lee"  (or "Smith et al." if > 4)
    """
    if not authors:
        return ""
    last_names = [a.get("lastName", "") for a in authors if a.get("lastName")]
    if not last_names:
        return ""
    if len(last_names) == 1:
        return last_names[0]
    if len(last_names) == 2:
        return f"{last_names[0]} & {last_names[1]}"
    if len(last_names) <= 4:
        return ", ".join(last_names[:-1]) + f" & {last_names[-1]}"
    return f"{last_names[0]} et al."
