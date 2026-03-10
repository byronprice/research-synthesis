"""
Query the Zotero SQLite database to build a master paper inventory.
Returns paper metadata + PDF file paths without touching any PDFs.

Usage:
    reader = ZoteroReader("~/Zotero/zotero.sqlite",
                          "~/Zotero/storage")
    papers = reader.get_all_papers()
"""

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


# Zotero field IDs (stable across versions)
FIELD_TITLE = 1
FIELD_ABSTRACT = 2
FIELD_DATE = 6
FIELD_PUBLICATION = 38
FIELD_DOI = 59

# Item type IDs to include
PAPER_TYPE_IDS = {
    7,   # book
    8,   # bookSection
    11,  # conferencePaper
    22,  # journalArticle
    31,  # preprint
    34,  # report
}


@dataclass
class PaperRecord:
    paper_id: int               # Zotero itemID of the parent item
    zotero_key: str             # Zotero key of the parent item (e.g. AIF9JL35)
    attachment_key: str         # Zotero key of the PDF attachment item (storage dir name)
    title: str
    abstract: str
    year: str                   # raw date string from Zotero (e.g. "2023" or "2023-05")
    journal: str
    doi: str
    pdf_path: Path              # absolute path to the PDF file
    collections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "zotero_key": self.zotero_key,
            "attachment_key": self.attachment_key,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "pdf_path": str(self.pdf_path),
            "collections": self.collections,
        }


class ZoteroReader:
    """
    Read-only interface to the Zotero SQLite database.

    Always works on a copy of the database to avoid locking conflicts with the
    Zotero application.
    """

    def __init__(self, db_path: str, storage_path: str):
        self.db_path = Path(db_path)
        self.storage_path = Path(storage_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Zotero database not found: {self.db_path}")
        if not self.storage_path.exists():
            raise FileNotFoundError(f"Zotero storage not found: {self.storage_path}")

        # Work on a copy so we never lock the live database
        self._db_copy = Path("/tmp/zotero_research_synthesis.sqlite")
        shutil.copy2(self.db_path, self._db_copy)
        logger.info(f"Copied Zotero DB to {self._db_copy}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_copy)
        conn.row_factory = sqlite3.Row
        return conn

    def _build_field_lookup(self, conn: sqlite3.Connection, item_ids: list[int]) -> dict:
        """
        Return {item_id: {field_name: value}} for the given item IDs.
        Fetches title, abstract, date, publicationTitle, DOI in one query.
        """
        if not item_ids:
            return {}

        placeholders = ",".join("?" * len(item_ids))
        field_ids = (FIELD_TITLE, FIELD_ABSTRACT, FIELD_DATE, FIELD_PUBLICATION, FIELD_DOI)
        field_id_placeholders = ",".join("?" * len(field_ids))

        rows = conn.execute(
            f"""
            SELECT id.itemID, id.fieldID, idv.value
            FROM itemData id
            JOIN itemDataValues idv ON id.valueID = idv.valueID
            WHERE id.itemID IN ({placeholders})
              AND id.fieldID IN ({field_id_placeholders})
            """,
            [*item_ids, *field_ids],
        ).fetchall()

        field_id_to_name = {
            FIELD_TITLE: "title",
            FIELD_ABSTRACT: "abstract",
            FIELD_DATE: "date",
            FIELD_PUBLICATION: "journal",
            FIELD_DOI: "doi",
        }

        result: dict[int, dict] = {iid: {} for iid in item_ids}
        for row in rows:
            name = field_id_to_name.get(row["fieldID"])
            if name:
                result[row["itemID"]][name] = row["value"]
        return result

    def _build_collection_lookup(self, conn: sqlite3.Connection, item_ids: list[int]) -> dict:
        """Return {item_id: [collection_name, ...]}"""
        if not item_ids:
            return {}

        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT ci.itemID, c.collectionName
            FROM collectionItems ci
            JOIN collections c ON ci.collectionID = c.collectionID
            WHERE ci.itemID IN ({placeholders})
            """,
            item_ids,
        ).fetchall()

        result: dict[int, list] = {iid: [] for iid in item_ids}
        for row in rows:
            result[row["itemID"]].append(row["collectionName"])
        return result

    def get_all_papers(self) -> list[PaperRecord]:
        """
        Return a PaperRecord for every paper in Zotero that has a PDF attachment
        stored in the local Zotero storage directory.
        """
        conn = self._connect()
        type_placeholders = ",".join("?" * len(PAPER_TYPE_IDS))

        # Get all parent items of the right type that have PDF attachments
        rows = conn.execute(
            f"""
            SELECT
                i.itemID        AS parent_item_id,
                i.key           AS parent_key,
                i_att.key       AS attachment_key,
                ia.path         AS attachment_path
            FROM items i
            JOIN itemAttachments ia ON i.itemID = ia.parentItemID
            JOIN items i_att ON ia.itemID = i_att.itemID
            WHERE i.itemTypeID IN ({type_placeholders})
              AND ia.contentType = 'application/pdf'
            """,
            list(PAPER_TYPE_IDS),
        ).fetchall()

        logger.info(f"Found {len(rows)} PDF attachment rows in Zotero DB")

        # Resolve PDF paths and filter to files that actually exist on disk
        valid_rows = []
        missing = 0
        for row in rows:
            att_path_str = row["attachment_path"]
            if att_path_str is None:
                missing += 1
                continue
            # Format is "storage:Filename.pdf" — strip the "storage:" prefix
            if att_path_str.startswith("storage:"):
                filename = att_path_str[len("storage:"):]
            else:
                filename = att_path_str

            pdf_path = self.storage_path / row["attachment_key"] / filename
            if pdf_path.exists():
                valid_rows.append((row, pdf_path))
            else:
                missing += 1

        logger.info(
            f"{len(valid_rows)} PDFs found on disk, {missing} attachment paths not found"
        )

        parent_ids = [r[0]["parent_item_id"] for r in valid_rows]

        # Batch-fetch metadata and collections
        fields = self._build_field_lookup(conn, parent_ids)
        collections = self._build_collection_lookup(conn, parent_ids)
        conn.close()

        # Deduplicate: keep one PDF per parent_item_id (some papers have multiple attachments)
        seen_parent_ids: set[int] = set()
        papers: list[PaperRecord] = []

        for row, pdf_path in valid_rows:
            pid = row["parent_item_id"]
            if pid in seen_parent_ids:
                continue
            seen_parent_ids.add(pid)

            f = fields.get(pid, {})
            year_raw = f.get("date", "")
            # Extract 4-digit year from date strings like "2023", "2023-05-01", "May 2023"
            year = _extract_year(year_raw)

            papers.append(
                PaperRecord(
                    paper_id=pid,
                    zotero_key=row["parent_key"],
                    attachment_key=row["attachment_key"],
                    title=f.get("title", ""),
                    abstract=f.get("abstract", ""),
                    year=year,
                    journal=f.get("journal", ""),
                    doi=f.get("doi", ""),
                    pdf_path=pdf_path,
                    collections=collections.get(pid, []),
                )
            )

        logger.info(f"Built inventory of {len(papers)} unique papers with PDFs")
        return papers


def _extract_year(date_str: str) -> str:
    """Extract 4-digit year from a Zotero date string."""
    import re
    match = re.search(r"\b(19|20)\d{2}\b", date_str)
    return match.group(0) if match else ""
