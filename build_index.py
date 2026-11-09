"""בניית המאגר (מריצים פעם אחת, ושוב כשהקבצים משתנים).

טוען קבצי .md מ-data/, חותך לפי כותרות ##, שומר שדות כ-metadata,
יוצר Embeddings ושומר ב-ChromaDB. ההרצה מוחקת ובונה מחדש את ה-collection,
ולכן אין כפילויות בהרצה חוזרת.
"""
import hashlib
import re
import sys
from pathlib import Path

import config as cfg

# כותרת ברמה 2 בדיוק: "## " (ולא "###")
HEADER_RE = re.compile(r"^##\s+(\S.*)$")


def extract_field(body: str, label: str) -> str:
    """שולף ערך שדה כמו 'מזהה: X' או '- **מזהה:** X' (סובל מ-Markdown מודגש)."""
    pattern = rf"^[ \t>*\-_]*{re.escape(label)}[ \t]*\**[ \t]*[:：][ \t]*\**[ \t]*(.+?)[ \t]*$"
    m = re.search(pattern, body, re.MULTILINE)
    return m.group(1).strip("*_ \t") if m else ""


def parse_items(path: Path) -> list[dict]:
    """מפצל קובץ לפריטים לפי כותרות ##; מתעלם מכותרות בתוך בלוקי קוד."""
    sections, title, lines, in_code = [], None, [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADER_RE.match(line)
        if m:
            if title is not None:
                sections.append((title, lines))
            title, lines = m.group(1).strip(), []
        elif title is not None:  # טקסט לפני ה-## הראשון (כותרת הקובץ) מדולג
            lines.append(line)
    if title is not None:
        sections.append((title, lines))

    items = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        fields = {k: extract_field(body, label) for k, label in cfg.FIELD_LABELS.items()}
        for k, v in fields.items():
            if not v:
                print(f"  אזהרה: בפריט '{title}' ({path.name}) חסר השדה '{cfg.FIELD_LABELS[k]}'")
        items.append({"title": title, "body": body, "source_file": path.name, **fields})
    return items


def build_nodes(items: list[dict]):
    """פריט = Chunk אחד. פריט ארוך מאוד מפוצל עם חפיפה (נדיר)."""
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode

    splitter = SentenceSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    nodes = []
    for idx, it in enumerate(items):
        parts = splitter.split_text(f"## {it['title']}\n{it['body']}") or [it["title"]]
        for i, part in enumerate(parts):
            if i > 0:  # בחלקי המשך מחזירים את הכותרת, כדי שלא יאבד ההקשר
                part = f"## {it['title']} (המשך)\n{part}"
            meta = {
                "item_id": it["item_id"],
                "item_type": it["item_type"],
                "event_name": it["event_name"],
                "item_title": it["title"],
                "source_file": it["source_file"],
                "part": i,
            }
            node_id = hashlib.sha1(f"{it['source_file']}|{idx}|{i}".encode()).hexdigest()
            nodes.append(
                TextNode(
                    id_=node_id,
                    text=part,
                    metadata=meta,
                    # ה-metadata נשמר לתצוגת מקורות; הטקסט עצמו הוא מה שנכנס ל-Embedding
                    excluded_embed_metadata_keys=list(meta),
                    excluded_llm_metadata_keys=list(meta),
                )
            )
    return nodes


def main() -> None:
    files = sorted(cfg.DATA_DIR.glob("*.md"))
    if not files:
        sys.exit(f"לא נמצאו קבצי .md בתיקייה {cfg.DATA_DIR}")

    items = []
    for f in files:
        file_items = parse_items(f)
        print(f"{f.name}: {len(file_items)} פריטים")
        items.extend(file_items)
    nodes = build_nodes(items)
    print(f"סה\"כ {len(items)} פריטים, {len(nodes)} Chunks")

    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from models import get_embed_model

    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    try:  # מחיקה ובנייה מחדש = הרצה חוזרת בלי כפילויות
        client.delete_collection(cfg.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(cfg.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model(), show_progress=True)
    print(f"המאגר נבנה: {collection.count()} Chunks נשמרו ב-{cfg.CHROMA_DIR}")


if __name__ == "__main__":
    main()
