"""
chunker.py
==========

Creates semantic chunks from structured PDF and audio documents.

Features
--------
✓ Heading-aware chunking
✓ Section-aware chunking
✓ Table preservation
✓ Audio preservation
✓ Metadata preservation
✓ Stable chunk IDs
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy


class Chunker:
    def __init__(self, max_chars=1200, overlap=200):

        self.max_chars = max_chars
        self.overlap = overlap

    # -----------------------------------------------------
    # Chunk ID
    # -----------------------------------------------------

    def chunk_id(self, chunk):

        # Exclude fields that vary between runs even when the underlying
        # content is identical — processed_at is a fresh timestamp every
        # time data_processing.py runs, and source_file_hash changes on
        # any re-save of the source file (even a no-op one, e.g. touching
        # file metadata). Hashing either of these means reprocessing the
        # exact same page/table/audio segment produces a DIFFERENT
        # chunk_id, silently breaking every downstream reference to the
        # old id (Neo4j nodes, embeddings, failed_chunks.json, ground
        # truth provenance) even though nothing about the content changed.
        stable_fields = {
            k: v
            for k, v in chunk.items()
            if k not in ("processed_at", "source_file_hash")
        }

        payload = json.dumps(stable_fields, sort_keys=True)

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -----------------------------------------------------
    # Split paragraph
    # -----------------------------------------------------

    def split_long_text(self, text):

        if len(text) <= self.max_chars:
            return [text]

        paragraphs = text.split("\n")

        chunks = []

        current = ""

        for para in paragraphs:
            para = para.strip()

            if not para:
                continue

            if len(current) + len(para) < self.max_chars:
                current += "\n" + para

            else:
                chunks.append(current.strip())

                overlap = current[-self.overlap :]

                current = overlap + "\n" + para

        if current:
            chunks.append(current.strip())

        return chunks

    # -----------------------------------------------------
    # Process one page
    # -----------------------------------------------------

    def process_page(self, page_json, metadata):

        output = []

        current_heading = None

        current_section = None

        page_type = page_json.get("page_type", "financial_statement")

        buffer = ""

        for item in page_json["content"]:
            item_type = item["type"]

            # -----------------------------
            # Heading
            # -----------------------------

            if item_type == "heading":
                if buffer.strip():
                    output.extend(
                        self.create_chunks(
                            buffer,
                            metadata,
                            current_heading,
                            current_section,
                            page_json["page"],
                            page_type,
                        )
                    )

                    buffer = ""

                current_heading = item["text"]

                current_section = item.get("section", current_section)

            # -----------------------------
            # Normal text
            # -----------------------------

            elif item_type == "text":
                buffer += "\n" + item["text"]

            # -----------------------------
            # Table
            # -----------------------------

            elif item_type == "table":
                if buffer.strip():
                    output.extend(
                        self.create_chunks(
                            buffer,
                            metadata,
                            current_heading,
                            current_section,
                            page_json["page"],
                            page_type,
                        )
                    )

                    buffer = ""

                table_chunk = self.table_chunk(
                    item,
                    metadata,
                    current_heading,
                    current_section,
                    page_json["page"],
                    page_type,
                )

                if table_chunk["embedding_text"].strip():
                    output.append(table_chunk)

        if buffer.strip():
            output.extend(
                self.create_chunks(
                    buffer,
                    metadata,
                    current_heading,
                    current_section,
                    page_json["page"],
                    page_type,
                )
            )

        return output

        # -----------------------------------------------------

    # Create text chunks
    # -----------------------------------------------------

    def create_chunks(
        self, text, metadata, heading, section, page, page_type="financial_statement"
    ):

        chunks = []

        pieces = self.split_long_text(text)

        for part in pieces:
            if len(part.strip()) < 30:
                # Catches whitespace-only overlap fragments AND short
                # noise chunks like "August 12, 2025" or "Continued..."
                # that carry no retrievable financial content.
                continue

            chunk = {
                **deepcopy(metadata),
                "chunk_type": "text",
                "page": page,
                "page_type": page_type,
                "heading": heading,
                "section": section,
                "text": part,
                "embedding_text": part,
            }

            chunk["chunk_id"] = self.chunk_id(chunk)

            chunks.append(chunk)

        return chunks

        # -----------------------------------------------------

    # Create Table Chunk
    # -----------------------------------------------------

    def table_chunk(
        self, table, metadata, heading, section, page, page_type="financial_statement"
    ):

        chunk = {
            **deepcopy(metadata),
            "chunk_type": "table",
            "page": page,
            "page_type": page_type,
            "heading": heading,
            "section": section,
            "title": table.get("title"),
            "headers": table.get("headers", []),
            "rows": table.get("rows", []),
            "embedding_text": table.get("embedding_text", ""),
        }

        chunk["chunk_id"] = self.chunk_id(chunk)

        return chunk

    # -----------------------------------------------------
    # Audio Chunks
    # -----------------------------------------------------

    def process_audio(self, audio_chunks):

        output = []

        for chunk in audio_chunks:
            new_chunk = deepcopy(chunk)

            new_chunk["chunk_type"] = "audio"

            new_chunk["chunk_id"] = self.chunk_id(new_chunk)

            output.append(new_chunk)

        return output

    # -----------------------------------------------------
    # PDF Document
    # -----------------------------------------------------

    def process_pdf(self, document, metadata):

        chunks = []

        for page in document:
            page_chunks = self.process_page(page, metadata)

            chunks.extend(page_chunks)

        return chunks

    # -----------------------------------------------------
    # Generic Document
    # -----------------------------------------------------

    def process_document(self, document, document_type, metadata):

        if document_type == "pdf":
            return self.process_pdf(document, metadata)

        elif document_type == "audio":
            return self.process_audio(document)

        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    def statistics(self, chunks):

        stats = {
            "total_chunks": len(chunks),
            "text_chunks": 0,
            "table_chunks": 0,
            "audio_chunks": 0,
        }

        for chunk in chunks:
            t = chunk["chunk_type"]

            if t == "text":
                stats["text_chunks"] += 1

            elif t == "table":
                stats["table_chunks"] += 1

            elif t == "audio":
                stats["audio_chunks"] += 1

        return stats

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    def export_json(self, chunks, output_path):

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

            # -----------------------------------------------------


# Singleton
# -----------------------------------------------------

_chunker = Chunker()


def build_chunks(document, document_type, metadata):

    return _chunker.process_document(document, document_type, metadata)


# -----------------------------------------------------
# CLI
# -----------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input")

    parser.add_argument("--type")

    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        document = json.load(f)

    chunks = build_chunks(document, args.type, {})

    print(json.dumps(chunks, indent=2, ensure_ascii=False))
