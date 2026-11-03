from pathlib import Path

import pytest

from trustguard.etl.chunker import chunk_pages, split_sections, title_from_filename
from trustguard.etl.loader import (
    PageText,
    _unwrap,
    discover_documents,
    load_document,
    load_txt,
)
from trustguard.etl.pipeline import ingest_directory, ingest_files


class TestLoader:
    def test_txt_splits_pages_on_form_feed(self, sample_txt: Path) -> None:
        pages = load_txt(sample_txt)
        assert [p.page for p in pages] == [1, 2]
        assert pages[0].source == "wire_policy.txt"
        assert "dual authorization" in pages[0].text
        assert pages[1].text.startswith("2. Fees")

    def test_pdf_pages_keep_page_numbers(self, policy_dir: Path) -> None:
        pages = load_document(policy_dir / "transaction_limits_policy.pdf")
        assert [p.page for p in pages] == [1, 2, 3]
        assert "$25,000 per business day" in pages[1].text

    def test_pdf_soft_wraps_are_joined(self, policy_dir: Path) -> None:
        pages = load_document(policy_dir / "transaction_limits_policy.pdf")
        assert "through online banking" in pages[1].text

    def test_unwrap_keeps_clause_breaks(self) -> None:
        assert _unwrap("limit is\nhigher.\n2.1 Next") == "limit is higher.\n2.1 Next"

    def test_unsupported_type(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.docx"
        path.write_text("x")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "missing.pdf")

    def test_discover_is_recursive_and_filtered(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub" / "b.pdf").write_bytes(b"%PDF")
        (tmp_path / "c.md").write_text("c")
        assert [p.name for p in discover_documents(tmp_path)] == ["a.txt", "b.pdf"]


class TestChunker:
    def test_sections_are_split_at_headings(self) -> None:
        sections = split_sections(
            "Intro line\n1. Scope\nText\n2. Limits\nMore", inherited="0. Prior"
        )
        assert [s for s, _ in sections] == ["0. Prior", "1. Scope", "2. Limits"]
        assert sections[1][1] == "1. Scope\nText"

    def test_no_headings_inherits_section(self) -> None:
        assert split_sections("plain text", inherited="3. Fees") == [("3. Fees", "plain text")]
        assert split_sections("   ") == []

    def test_chunks_carry_metadata_and_never_span_pages(self, sample_txt: Path) -> None:
        chunks = chunk_pages(load_txt(sample_txt), chunk_size=200, chunk_overlap=20)
        assert {c.page for c in chunks} == {1, 2}
        by_section = {c.section for c in chunks}
        assert {"1. Limits", "2. Fees"} <= by_section
        limit_chunk = next(c for c in chunks if "dual authorization" in c.text)
        assert limit_chunk.section == "1. Limits"
        assert limit_chunk.to_metadata()["source"] == "wire_policy.txt"
        assert limit_chunk.embedding_text.startswith("Wire Policy > 1. Limits\n")

    def test_section_carries_over_to_next_page(self) -> None:
        pages = [
            PageText("1. Wires\nWires are limited.", "doc.txt", 1),
            PageText("Continued clause about wires.", "doc.txt", 2),
        ]
        chunks = chunk_pages(pages)
        assert chunks[-1].page == 2
        assert chunks[-1].section == "1. Wires"

    def test_long_sections_are_split_with_overlap(self) -> None:
        sentence = "Clause text about limits and fees. "
        page = PageText("1. Long\n" + sentence * 60, "long.txt", 1)
        chunks = chunk_pages([page], chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 3
        assert all(len(c.text) <= 300 for c in chunks)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_ids_are_deterministic(self, sample_txt: Path) -> None:
        first = [c.chunk_id for c in chunk_pages(load_txt(sample_txt))]
        second = [c.chunk_id for c in chunk_pages(load_txt(sample_txt))]
        assert first == second

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError):
            chunk_pages([], chunk_size=100, chunk_overlap=100)

    def test_title_from_filename(self) -> None:
        assert title_from_filename("aml_kyc_policy.pdf") == "AML KYC Policy"
        assert title_from_filename("fincen_ctr-guide.pdf") == "FinCEN CTR Guide"


class TestPipeline:
    def test_ingest_directory_reports_counts(self, store, policy_dir: Path) -> None:
        report = ingest_directory(policy_dir, store)
        assert report.documents >= 6
        assert report.chunks == store.count() > 0
        sources = {row["source"] for row in report.per_document}
        assert "fee_schedule.pdf" in sources
        assert all(row["chunks"] > 0 for row in report.per_document)

    def test_reingest_replaces_instead_of_duplicating(self, store, sample_txt: Path) -> None:
        ingest_files([sample_txt], store)
        first = store.count()
        ingest_files([sample_txt], store)
        assert store.count() == first

    def test_reingest_drops_stale_chunks(self, store, sample_txt: Path) -> None:
        ingest_files([sample_txt], store)
        sample_txt.write_text("1. Only\nOne short clause.", encoding="utf-8")
        ingest_files([sample_txt], store)
        assert store.count() == 1

    def test_empty_ingest(self, store) -> None:
        report = ingest_files([], store)
        assert report.chunks == 0
        assert report.per_document == []
