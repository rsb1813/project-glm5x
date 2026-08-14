# K3X 변환 중단 뒤 검증된 extent만 재사용하는지 검증합니다.
import json
from pathlib import Path

import google_crc32c
import pytest

from k3x_converter.format import K3XError
from k3x_converter.resume import read_resume_manifest
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def test_conversion_resumes_without_rewriting_completed_extents(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "synthetic.k3x"
    first = convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=3)
    assert first.completed is False
    before = read_resume_manifest(output.with_suffix(".k3x.resume.json"))

    second = convert(synthetic_source, output, chunk_bytes=257)
    assert second.completed is True
    assert second.reused_extent_ids == tuple(item.extent_id for item in before.completed)
    assert second.maximum_source_read_bytes <= 257
    assert output.exists()
    assert not output.with_suffix(".k3x.partial").exists()


def test_resume_rejects_changed_source(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "synthetic.k3x"
    convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=1)
    shard = next(synthetic_source.glob("*.safetensors"))
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(K3XError, match="SOURCE_FINGERPRINT_MISMATCH"):
        convert(synthetic_source, output, chunk_bytes=257)


def test_resume_recovers_crash_after_final_rename(
    synthetic_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "synthetic.k3x"
    resume = output.with_suffix(".k3x.resume.json")
    original_unlink = Path.unlink

    def interrupt_ledger_cleanup(path: Path, *args, **kwargs) -> None:
        if path == resume:
            raise RuntimeError("simulated crash after final rename")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_ledger_cleanup)
    with pytest.raises(RuntimeError, match="simulated crash"):
        convert(synthetic_source, output, chunk_bytes=257)
    assert output.exists()
    assert resume.exists()
    assert not output.with_suffix(".k3x.partial").exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = convert(synthetic_source, output, chunk_bytes=257)
    assert recovered.completed is True
    assert not resume.exists()


def test_full_dimension_storage_fixture_resumes_exact_extents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"

    first = convert(
        source,
        output,
        chunk_bytes=193 * 1024,
        stop_after_extents=2,
    )
    before = read_resume_manifest(output.with_suffix(".k3x.resume.json"))
    second = convert(source, output, chunk_bytes=193 * 1024)

    assert first.completed is False
    assert len(before.completed) == 2
    assert second.completed is True
    assert second.reused_extent_ids == tuple(
        item.extent_id for item in before.completed
    )
    assert second.maximum_source_read_bytes <= 193 * 1024


def test_full_dimension_storage_fixture_resume_rejects_changed_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    report = write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=1)

    with report.shard_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(K3XError, match="SOURCE_SHARD_SHA256_MISMATCH"):
        convert(source, output, chunk_bytes=193 * 1024)


@pytest.mark.parametrize("corruption", ("duplicate", "unknown", "zero", "offset"))
def test_storage_fixture_resume_rejects_noncanonical_extent_ledger(
    tmp_path: Path, corruption: str
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=2)
    resume = output.with_suffix(".k3x.resume.json")
    ledger = json.loads(resume.read_text(encoding="utf-8"))

    if corruption == "duplicate":
        ledger["completed"][1]["extent_id"] = ledger["completed"][0]["extent_id"]
    elif corruption == "unknown":
        ledger["completed"][0]["extent_id"] = "0000000000000000:data"
    elif corruption == "zero":
        ledger["completed"][0]["length"] = 0
    else:
        ledger["completed"][0]["offset"] += 4096
    resume.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(K3XError, match="INVALID_RESUME_EXTENT"):
        convert(source, output, chunk_bytes=193 * 1024)


def test_storage_fixture_resume_rejects_partial_bytes_not_matching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=1)
    resume = output.with_suffix(".k3x.resume.json")
    partial = output.with_suffix(".k3x.partial")
    ledger = json.loads(resume.read_text(encoding="utf-8"))
    item = ledger["completed"][0]

    with partial.open("r+b") as stream:
        stream.seek(item["offset"])
        original = stream.read(1)
        stream.seek(item["offset"])
        stream.write(bytes([original[0] ^ 1]))
        stream.seek(item["offset"])
        checksum = google_crc32c.Checksum()
        remaining = item["length"]
        while remaining:
            chunk = stream.read(min(193 * 1024, remaining))
            checksum.update(chunk)
            remaining -= len(chunk)
    item["crc32c"] = int.from_bytes(checksum.digest(), "big")
    resume.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(K3XError, match="RESUME_SOURCE_EXTENT_MISMATCH"):
        convert(source, output, chunk_bytes=193 * 1024)
