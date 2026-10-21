"""Hermetic tests for ``scripts/fetch_assets.py``.

No network: the download path is exercised through ``file://`` URLs, which
``urllib.request.urlopen`` serves from disk, so a missing asset is a missing
file rather than a flaky request.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fetch_assets

POINTER_TEMPLATE = """outs:
- md5: {md5}
  size: {size}
  hash: md5
  path: {path}
"""


def _write_pointer(root: Path, pointer: str, payload: bytes) -> Path:
    """Create a DVC pointer and the matching object, and return the pointer."""
    pointer_path = root / pointer
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        POINTER_TEMPLATE.format(
            md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            size=len(payload),
            path=Path(pointer).name.replace(".dvc", ""),
        ),
        encoding="utf-8",
    )
    return pointer_path


def _repo(root: Path) -> tuple[Path, dict[str, bytes]]:
    """Build a miniature repo with one pointer per shipped asset."""
    payloads = {
        "models/onnx/model_int8.onnx.dvc": b"onnx-bytes",
        "models/baseline/final/tokenizer.json.dvc": b"tokenizer-bytes",
        "data/raw/Final_Data.csv.dvc": b"csv-bytes",
    }
    for pointer, payload in payloads.items():
        _write_pointer(root, pointer, payload)
    return root, payloads


def _asset(root: Path, pointer: str) -> fetch_assets.Asset:
    return fetch_assets.build_assets(root, (pointer,))[0]


def test_parse_pointer_reads_md5_size_and_path(tmp_path: Path) -> None:
    pointer = _write_pointer(tmp_path, "data/raw/Final_Data.csv.dvc", b"csv")
    md5, size, path = fetch_assets.parse_pointer(pointer)
    assert md5 == hashlib.md5(b"csv", usedforsecurity=False).hexdigest()
    assert size == 3
    assert path == "Final_Data.csv"


def test_parse_pointer_rejects_a_file_that_is_not_a_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "broken.dvc"
    pointer.write_text("outs:\n- path: thing\n", encoding="utf-8")
    with pytest.raises(fetch_assets.AssetError):
        fetch_assets.parse_pointer(pointer)


def test_build_assets_places_the_object_next_to_its_pointer(tmp_path: Path) -> None:
    _repo(tmp_path)
    assets = fetch_assets.build_assets(tmp_path, ("data/raw/Final_Data.csv.dvc",))
    assert assets[0].target == tmp_path / "data/raw/Final_Data.csv"
    assert assets[0].name == "Final_Data.csv"


def test_build_assets_reports_a_missing_pointer(tmp_path: Path) -> None:
    with pytest.raises(fetch_assets.AssetError):
        fetch_assets.build_assets(tmp_path, ("models/onnx/nope.dvc",))


def test_fetch_asset_downloads_and_verifies(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    source = tmp_path / "release"
    source.mkdir()
    (source / "Final_Data.csv").write_bytes(payloads["data/raw/Final_Data.csv.dvc"])
    asset = _asset(root, "data/raw/Final_Data.csv.dvc")

    state = fetch_assets.fetch_asset(asset, source.as_uri())

    assert state == "downloaded"
    assert asset.target.read_bytes() == b"csv-bytes"


def test_fetch_asset_skips_a_verified_copy(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    asset = _asset(root, "data/raw/Final_Data.csv.dvc")
    asset.target.parent.mkdir(parents=True, exist_ok=True)
    asset.target.write_bytes(payloads["data/raw/Final_Data.csv.dvc"])

    # A base URL that cannot be read: a cached asset must not touch the network.
    state = fetch_assets.fetch_asset(asset, (tmp_path / "absent").as_uri())

    assert state == "cached"


def test_fetch_asset_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    root, _ = _repo(tmp_path)
    source = tmp_path / "release"
    source.mkdir()
    (source / "Final_Data.csv").write_bytes(b"not-the-tracked-bytes")
    asset = _asset(root, "data/raw/Final_Data.csv.dvc")

    with pytest.raises(fetch_assets.AssetError):
        fetch_assets.fetch_asset(asset, source.as_uri())

    assert not asset.target.exists()
    assert not asset.target.with_name(f"{asset.target.name}.part").exists()


def test_fetch_asset_reports_a_missing_release_asset(tmp_path: Path) -> None:
    root, _ = _repo(tmp_path)
    asset = _asset(root, "data/raw/Final_Data.csv.dvc")

    with pytest.raises(fetch_assets.AssetError):
        fetch_assets.fetch_asset(asset, (tmp_path / "absent").as_uri())


def _verify_only(root: Path, *pointers: str) -> int:
    return fetch_assets.main(
        [
            "--repo-root",
            str(root),
            "--verify-only",
            "--pointers",
            ",".join(pointers),
        ]
    )


def _fetch(root: Path, base_url: str, *pointers: str) -> int:
    return fetch_assets.main(
        [
            "--repo-root",
            str(root),
            "--base-url",
            base_url,
            "--pointers",
            ",".join(pointers),
        ]
    )


def test_verify_only_mode_reports_a_clean_tree(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    for pointer, payload in payloads.items():
        target = root / pointer.replace(".dvc", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    assert _verify_only(root, *payloads) == 0


def test_verify_only_mode_fails_on_a_missing_file(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    assert _verify_only(root, *payloads) == 1


def test_main_fetches_every_pointer(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    source = tmp_path / "release"
    source.mkdir()
    for pointer, payload in payloads.items():
        (source / pointer.replace(".dvc", "").split("/")[-1]).write_bytes(payload)

    code = _fetch(root, source.as_uri(), *payloads)

    assert code == 0
    assert (root / "models/onnx/model_int8.onnx").read_bytes() == b"onnx-bytes"
    assert (root / "models/baseline/final/tokenizer.json").read_bytes() == (
        b"tokenizer-bytes"
    )
    assert (root / "data/raw/Final_Data.csv").read_bytes() == b"csv-bytes"


def test_main_fails_when_one_asset_is_unavailable(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    code = _fetch(root, (tmp_path / "absent").as_uri(), *payloads)
    assert code == 1


def test_main_fails_when_a_pointer_is_missing(tmp_path: Path) -> None:
    root, payloads = _repo(tmp_path)
    code = _verify_only(root, *payloads, "models/onnx/absent.dvc")
    assert code == 1
