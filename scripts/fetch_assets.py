"""Fetch the DVC-tracked serving assets from a public GitHub release.

Why this script exists
----------------------
The DVC remote is ``https://dagshub.com/shhth0034/Raay.dvc`` -- a repo in our
own Dagshub namespace. A reviewer who clones this repository cannot pull it:
their own Dagshub key has no access to our repo and an anonymous pull is
rejected. The objects a reviewer actually needs (the INT8 ONNX graph, the
baseline tokenizer, the raw review CSV) are therefore also published as
release assets on the public GitHub repository, so the project can be run end
to end with **no provider key at all**.

    uv run python scripts/fetch_assets.py

Every file is verified against the md5 recorded in its ``*.dvc`` pointer before
it is moved into place, so a fetch either reproduces the exact tracked bytes or
fails loudly -- the bytes on disk are the ones that produced
``reports/eval_int8.json`` and the graph the service serves.

The pointers are the only thing this script hardcodes: the target path and the
expected md5 are read from the pointer itself, so re-pointing DVC never leaves
a stale copy of the download map behind.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_BASE_URL = (
    "https://github.com/ahmeddiab1234/Raay/releases/download/raay-assets-v1"
)
BASE_URL_ENV = "RAAY_ASSETS_URL"
USER_AGENT = "raay-fetch-assets/1.0"
CHUNK = 1 << 20

# The pointers whose objects are needed to serve the API and to reproduce the
# data pipeline. ``data/raw/arabic_sentiment_reviews.csv.dvc`` is deliberately
# absent: it is 212 MB and only the EDA notebooks read it, so the quickstart
# never needs it.
POINTERS = (
    "models/onnx/model_int8.onnx.dvc",
    "models/baseline/final/config.json.dvc",
    "models/baseline/final/tokenizer.json.dvc",
    "models/baseline/final/tokenizer_config.json.dvc",
    "data/raw/Final_Data.csv.dvc",
    "data/feedback/reviewed/overrides.csv.dvc",
)

_MD5_RE = re.compile(r"^\s*-?\s*md5:\s*([0-9a-f]{32})\s*$", re.MULTILINE)
_SIZE_RE = re.compile(r"^\s*-?\s*size:\s*(\d+)\s*$", re.MULTILINE)
_PATH_RE = re.compile(r"^\s*-?\s*path:\s*(\S+)\s*$", re.MULTILINE)

__all__ = [
    "DEFAULT_BASE_URL",
    "POINTERS",
    "Asset",
    "AssetError",
    "build_assets",
    "fetch_asset",
    "main",
    "md5_of",
    "parse_pointer",
]


class AssetError(RuntimeError):
    """A fetch could not be completed, or the bytes did not verify."""


@dataclass(frozen=True)
class Asset:
    """One DVC-tracked object and where it belongs on disk."""

    pointer: Path
    target: Path
    md5: str
    size: int

    @property
    def name(self) -> str:
        """The release asset name (the file's basename)."""
        return self.target.name


def parse_pointer(pointer: Path) -> tuple[str, int, str]:
    """Read ``(md5, size, path)`` out of a DVC pointer file.

    The pointer is a five-line YAML fragment, so it is parsed with regexes
    rather than a YAML dependency: this script has to run on a clean machine
    with nothing but the standard library.
    """
    text = pointer.read_text(encoding="utf-8")
    md5 = _MD5_RE.search(text)
    size = _SIZE_RE.search(text)
    path = _PATH_RE.search(text)
    if not (md5 and size and path):
        raise AssetError(f"{pointer}: not a DVC pointer (md5/size/path missing)")
    return md5.group(1), int(size.group(1)), path.group(1)


def build_assets(repo_root: Path, pointers: tuple[str, ...] = POINTERS) -> list[Asset]:
    """Resolve every pointer into an :class:`Asset` under ``repo_root``."""
    assets: list[Asset] = []
    for relative in pointers:
        pointer = repo_root / relative
        if not pointer.is_file():
            raise AssetError(f"{pointer}: pointer file not found")
        md5, size, path = parse_pointer(pointer)
        assets.append(
            Asset(
                pointer=pointer,
                target=pointer.parent / path,
                md5=md5,
                size=size,
            )
        )
    return assets


def md5_of(path: Path) -> str:
    """Checksum a file on disk (chunked, so a 136 MB graph stays out of RAM)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            destination.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=CHUNK)
    except urllib.error.HTTPError as error:
        raise AssetError(f"{url}: HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise AssetError(f"{url}: {error.reason}") from error


def fetch_asset(
    asset: Asset,
    base_url: str,
    timeout: float = 120.0,
    force: bool = False,
) -> str:
    """Download one asset into place, verifying its md5.

    Returns ``"cached"`` when the file was already there and verified, and
    ``"downloaded"`` when it was fetched. Raises :class:`AssetError` on a
    transport failure or a checksum mismatch -- a half-written file is never
    left behind, because the download goes to a ``.part`` file that is only
    renamed after it verifies.
    """
    if asset.target.exists() and not force and md5_of(asset.target) == asset.md5:
        return "cached"

    asset.target.parent.mkdir(parents=True, exist_ok=True)
    temp = asset.target.with_name(f"{asset.target.name}.part")
    url = f"{base_url.rstrip('/')}/{asset.name}"
    try:
        _download(url, temp, timeout)
        actual_md5 = md5_of(temp)
        actual_size = temp.stat().st_size
        if actual_md5 != asset.md5:
            raise AssetError(
                f"{asset.target}: md5 mismatch (expected {asset.md5}, got {actual_md5})"
            )
        if actual_size != asset.size:
            raise AssetError(
                f"{asset.target}: size mismatch "
                f"(expected {asset.size}, got {actual_size})"
            )
        os.replace(temp, asset.target)
    finally:
        temp.unlink(missing_ok=True)
    return "downloaded"


def _verify(asset: Asset) -> bool:
    if not asset.target.exists():
        return False
    return md5_of(asset.target) == asset.md5


def _human(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _pointers(raw: str) -> tuple[str, ...]:
    """Parse the comma-separated ``--pointers`` value."""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    """Entry point: fetch (or verify) every asset, exit 0 only if all are good."""
    parser = argparse.ArgumentParser(
        description="Fetch the DVC-tracked serving assets from a public release."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: the checkout this script lives in)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
        help=f"release asset base URL (default: ${BASE_URL_ENV} or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--pointers",
        default=",".join(POINTERS),
        help="comma-separated DVC pointers to fetch (default: every shipped asset)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check the files already on disk; download nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when a verified copy is already present",
    )
    args = parser.parse_args(argv)

    try:
        assets = build_assets(args.repo_root, _pointers(args.pointers))
    except AssetError as error:
        logger.error(str(error))
        return 1

    failures: list[str] = []
    for asset in assets:
        if args.verify_only:
            ok = _verify(asset)
            state = "ok" if ok else "missing or corrupt"
            logger.info(f"{state}: {asset.target} ({_human(asset.size)})")
            if not ok:
                failures.append(str(asset.target))
            continue
        try:
            state = fetch_asset(
                asset, args.base_url, timeout=args.timeout, force=args.force
            )
        except AssetError as error:
            logger.error(f"failed: {error}")
            failures.append(str(error))
            continue
        logger.info(f"{state}: {asset.target} ({_human(asset.size)})")

    if failures:
        logger.error(f"{len(failures)} asset(s) unavailable")
        return 1
    logger.info(f"all {len(assets)} asset(s) present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
