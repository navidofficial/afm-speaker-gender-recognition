"""
Download VoxCeleb1 test audio (vox1_test_wav.zip) with resume support.

Usage:
  python scripts/download_voxceleb1_test.py
  python scripts/download_voxceleb1_test.py --extract
  python scripts/download_voxceleb1_test.py --dest "E:/path/to/data/VoxCeleb1"

Notes:
  - Official Oxford URL often returns 401 without auth, so this uses Hugging Face.
  - Safe to re-run: resumes from the existing partial file.
  - Expected zip size: ~1.00 GB (1,072,793,438 bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# Hugging Face mirror of the official VoxCeleb1 test wav archive
DEFAULT_URL = (
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/"
    "resolve/main/vox1/vox1_test_wav.zip"
)
EXPECTED_SIZE = 1_072_793_438  # bytes
EXPECTED_SHA256 = "8de57f347fe22b2c24526e9f444f689ecf5096fc2a92018cf420ff6b5b15eaea"
CHUNK = 1024 * 1024  # 1 MB


def default_dest_dir() -> Path:
    # .../paper1_implement2/scripts/this_file.py -> .../paper1_implement2/data/VoxCeleb1
    return Path(__file__).resolve().parents[1] / "data" / "VoxCeleb1"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = out_path.stat().st_size if out_path.exists() else 0

    if existing == EXPECTED_SIZE:
        print(f"[ok] already complete: {out_path} ({human(existing)})")
        return

    if existing > EXPECTED_SIZE:
        print(f"[warn] file larger than expected ({human(existing)}); deleting and restarting")
        out_path.unlink()
        existing = 0

    headers = {"User-Agent": "voxceleb1-test-downloader/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] from byte {existing:,} ({human(existing)})")
    else:
        print("[start] fresh download")

    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as resp:
        # Some servers ignore Range and resend full file
        content_range = resp.headers.get("Content-Range", "")
        mode = "ab"
        if existing > 0 and not content_range:
            print("[warn] server ignored Range; restarting from 0")
            existing = 0
            mode = "wb"
        elif existing == 0:
            mode = "wb"

        total = EXPECTED_SIZE
        cl = resp.headers.get("Content-Length")
        if cl and content_range:
            # Content-Length is remaining bytes when Range is honored
            total = existing + int(cl)
        elif cl and existing == 0:
            total = int(cl)

        print(f"[info] saving to: {out_path}")
        print(f"[info] target size: ~{human(EXPECTED_SIZE)}")

        with open(out_path, mode) as f:
            downloaded = existing
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                pct = 100.0 * downloaded / total if total else 0.0
                print(
                    f"\r[progress] {human(downloaded)} / {human(total)} ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
        print()

    final = out_path.stat().st_size
    if final != EXPECTED_SIZE:
        raise SystemExit(
            f"[error] incomplete download: got {final:,} bytes, "
            f"expected {EXPECTED_SIZE:,}. Re-run the script to resume."
        )
    print(f"[ok] download complete ({human(final)})")


def verify_sha256(path: Path, expected: str) -> None:
    print("[info] verifying SHA-256 (may take a minute)...")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
    digest = h.hexdigest()
    if digest != expected:
        raise SystemExit(f"[error] SHA-256 mismatch:\n  got {digest}\n  expected {expected}")
    print("[ok] SHA-256 verified")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"[info] extracting to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    wavs = list(dest_dir.rglob("*.wav"))
    print(f"[ok] extracted. wav files found: {len(wavs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoxCeleb1 test wav zip")
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest_dir(),
        help="Destination folder (default: <project>/data/VoxCeleb1)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Download URL (default: Hugging Face mirror)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract zip after successful download",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA-256 verification",
    )
    args = parser.parse_args()

    dest_dir: Path = args.dest
    zip_path = dest_dir / "vox1_test_wav.zip"

    print("=== VoxCeleb1 test download ===")
    print(f"URL : {args.url}")
    print(f"Out : {zip_path}")
    print("This is the official test split (~40 speakers, ~1 GB), used as VoxCeleb eval data.\n")

    try:
        download(args.url, zip_path)
        if not args.skip_hash:
            verify_sha256(zip_path, EXPECTED_SHA256)
        if args.extract:
            extract_zip(zip_path, dest_dir)
        print("\nDone. Use data/VoxCeleb1 (test wavs) as your VoxCeleb evaluation set.")
    except KeyboardInterrupt:
        print("\n[interrupted] partial file kept; re-run the same command to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
