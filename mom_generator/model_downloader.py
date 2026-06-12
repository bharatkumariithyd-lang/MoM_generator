"""
model_downloader.py
===================

Robustly fetch the faster-whisper model files and cache them locally.

WHY THIS EXISTS
---------------
Normally faster-whisper downloads its model from huggingface.co automatically.
But some networks/ISPs block or randomly reset connections to huggingface.co.
On such networks the normal download fails with errors like
"WinError 10054: An existing connection was forcibly closed".

This module works around that by:
  1. Using a configurable endpoint (set HF_ENDPOINT in your .env to a mirror
     such as https://hf-mirror.com if the main site is blocked).
  2. Downloading each model file with Python's own network stack and RETRYING
     several times, because the blocking is often intermittent.
  3. Caching the files under  models/<model-name>/  so it only downloads once.

If your internet works normally, this simply downloads the files the first time
and reuses them afterwards.
"""

import os
import time
import ssl
import urllib.request
import urllib.error


# Which Hugging Face repo holds each Whisper model size.
REPO_BY_SIZE = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Files we try to download. The first three are required; the rest are
# best-effort (different model repos include different optional files, so a
# "404 not found" on these is normal and simply skipped).
REQUIRED_FILES = ["config.json", "model.bin", "tokenizer.json"]
OPTIONAL_FILES = ["vocabulary.txt", "vocabulary.json", "preprocessor_config.json"]


def _endpoint() -> str:
    """The base host to download from (mirror-aware)."""
    # HF_ENDPOINT lets the user point at a mirror when huggingface.co is blocked.
    return os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _download_file(url: str, destination: str, max_retries: int = 15) -> bool:
    """
    Download a single file, retrying on connection errors.

    Returns True on success, False if the file genuinely does not exist (404).
    Raises ConnectionError if it never succeeds after all retries.
    """
    ssl_context = ssl.create_default_context()

    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
                data = response.read()
            with open(destination, "wb") as f:
                f.write(data)
            return True

        except urllib.error.HTTPError as error:
            # 404 = this optional file isn't in the repo. Don't retry.
            if error.code == 404:
                return False
            # Other HTTP errors: worth a retry (server hiccup, etc.).
            print(f"    [{attempt}/{max_retries}] HTTP {error.code}, retrying...")

        except Exception as error:
            # Connection reset / timeout — exactly the intermittent block we
            # are trying to outlast. Wait briefly and try again.
            print(f"    [{attempt}/{max_retries}] {type(error).__name__}, retrying...")

        time.sleep(1.5)

    raise ConnectionError(
        f"Could not download after {max_retries} attempts:\n  {url}\n"
        f"Your network may be blocking the download host. Try setting a mirror "
        f"in your .env file:  HF_ENDPOINT=https://hf-mirror.com"
    )


def ensure_local_model(model_size: str, models_dir: str = "models") -> str:
    """
    Make sure the model files for `model_size` exist locally and return the
    folder path that faster-whisper's WhisperModel can load from.

    Already-downloaded files are reused, so this is fast after the first run.
    """
    if model_size not in REPO_BY_SIZE:
        # If the user passed a custom path/repo, just hand it straight back to
        # faster-whisper and let it deal with it.
        return model_size

    repo = REPO_BY_SIZE[model_size]
    local_dir = os.path.join(models_dir, f"faster-whisper-{model_size}")
    os.makedirs(local_dir, exist_ok=True)

    # Fast path: if every required file is already cached, we're done. This
    # also avoids re-probing the (non-existent) optional files on every run.
    already_cached = all(
        os.path.exists(os.path.join(local_dir, f))
        and os.path.getsize(os.path.join(local_dir, f)) > 0
        for f in REQUIRED_FILES
    )
    if already_cached:
        return local_dir

    base_url = f"{_endpoint()}/{repo}/resolve/main/"

    # --- Required files ---------------------------------------------------
    for filename in REQUIRED_FILES:
        target = os.path.join(local_dir, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue  # already cached
        print(f"  Downloading {filename} ...")
        found = _download_file(base_url + filename, target)
        if not found:
            raise FileNotFoundError(
                f"Required model file '{filename}' was not found in repo '{repo}'."
            )

    # --- Optional files (skip silently if missing) ------------------------
    for filename in OPTIONAL_FILES:
        target = os.path.join(local_dir, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        print(f"  Downloading {filename} (optional) ...")
        try:
            _download_file(base_url + filename, target)
        except ConnectionError:
            pass  # optional file, don't fail the whole run over it
        # If the file was a 404, _download_file returns False and leaves an
        # empty file; remove it so it doesn't look like a valid cached file.
        if os.path.exists(target) and os.path.getsize(target) == 0:
            os.remove(target)

    return local_dir
