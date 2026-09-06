"""
build_zip.py — package the project for Google Colab upload.

Run:  python build_zip.py

Creates MoM_generator.zip containing just the code (project.py, the
mom_generator package, and the requirements files) — NOT .venv / models /
output / audio. Uses forward-slash paths so the zip extracts correctly on
Linux (Colab); Windows' Compress-Archive / right-click "Send to zip" write
backslash paths that Colab mis-reads as flat filenames.
"""

import os
import zipfile

OUTPUT = "MoM_generator.zip"
TOP_FILES = ["project.py", "requirements.txt", "requirements-voice.txt",
             "requirements-pyannote.txt"]
PACKAGE = "mom_generator"


def main():
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in TOP_FILES:
            z.write(f, f)
        for root, _dirs, fnames in os.walk(PACKAGE):
            if "__pycache__" in root:
                continue
            for fn in fnames:
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(root, fn)
                z.write(full, full.replace(os.sep, "/"))  # forward slashes

    with zipfile.ZipFile(OUTPUT) as z:
        names = z.namelist()
    print(f"Wrote {OUTPUT} with {len(names)} files:")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
