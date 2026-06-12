"""
project.py  —  Minutes of Meeting (MoM) Generator
=================================================

Turn a meeting audio file into a polished Word document of minutes.

What it does, end to end:
    1. Transcribe the audio locally with faster-whisper (no API key, free).
    2. Add approximate speaker labels based on pauses (basic, optional).
    3. Send the transcript to the Groq LLM to extract structured minutes.
    4. Save everything as a formatted .docx file.

-------------------------------------------------------------------------------
HOW TO USE
-------------------------------------------------------------------------------
1. Put your Groq API key in a file named ".env" next to this script:

       GROQ_API_KEY=your_real_key_here

   (Copy .env.example to .env and paste your key. Get a free key at
    https://console.groq.com/keys)

2. Run it from a terminal:

       python project.py path/to/meeting.mp3

   Useful options:
       python project.py meeting.m4a --model small --output output/board.docx
       python project.py meeting.wav --no-diarization
       python project.py meeting.mp3 --language en

The finished .docx lands in the "output/" folder by default.
"""

import argparse
import os
import sys
from datetime import datetime

# python-dotenv reads the .env file and loads GROQ_API_KEY into the
# environment so the rest of the code can find it.
from dotenv import load_dotenv

# Our own package (the three pipeline steps).
from mom_generator import (
    transcribe_audio,
    format_transcript,
    build_mom,
    export_to_docx,
)
from mom_generator.transcriber import assign_basic_speakers


# File types we accept. faster-whisper can handle these (and more), but we
# guard the input so the user gets a friendly message instead of a crash.
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


def parse_arguments():
    """Define and read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate Minutes of Meeting (.docx) from a meeting audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        help="Path to the meeting audio file (.mp3, .wav, or .m4a).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the .docx. Defaults to output/<audio name>_MoM.docx.",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size. Bigger = more accurate but slower.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code like 'en'. Omit to auto-detect.",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip the basic speaker labeling step.",
    )
    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Also save the raw transcript as a .txt file next to the .docx.",
    )
    return parser.parse_args()


def validate_audio_path(audio_path: str):
    """Stop early with a clear message if the audio file is wrong/missing."""
    if not os.path.isfile(audio_path):
        sys.exit(f"ERROR: Audio file not found: {audio_path}")

    extension = os.path.splitext(audio_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        sys.exit(
            f"ERROR: Unsupported file type '{extension}'. "
            f"Please use one of: {supported}"
        )


def decide_output_path(audio_path: str, requested_output: str | None) -> str:
    """Work out where to save the .docx and make sure the folder exists."""
    if requested_output:
        output_path = requested_output
    else:
        # Default: output/<original filename>_MoM.docx
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join("output", f"{base_name}_MoM.docx")

    # Create the containing folder if it does not exist yet.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    return output_path


def main():
    args = parse_arguments()

    # Load GROQ_API_KEY from .env (if present) into the environment.
    load_dotenv()

    # --- Friendly checks before doing any heavy work ----------------------
    validate_audio_path(args.audio)
    if not os.getenv("GROQ_API_KEY"):
        sys.exit(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Create a .env file (copy .env.example) and add your Groq key:\n"
            "    GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com/keys"
        )

    output_path = decide_output_path(args.audio, args.output)

    started = datetime.now()
    print("=" * 60)
    print(" Minutes of Meeting Generator")
    print("=" * 60)

    # --- Step 1: Transcribe ----------------------------------------------
    segments = transcribe_audio(
        args.audio,
        model_size=args.model,
        language=args.language,
    )
    if not segments:
        sys.exit("ERROR: No speech was detected in the audio. Nothing to do.")

    # --- Step 2: Basic speaker labels (optional) -------------------------
    use_speakers = not args.no_diarization
    if use_speakers:
        segments = assign_basic_speakers(segments)
        print("[project] Added approximate speaker labels "
              "(rough guess based on pauses).")

    transcript = format_transcript(segments, with_speakers=use_speakers)

    # Optionally save the raw transcript so the user can inspect it.
    if args.save_transcript:
        transcript_path = os.path.splitext(output_path)[0] + "_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"[project] Transcript saved to: {transcript_path}")

    # --- Step 3: Build structured minutes with Groq ----------------------
    mom = build_mom(transcript)

    # --- Step 4: Write the .docx -----------------------------------------
    export_to_docx(mom, output_path)

    elapsed = (datetime.now() - started).total_seconds()
    print("=" * 60)
    print(f" Done in {elapsed:.0f}s")
    print(f" Your minutes: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
