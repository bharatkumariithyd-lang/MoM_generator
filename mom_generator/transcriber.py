"""
transcriber.py
==============

Step 1 of the pipeline: turn an audio file into text using faster-whisper.

faster-whisper runs the Whisper speech-to-text model *locally* on your
machine. It is free and needs no API key. The first time you run it, it will
download the chosen model (a few hundred MB) and cache it, so the first run is
slower than later runs.

We also do a very *basic* speaker separation here. Read the big note above
`assign_basic_speakers()` before trusting those labels — it is a rough guess,
not real voice identification.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# A simple container for one chunk ("segment") of transcribed speech.
# Using a dataclass keeps the rest of the code readable: seg.text, seg.start...
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    start: float        # start time in seconds
    end: float          # end time in seconds
    text: str           # the transcribed words for this chunk
    speaker: str = ""   # filled in later by assign_basic_speakers()


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
):
    """
    Transcribe an audio file and return a list of Segment objects.

    Parameters
    ----------
    audio_path : str
        Path to the .mp3, .wav, or .m4a file.
    model_size : str
        Whisper model size. Bigger = more accurate but slower.
        Options: "tiny", "base", "small", "medium", "large-v3",
        "large-v3-turbo" (distilled large-v3: similar accuracy, much faster).
        "base" is a good starting point on a normal laptop CPU.
    language : str | None
        Language code like "en". Leave as None to let Whisper auto-detect.
    device : str
        "cpu" (works everywhere) or "cuda" (only if you have an NVIDIA GPU).
    compute_type : str
        "int8" is fast and light on CPU. Use "float16" on a GPU.

    Returns
    -------
    list[Segment]
    """
    # Imported here (not at the top of the file) so that simply importing this
    # module is cheap. The heavy faster-whisper import only happens when you
    # actually transcribe something.
    from faster_whisper import WhisperModel
    from .model_downloader import ensure_local_model

    # Make sure the model files are available locally (downloading them with
    # retries if needed — see model_downloader.py for why). This returns a
    # local folder path that we then load the model from.
    print(f"[transcriber] Preparing Whisper model '{model_size}' "
          f"(first run downloads it, please wait)...")
    model_path = ensure_local_model(model_size)

    print(f"[transcriber] Loading model from: {model_path}")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)

    print(f"[transcriber] Transcribing: {audio_path}")
    # `segments` is a generator; `info` holds detected language etc.
    # vad_filter=True uses voice-activity detection to skip silent parts,
    # which improves both speed and quality.
    segments_generator, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    print(f"[transcriber] Detected language: {info.language} "
          f"(probability {info.language_probability:.2f})")

    # Pull the generator into a normal list of our simple Segment objects.
    segments: list[Segment] = []
    for seg in segments_generator:
        text = seg.text.strip()
        if text:  # skip empty chunks
            segments.append(Segment(start=seg.start, end=seg.end, text=text))

    print(f"[transcriber] Done. {len(segments)} speech segments found.")
    return segments


# ---------------------------------------------------------------------------
# BASIC speaker separation — please read this honestly.
#
# True speaker diarization ("who spoke when") needs a dedicated model such as
# pyannote.audio, which requires PyTorch and a Hugging Face token. To keep this
# project simple and dependency-light, we do NOT do that here.
#
# Instead we use a crude heuristic: whenever there is a long SILENCE between two
# segments, we assume the speaker probably changed and we flip to the "next"
# speaker label. This only detects *turn changes*; it cannot tell whether two
# different turns belong to the same real person. Treat the labels as a helpful
# hint, not ground truth.
#
# To upgrade to real diarization later, replace this function with a
# pyannote.audio pipeline and map its speaker IDs onto these segments.
# ---------------------------------------------------------------------------
def assign_basic_speakers(
    segments: list[Segment],
    pause_threshold: float = 1.5,
    max_speakers: int = 4,
):
    """
    Label each segment with an approximate "Speaker N" based on silence gaps.

    Parameters
    ----------
    segments : list[Segment]
        Segments from transcribe_audio().
    pause_threshold : float
        A silence longer than this many seconds is treated as a likely
        speaker change.
    max_speakers : int
        We cycle speaker labels within this range (Speaker 1 .. Speaker N).

    Returns
    -------
    list[Segment]  (the same list, now with .speaker filled in)
    """
    if not segments:
        return segments

    speaker_index = 1
    previous_end = segments[0].start

    for seg in segments:
        gap = seg.start - previous_end
        # A long pause -> guess that a new person started talking.
        if gap > pause_threshold:
            speaker_index += 1
            if speaker_index > max_speakers:
                speaker_index = 1  # cycle back to the first speaker
        seg.speaker = f"Speaker {speaker_index}"
        previous_end = seg.end

    return segments


def _format_timestamp(seconds: float) -> str:
    """Turn 75.4 seconds into '01:15' for a readable transcript."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(segments: list[Segment], with_speakers: bool = True) -> str:
    """
    Build a single human-readable transcript string from the segments.

    This is the text we will later hand to the Groq LLM. Including timestamps
    and (approximate) speaker labels gives the model useful structure.

    Example line:
        [01:15] Speaker 2: Let's move the deadline to Friday.
    """
    lines = []
    for seg in segments:
        timestamp = _format_timestamp(seg.start)
        if with_speakers and seg.speaker:
            lines.append(f"[{timestamp}] {seg.speaker}: {seg.text}")
        else:
            lines.append(f"[{timestamp}] {seg.text}")
    return "\n".join(lines)
