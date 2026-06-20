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
    device: str = "auto",
    compute_type: str = "auto",
    hotwords: str | None = None,
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
        "auto" (default) uses "cuda" if an NVIDIA GPU is visible (e.g. a Colab
        T4 — much faster), otherwise "cpu". Pass "cpu" or "cuda" to force one.
    compute_type : str
        "auto" (default) pairs with the device: "float16" on GPU, "int8" on CPU.
    hotwords : str | None
        Domain terms or names to bias the model toward (e.g. product names,
        acronyms, or unusual personal names). Useful for rare words the model
        otherwise mishears as a more common look-alike. Only list the stubborn
        ones; you do not need to list every word.

    Returns
    -------
    list[Segment]
    """
    # Imported here (not at the top of the file) so that simply importing this
    # module is cheap. The heavy faster-whisper import only happens when you
    # actually transcribe something.
    from faster_whisper import WhisperModel
    import ctranslate2
    from .model_downloader import ensure_local_model

    # Pick the fastest device we can. On a normal laptop this stays on CPU; on a
    # machine with a visible NVIDIA GPU (e.g. a free Colab T4) it switches to
    # CUDA + float16, which transcribes large-v3 roughly 20-40x faster. "auto"
    # is the default; passing an explicit device/compute_type overrides it.
    if device == "auto":
        has_gpu = ctranslate2.get_cuda_device_count() > 0
        device = "cuda" if has_gpu else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if has_gpu else "int8"
    elif compute_type == "auto":
        compute_type = "int8"
    print(f"[transcriber] Device: {device} (compute_type={compute_type}).")

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
    #
    # `hotwords` biases the decoder toward the listed terms across the whole
    # audio, which is what rescues a rare term from being heard as a more common
    # look-alike word. Passing None simply has no effect.
    #
    # The remaining settings harden the decoder against HALLUCINATIONS — Whisper's
    # habit of inventing fluent-sounding text over silence or noise (e.g. a
    # garbled pre-meeting intro). None of them rewrite recognised words; they only
    # configure how the decoder behaves, so there is no risk of inventing content
    # (unlike a later LLM cleanup pass would have):
    #   * condition_on_previous_text=True   — Whisper's default: feed each 30s
    #     window the previous text so long monologues stay properly punctuated and
    #     capitalised. (We tested False to curb snowballing repeats, but on real
    #     meeting audio it made the text run-on/lowercase mid-utterance, so we
    #     reverted to True — readability mattered more than the rare repeat.)
    #   * word_timestamps=True              — required for the silence filter below;
    #     also gives cleaner segment boundaries.
    #   * hallucination_silence_threshold   — distrust/skip likely-invented text
    #     that appears across silent gaps longer than this many seconds.
    #   * initial_prompt                    — prime a formal meeting style so clean
    #     speech transcribes with better punctuation and capitalisation.
    #   * vad_parameters                    — trim dead air a little more eagerly
    #     (lower min_silence) while keeping generous padding so we do NOT clip
    #     quiet, one-word replies like "Yes." / "Yeah.".
    segments_generator, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        hotwords=hotwords,
        condition_on_previous_text=True,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
        initial_prompt="The following is a professional meeting transcript.",
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        ),
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

    This is the text we later hand to the Groq LLM, and (with --save-transcript)
    what we write to the .txt file. To keep it readable we GROUP consecutive
    lines by the same speaker: the speaker label and that turn's timestamp are
    printed once at the top of the block, then each line of the turn follows.

    Example:
        [Speaker 1] (01:15)
        Let's move the deadline to Friday.
        And assign the final report to marketing.

        [Speaker 2] (01:28)
        That works for me.
    """
    if not segments:
        return ""

    lines: list[str] = []
    current_speaker = None

    for seg in segments:
        if with_speakers and seg.speaker:
            # Start a new block whenever the speaker changes (and for the first).
            if seg.speaker != current_speaker:
                if lines:  # blank line between blocks, but not before the first
                    lines.append("")
                timestamp = _format_timestamp(seg.start)
                lines.append(f"[{seg.speaker}] ({timestamp})")
                current_speaker = seg.speaker
            lines.append(seg.text)
        else:
            # No speaker labels: keep it simple — one timestamped line per segment.
            timestamp = _format_timestamp(seg.start)
            lines.append(f"[{timestamp}] {seg.text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Speaker NAMING — turn the anonymous "Speaker 1/2/3" labels diarization gives
# us into the real people's names. Diarization can tell voices apart but has no
# way to know *who* each voice is, so a human supplies the mapping (in the CLI
# via --speaker-names, or in the web UI via text boxes). These three helpers are
# shared by both front-ends.
# ---------------------------------------------------------------------------
def speaker_samples(segments: list[Segment], max_chars: int = 90) -> dict:
    """
    Return ``{speaker_label: a short sample line}`` so a human can tell which
    detected speaker is who before naming them.

    For each speaker we keep their first line, but upgrade to a longer one if the
    first was very short (so the sample isn't just "Yes."). Order follows first
    appearance in the meeting.
    """
    samples: dict[str, str] = {}
    for seg in segments:
        speaker, text = seg.speaker, seg.text.strip()
        if not speaker or not text:
            continue
        if speaker not in samples or (
            len(samples[speaker]) < 15 and len(text) > len(samples[speaker])
        ):
            samples[speaker] = text
    return {
        label: (text[:max_chars] + ("…" if len(text) > max_chars else ""))
        for label, text in samples.items()
    }


def parse_speaker_names(spec: str | None) -> dict:
    """
    Parse a CLI mapping like ``"Speaker 1=Bharat, Speaker 2=Dr. Rao"`` into
    ``{"Speaker 1": "Bharat", "Speaker 2": "Dr. Rao"}``.

    Splits on commas between pairs and on the FIRST ``=`` within a pair (so names
    may contain ``=`` if ever needed). Blank or malformed pieces are ignored.
    """
    mapping: dict[str, str] = {}
    if not spec:
        return mapping
    for pair in spec.split(","):
        if "=" not in pair:
            continue
        label, name = pair.split("=", 1)
        label, name = label.strip(), name.strip()
        if label and name:
            mapping[label] = name
    return mapping


def rename_speakers(segments: list[Segment], name_map: dict) -> list[Segment]:
    """
    Replace each segment's ``.speaker`` using ``name_map`` (e.g.
    ``{"Speaker 1": "Bharat"}``). Labels not in the map keep their original value,
    so a partial mapping is fine. Returns the same list, mutated in place.
    """
    if not name_map:
        return segments
    for seg in segments:
        if seg.speaker in name_map:
            seg.speaker = name_map[seg.speaker]
    return segments
