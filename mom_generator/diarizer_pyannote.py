"""
diarizer_pyannote.py
====================

A SECOND, higher-accuracy speaker-diarization option, built on pyannote.audio.

Why a separate file from diarizer.py?
    diarizer.py uses Resemblyzer — light, no token, but it tends to under-count
    speakers and lump a fast multi-person discussion into one label. pyannote is
    a dedicated diarization model and is usually much more accurate, but it is
    heavier and its pretrained models are GATED on Hugging Face. Keeping it in
    its own module means importing the rest of the project stays cheap, and users
    who don't want pyannote never need to install it.

How it plugs in:
    This function returns the SAME `SpeakerTurn` objects diarizer.py produces, so
    the existing `assign_voice_speakers()` maps them onto the transcript with no
    changes anywhere else. pyannote is just another *producer* of speaker turns.

One-time setup before this can run (see requirements-pyannote.txt):
    1. Accept the model terms at
       huggingface.co/pyannote/speaker-diarization-community-1
    2. Create a token (with public-gated-repo read access) and put it in .env
       as  HF_TOKEN=hf_...
"""

import os

# Some networks (firewalls / DPI / antivirus) forcibly reset connections to
# Hugging Face's newer "Xet" download CDN — it shows up as a WinError 10054
# "connection forcibly closed" mid-download. Falling back to the classic LFS
# transfer avoids that. setdefault so an explicit env var still wins.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Reuse the exact dataclass (and 16 kHz rate) the Resemblyzer diarizer uses, so
# downstream code (assign_voice_speakers, formatting, the MoM step) can't tell
# the difference between the two diarization back-ends.
from .diarizer import SpeakerTurn, SAMPLING_RATE

# pyannote.audio 4.x's flagship open pipeline. It bundles its own segmentation
# and embedding, so this is the only gated repo whose terms you must accept.
MODEL_ID = "pyannote/speaker-diarization-community-1"


def diarize_audio_pyannote(audio_path: str, num_speakers: int | None = None):
    """
    Run pyannote diarization and return a list of SpeakerTurn objects.

    Parameters
    ----------
    audio_path : str
        Path to the meeting audio.
    num_speakers : int | None
        If you know how many people spoke, pass it for best accuracy. Leave as
        None to let pyannote estimate.

    Returns
    -------
    list[SpeakerTurn]  (start, end, "Speaker N") sorted by start time.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "pyannote needs a Hugging Face token. One-time setup:\n"
            "  1. Accept the terms at\n"
            "       https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2. Create a token at https://huggingface.co/settings/tokens WITH\n"
            "     'Read access to contents of all public gated repos you can access'\n"
            "  3. Add it to your .env as  HF_TOKEN=hf_xxx"
        )

    # Heavy imports kept inside the function (same pattern as the other modules).
    import librosa
    import torch
    from pyannote.audio import Pipeline

    print(f"[pyannote] Loading '{MODEL_ID}' (first run downloads it, please wait)...")
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    if pipeline is None:
        # from_pretrained returns None when the token can't access the gated model.
        raise RuntimeError(
            "Could not load the pyannote model. Most likely the model terms have "
            "not been accepted with this token, or the token is wrong. See the "
            "setup steps above."
        )

    # Use the GPU automatically when one is visible (same idea as the transcriber).
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("[pyannote] Running on GPU (cuda).")
    else:
        print("[pyannote] Running on CPU — this is slow on long files.")

    # Passing num_speakers pins the count; omitting it lets pyannote estimate.
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    # Load the audio ourselves at 16 kHz mono and hand pyannote an in-memory
    # waveform. pyannote 4.x otherwise decodes via torchcodec, which needs a full
    # FFmpeg install (with DLLs) that many machines lack; passing the waveform
    # directly sidesteps that and reuses the librosa we already depend on.
    print("[pyannote] Loading audio...")
    wav, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    waveform = torch.from_numpy(wav).unsqueeze(0)  # shape (1, num_samples)

    print("[pyannote] Analysing voices (this is the slow part)...")
    result = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLING_RATE}, **kwargs
    )

    # pyannote 4.x returns a DiarizeOutput; its `exclusive_speaker_diarization`
    # is an Annotation with NO overlapping turns — exactly right for mapping each
    # transcript line to a single speaker. Older versions return an Annotation
    # directly (which already has .itertracks()), so fall back to that.
    annotation = getattr(result, "exclusive_speaker_diarization", result)

    # pyannote labels speakers "SPEAKER_00", "SPEAKER_01", ... Collect the turns.
    raw_turns = [
        (segment.start, segment.end, label)
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    if not raw_turns:
        return []

    # Rename to "Speaker N" with the most-talkative person as Speaker 1 — the same
    # convention the Resemblyzer diarizer uses, so labels stay consistent whichever
    # option the user picks.
    talk_time: dict[str, float] = {}
    for start, end, label in raw_turns:
        talk_time[label] = talk_time.get(label, 0.0) + (end - start)
    ordered = sorted(talk_time, key=lambda lbl: talk_time[lbl], reverse=True)
    rename = {label: f"Speaker {i}" for i, label in enumerate(ordered, start=1)}

    turns = [
        SpeakerTurn(start=start, end=end, speaker=rename[label])
        for start, end, label in raw_turns
    ]
    turns.sort(key=lambda t: t.start)

    print(f"[pyannote] Identified {len(rename)} distinct speaker(s) "
          f"across {len(turns)} turn(s).")
    return turns
