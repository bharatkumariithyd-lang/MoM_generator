"""
diarizer.py
===========

REAL speaker recognition (diarization) using Resemblyzer.

Unlike transcriber.assign_basic_speakers() — which only *guesses* a speaker
change whenever there is a long pause — this module actually listens to the
VOICES. That means it can tell that the person speaking at minute 1 and the
person speaking at minute 10 are the *same* human, something the pause trick can
never do.

How it works (the whole idea in five steps):

    1. Load the audio at 16 kHz.
    2. Slide a ~1.6s window across it and turn each window into a "voiceprint"
       — a 256-number vector (an embedding) that captures the timbre of the
       voice in that window. Same person -> similar vectors.
    3. Cluster the voiceprints. Each cluster = one real speaker. (If you don't
       tell us how many people there are, we estimate it.)
    4. Merge neighbouring windows of the same speaker into "turns"
       (start, end, "Speaker N").
    5. Match those turns against the Whisper text segments by time overlap, so
       each line of transcript gets the right speaker label.

IMPORTANT — timeline alignment:
We deliberately do NOT use resemblyzer.preprocess_wav(), because it shortens
silences and would shift every timestamp, knocking the voiceprint timeline out
of sync with the Whisper transcript. Instead we load the raw waveform at 16 kHz
ourselves, so diarization time == real audio time == Whisper time.
"""

from dataclasses import dataclass

import numpy as np


# Resemblyzer's model always works at 16 kHz; its returned wav slices are in
# samples, so dividing a sample index by this gives the time in seconds.
SAMPLING_RATE = 16000

# When auto-detecting the number of speakers we try 2..MAX_AUTO_SPEAKERS and
# keep whichever split looks cleanest (best "silhouette" score).
MAX_AUTO_SPEAKERS = 6

# If even the best multi-speaker split looks weak (everyone sounds like one
# blurry cluster), we assume there was really just a single speaker. This is the
# quality bar for "yes, there are genuinely multiple distinct voices".
SINGLE_SPEAKER_SILHOUETTE_CUTOFF = 0.10


@dataclass
class SpeakerTurn:
    """One continuous stretch of speech attributed to a single speaker."""
    start: float    # seconds
    end: float      # seconds
    speaker: str    # e.g. "Speaker 1"


def diarize_audio(audio_path: str, num_speakers: int | None = None, rate: float = 1.6):
    """
    Listen to `audio_path` and return a list of SpeakerTurn objects describing
    who spoke when.

    Parameters
    ----------
    audio_path : str
        Path to the meeting audio (.mp3 / .wav / .m4a — anything librosa reads).
    num_speakers : int | None
        If you know how many people are in the meeting, pass it for best
        accuracy. Leave as None to auto-detect.
    rate : float
        How many voiceprint windows to take per second. 1.6 gives a new window
        roughly every 0.6s, a good balance of detail vs. speed.

    Returns
    -------
    list[SpeakerTurn]
    """
    # Heavy imports kept inside the function so that merely importing this
    # module stays cheap (same pattern as transcriber.py).
    import librosa
    from resemblyzer import VoiceEncoder

    print("[diarizer] Loading audio for voice analysis...")
    # sr=SAMPLING_RATE resamples to 16 kHz but PRESERVES the real timeline
    # (no silence trimming), so timestamps stay aligned with the transcript.
    wav, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)

    print("[diarizer] Computing voiceprints (this is the part that 'listens')...")
    encoder = VoiceEncoder()  # downloads a small pretrained model the first time
    _, partial_embeds, wav_slices = encoder.embed_utterance(
        wav, return_partials=True, rate=rate
    )

    # Convert each window's sample-slice into a (start, end) time in seconds.
    times = [(s.start / SAMPLING_RATE, s.stop / SAMPLING_RATE) for s in wav_slices]

    # Group the voiceprints into speakers.
    labels = _cluster_voiceprints(partial_embeds, num_speakers)
    labels = _smooth_labels(labels)
    speaker_names = _name_clusters_by_talk_time(labels)

    turns = _labels_to_turns(times, labels, speaker_names)

    n_speakers = len({t.speaker for t in turns})
    print(f"[diarizer] Identified {n_speakers} distinct speaker(s) "
          f"across {len(turns)} turn(s).")
    return turns


def _cluster_voiceprints(embeds: np.ndarray, num_speakers: int | None):
    """
    Group voiceprints so that each group is one person.

    We use agglomerative clustering with a COSINE distance, because two
    voiceprints of the same person point in nearly the same direction
    regardless of loudness.

    If `num_speakers` is given we cluster into exactly that many groups.
    Otherwise we try several counts and keep the one with the best silhouette
    score (a standard measure of "how cleanly separated are the clusters"),
    falling back to a single speaker if nothing separates well.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(embeds)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    # Caller told us the number of speakers -> just use it.
    if num_speakers and num_speakers >= 1:
        k = min(num_speakers, n)
        if k == 1:
            return np.zeros(n, dtype=int)
        return AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeds)

    # Auto-detect: try 2..MAX and keep the cleanest split.
    best_labels = np.zeros(n, dtype=int)   # default: everyone is one speaker
    best_score = -1.0
    upper = min(MAX_AUTO_SPEAKERS, n - 1)
    for k in range(2, upper + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeds)
        score = silhouette_score(embeds, labels, metric="cosine")
        if score > best_score:
            best_score, best_labels = score, labels

    # Too blurry to be sure there are multiple voices -> call it one speaker.
    if best_score < SINGLE_SPEAKER_SILHOUETTE_CUTOFF:
        return np.zeros(n, dtype=int)
    return best_labels


def _smooth_labels(labels: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Remove single-window jitter with a small majority vote.

    Clustering occasionally mislabels one lone window in the middle of a turn.
    Replacing each label with the most common label in a tiny neighbourhood
    cleans up those one-off flips without blurring real speaker changes.
    """
    n = len(labels)
    if n < window or window < 3:
        return labels
    half = window // 2
    smoothed = labels.copy()
    for i in range(half, n - half):
        neighbourhood = labels[i - half:i + half + 1]
        values, counts = np.unique(neighbourhood, return_counts=True)
        smoothed[i] = values[np.argmax(counts)]
    return smoothed


def _name_clusters_by_talk_time(labels: np.ndarray) -> dict:
    """
    Map raw cluster ids (0, 1, 2 ...) to friendly names, with the person who
    talked the most becoming "Speaker 1", the next "Speaker 2", and so on.
    """
    if len(labels) == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    # Order cluster ids from most-spoken to least-spoken.
    ordered = [val for val, _ in sorted(zip(values, counts),
                                        key=lambda vc: vc[1], reverse=True)]
    return {cluster_id: f"Speaker {rank}"
            for rank, cluster_id in enumerate(ordered, start=1)}


def _labels_to_turns(times, labels, speaker_names) -> list:
    """
    Merge consecutive same-speaker windows into continuous turns.

    Input is per-window (time range + cluster label); output is the tidy list of
    SpeakerTurn(start, end, "Speaker N") we actually use downstream.
    """
    turns: list[SpeakerTurn] = []
    for (start, end), label in zip(times, labels):
        name = speaker_names.get(label, "Speaker 1")
        # Extend the current turn if it's the same speaker, else start a new one.
        if turns and turns[-1].speaker == name:
            turns[-1].end = end
        else:
            turns.append(SpeakerTurn(start=start, end=end, speaker=name))
    return turns


def assign_voice_speakers(segments, turns):
    """
    Stamp each transcript Segment with the speaker whose turn it overlaps most.

    `segments` are transcriber.Segment objects (they have .start/.end/.speaker).
    We pick, for each segment, the speaker turn sharing the most time with it.
    If a segment sits in a gap with no overlapping turn, we fall back to the
    nearest turn by midpoint so every line still gets a label.

    Returns the same `segments` list, now with .speaker filled in.
    """
    if not turns:
        return segments

    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        if best_speaker is None:
            # No overlap at all -> attach to the nearest turn by midpoint.
            seg_mid = 0.5 * (seg.start + seg.end)
            best_speaker = min(
                turns,
                key=lambda t: abs(0.5 * (t.start + t.end) - seg_mid),
            ).speaker

        seg.speaker = best_speaker

    return segments
