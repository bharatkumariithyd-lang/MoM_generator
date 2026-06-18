# 📝 Minutes of Meeting (MoM) Generator

Turn a meeting **audio file** into a clean, professional **Minutes of Meeting**
Word document — automatically.

```
  meeting.mp3  ──▶  Transcribe (local)  ──▶  Structure (Groq LLM)  ──▶  meeting_MoM.docx
```

---

## How it works

| Step | What happens | Tool used |
|------|--------------|-----------|
| 1. Transcribe | The audio is converted to text **on your machine** (free, no API key). | `faster-whisper` |
| 2. Speaker labels | Either a quick guess from pauses, **or real voice recognition** that tells voices apart. | (heuristic) or `resemblyzer` |
| 3. Structure | The transcript is sent to an LLM that extracts title, attendees, decisions, action items, etc. | `groq` (`llama-3.3-70b-versatile`) |
| 4. Export | Everything is written into a formatted Word file. | `python-docx` |

> **Two ways to label speakers:**
> - `--speakers pause` (default): a rough guess from silence gaps. Fast, but it
>   only detects *turn changes* — it can't tell if two turns are the same person.
> - `--speakers voice`: **real voice recognition** (Resemblyzer). It listens to
>   the actual voices, so the same person gets the same label every time they
>   speak. Needs the optional extras — see
>   [Enabling real voice recognition](#enabling-real-voice-recognition).

---

## Setup (one time)

1. **Install the dependencies** (inside your virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key.** Copy the template and paste your key:

   ```bash
   copy .env.example .env        # Windows
   # cp .env.example .env        # macOS/Linux
   ```

   Then edit `.env`:

   ```
   GROQ_API_KEY=your_real_key_here
   ```

   Get a free key at <https://console.groq.com/keys>.

---

## Usage

Basic:

```bash
python project.py path/to/meeting.mp3
```

The finished document appears in `output/meeting_MoM.docx`.

### Options

| Option | Meaning | Example |
|--------|---------|---------|
| `--output` | Choose where to save the `.docx`. | `--output output/board.docx` |
| `--model` | Whisper size: `tiny` `base` `small` `medium` `large-v3` `large-v3-turbo`. Bigger = more accurate but slower. | `--model small` |
| `--language` | Force a language (else auto-detected). | `--language en` |
| `--speakers` | Speaker labels: `pause` (default), `voice` (real recognition), or `none`. | `--speakers voice` |
| `--num-speakers` | (voice mode) How many people are talking. Omit to auto-detect. | `--num-speakers 3` |
| `--save-transcript` | Also save the raw transcript as `.txt`. | `--save-transcript` |

Example with several options:

```bash
python project.py meeting.m4a --model small --language en --save-transcript
```

#### About `large-v3-turbo`

`large-v3-turbo` is a **distilled** version of `large-v3`: it keeps **similar
transcription accuracy** but runs **roughly 8× faster**. It's the best choice
when `large-v3` feels too slow on a CPU but you still want high accuracy on
technical terms or accented speech.

```bash
python project.py meeting.mp3 --model large-v3-turbo
```

(The first run downloads the turbo model once and caches it under `models/`,
just like the other sizes.)

---

## Supported audio formats

`.mp3`, `.wav`, `.m4a`

---

## Project structure

```
Experiment/
├── project.py              # main entry point — run this
├── mom_generator/
│   ├── __init__.py
│   ├── transcriber.py      # Step 1: faster-whisper + basic (pause) speaker labels
│   ├── diarizer.py         # Step 2 (optional): real voice recognition (Resemblyzer)
│   ├── model_downloader.py # robustly fetches the Whisper model (mirror + retries)
│   ├── mom_builder.py      # Step 3: Groq LLM -> structured minutes
│   └── docx_exporter.py    # Step 4: write the .docx
├── .env.example            # template for your API key
├── requirements.txt        # core dependencies
├── requirements-voice.txt  # optional extras for --speakers voice
├── README.md
└── output/                 # generated documents (created automatically)
```

---

## Troubleshooting

- **`GROQ_API_KEY is not set`** — You haven't created `.env` or pasted your key.
- **First run is slow** — faster-whisper downloads the model once (a few hundred
  MB) and caches it. Later runs are much faster.
- **Out of memory / very slow on CPU** — use a smaller model: `--model tiny`.
- **Poor transcription quality** — use a larger model: `--model small` or
  `--model medium`, and make sure the audio is reasonably clear.

---

## Enabling real voice recognition

By default speakers are guessed from pauses. To tell voices apart for real — so
the same person keeps the same label across the whole meeting — install the
optional extras once:

```bash
pip install -r requirements-voice.txt
pip install resemblyzer --no-deps
```

Then add `--speakers voice`:

```bash
python project.py meeting.mp3 --speakers voice
# or, if you already know the head count:
python project.py meeting.mp3 --speakers voice --num-speakers 3
```

Under the hood this uses [Resemblyzer](https://github.com/resemble-ai/Resemblyzer):
it turns ~1.6s slices of audio into "voiceprints", clusters them so each cluster
is one person, then matches those to the transcript by timestamp. The whole
thing lives in `mom_generator/diarizer.py` — it's written to be read. It's free
and runs locally (no API key, no Hugging Face token), just heavier because it
uses PyTorch. For an even more accurate option, swap in
[`pyannote.audio`](https://github.com/pyannote/pyannote-audio).
