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
| 2. Speaker labels | A quick guess from pauses, **real voice recognition**, or **high-accuracy diarization**. | (heuristic) / `resemblyzer` / `pyannote.audio` |
| 3. Structure | The transcript is sent to an LLM that extracts title, attendees, decisions, action items, etc. | `groq` (`llama-3.3-70b-versatile`) |
| 4. Export | Everything is written into a formatted Word file. | `python-docx` |

> **Three ways to label speakers** (each is *added*, not a replacement — pick per run):
> - `--speakers pause` (default): a rough guess from silence gaps. Fast, but it
>   only detects *turn changes* — it can't tell if two turns are the same person.
> - `--speakers voice`: **real voice recognition** (Resemblyzer). Listens to the
>   actual voices, so the same person keeps the same label. Needs the voice extras.
> - `--speakers pyannote`: **high-accuracy diarization** (pyannote.audio) — the
>   most accurate at counting speakers and splitting fast exchanges. Needs the
>   pyannote extras **and** a free Hugging Face token.
>
> See [Speaker recognition options](#speaker-recognition-options).

---

## What runs locally vs. in the cloud

Almost everything runs **on your machine**. The only time your meeting content
leaves the computer is the LLM step:

| Stage | Where it runs | Does your meeting data leave? |
|-------|---------------|-------------------------------|
| Transcription (`faster-whisper`) | **Local** (CPU or GPU) | No — audio never leaves |
| Speaker labels (`pause` / `voice` / `pyannote`) | **Local** inference | No — audio never leaves |
| Structuring the minutes (`groq`) | **Cloud (Groq API)** | **Yes — the transcript text is sent** |
| Word export (`python-docx`) | **Local** | No |

So your **audio is processed entirely locally**; only the **transcript text** is
sent to the Groq LLM to be structured into minutes.

> **One caveat — model downloads.** The first time you use a model (Whisper,
> Resemblyzer, or pyannote) its *weights* are downloaded from the internet and
> cached locally; after that, inference is offline. That's downloading model
> files, not uploading your data. (pyannote's model is gated, so its first
> download also needs a Hugging Face token.)

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
| `--speakers` | Speaker labels: `pause` (default), `voice` (Resemblyzer), `pyannote` (most accurate), or `none`. | `--speakers pyannote` |
| `--num-speakers` | (`voice`/`pyannote`) How many people are talking. Omit to auto-detect. | `--num-speakers 4` |
| `--vocab` | Domain terms/names to bias transcription toward (fixes rare-word mishears). | `--vocab "lance, Furkan"` |
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

#### Running on a GPU

Transcription automatically uses an **NVIDIA GPU** if one is visible (e.g. a
free Google Colab T4) and falls back to CPU otherwise — nothing to configure.
A GPU is dramatically faster for the larger models, and `pyannote` diarization
uses the GPU too. See `colab_mom.ipynb` for a ready-made GPU setup.

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
│   ├── transcriber.py       # Step 1: faster-whisper (auto GPU) + pause speaker labels
│   ├── diarizer.py          # Step 2 (optional): voice recognition (Resemblyzer)
│   ├── diarizer_pyannote.py # Step 2 (optional): high-accuracy diarization (pyannote)
│   ├── model_downloader.py  # robustly fetches the Whisper model (mirror + retries)
│   ├── mom_builder.py       # Step 3: Groq LLM -> structured minutes
│   └── docx_exporter.py     # Step 4: write the .docx
├── app.py                   # Streamlit web UI (run: streamlit run app.py)
├── .env.example             # template for your API key (and optional HF token)
├── requirements.txt         # core dependencies
├── requirements-voice.txt   # optional extras for --speakers voice
├── requirements-pyannote.txt # optional extras for --speakers pyannote
├── colab_mom.ipynb          # run the pipeline on a free Colab T4 GPU
├── build_zip.py             # package the code for Colab upload
├── README.md
└── output/                  # generated documents (created automatically)
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

## Speaker recognition options

By default speakers are guessed from pauses (`--speakers pause`). Two more
accurate options listen to the actual voices so the same person keeps the same
label across the meeting. Both are **optional add-ons** — installing them does
not change the default behaviour, and pyannote does **not** replace Resemblyzer.

### `--speakers voice` — Resemblyzer (free, no token)

Install the voice extras once:

```bash
pip install -r requirements-voice.txt
pip install resemblyzer --no-deps
```

Then:

```bash
python project.py meeting.mp3 --speakers voice
python project.py meeting.mp3 --speakers voice --num-speakers 3   # if you know the count
```

It turns ~1.6s slices into "voiceprints", clusters them into speakers, and
matches them to the transcript by timestamp (`mom_generator/diarizer.py`). Free,
runs locally, no Hugging Face token — just heavier (uses PyTorch).

### `--speakers pyannote` — pyannote.audio (most accurate, needs a token)

[pyannote.audio](https://github.com/pyannote/pyannote-audio) is a dedicated
diarization model — the best at counting speakers and splitting fast exchanges.
Its model is **gated**, so there is a one-time setup:

```bash
pip install -r requirements-pyannote.txt
```

1. Accept the terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>
2. Create a token with **"Read access to contents of all public gated repos you
   can access"** at <https://huggingface.co/settings/tokens>
3. Add it to your `.env`:  `HF_TOKEN=hf_...`

Then:

```bash
python project.py meeting.mp3 --speakers pyannote
python project.py meeting.mp3 --speakers pyannote --num-speakers 4
```

Inference is local (only the model download needs the internet/token). Slow on
CPU, fast on GPU. It lives in `mom_generator/diarizer_pyannote.py` and reuses the
same speaker-assignment as the voice option, so the output format is identical.
