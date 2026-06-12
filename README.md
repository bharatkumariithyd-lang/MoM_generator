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
| 2. Speaker labels | A *basic* "Speaker 1 / Speaker 2" guess is added based on pauses. | (built-in heuristic) |
| 3. Structure | The transcript is sent to an LLM that extracts title, attendees, decisions, action items, etc. | `groq` (`llama-3.3-70b-versatile`) |
| 4. Export | Everything is written into a formatted Word file. | `python-docx` |

> **About speaker labels:** this is a rough guess based on silence between
> sentences — it detects *turn changes*, not real voices. For accurate
> "who said what", a dedicated diarization model (e.g. `pyannote.audio`) is
> needed. See [Upgrading](#upgrading-real-speaker-diarization).

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
| `--model` | Whisper size: `tiny` `base` `small` `medium` `large-v3`. Bigger = more accurate but slower. | `--model small` |
| `--language` | Force a language (else auto-detected). | `--language en` |
| `--no-diarization` | Skip the speaker-labeling step. | `--no-diarization` |
| `--save-transcript` | Also save the raw transcript as `.txt`. | `--save-transcript` |

Example with several options:

```bash
python project.py meeting.m4a --model small --language en --save-transcript
```

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
│   ├── transcriber.py      # Step 1+2: faster-whisper + basic speaker labels
│   ├── mom_builder.py      # Step 3: Groq LLM -> structured minutes
│   └── docx_exporter.py    # Step 4: write the .docx
├── .env.example            # template for your API key
├── requirements.txt
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

## Upgrading: real speaker diarization

The built-in speaker labeling is intentionally simple. To get accurate speaker
identification, replace `assign_basic_speakers()` in
`mom_generator/transcriber.py` with a [`pyannote.audio`](https://github.com/pyannote/pyannote-audio)
pipeline (requires PyTorch and a free Hugging Face token), then map its speaker
IDs onto the Whisper segments by timestamp.
