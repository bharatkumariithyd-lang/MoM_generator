"""
app.py  —  Streamlit web UI for the Minutes of Meeting (MoM) Generator
=====================================================================

A friendly, browser-based front end for the same pipeline that project.py
runs on the command line. Nothing about the core logic changes — this file
just *calls* the existing functions in the mom_generator package and wraps
them in a web page with an upload box, option controls, a progress display,
and a download button.

Run it with:

    streamlit run app.py

Streamlit then opens a local web page (usually http://localhost:8501).

WHY STREAMLIT?
    Streamlit turns a plain Python script into a web app. There is no HTML,
    CSS, or JavaScript to write: every `st.something(...)` call draws a widget.
    The whole script re-runs top to bottom each time the user clicks something,
    and Streamlit keeps the page in sync. That makes it the least-code way to
    put a UI on top of code you already have.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# The exact same pipeline pieces the command-line tool uses.
from mom_generator import (
    transcribe_audio,
    format_transcript,
    build_mom,
    export_to_docx,
)
from mom_generator.transcriber import (
    assign_basic_speakers,
    speaker_samples,
    rename_speakers,
)


# Load GROQ_API_KEY from a .env file (if present) into the environment, just
# like project.py does. The user can still paste a key in the sidebar below.
load_dotenv()

SUPPORTED_EXTENSIONS = ["mp3", "wav", "m4a"]


# ---------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Minutes of Meeting Generator",
    page_icon="📝",
    layout="centered",
)

st.title("📝 Minutes of Meeting Generator")
st.write(
    "Upload a meeting recording and get a polished Word document of the "
    "minutes — transcribed locally, then structured by an LLM."
)


# ---------------------------------------------------------------------------
# Sidebar: the options (these mirror the command-line flags in project.py)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")

    # Whisper model size: bigger = more accurate but slower.
    model_size = st.selectbox(
        "Transcription model",
        options=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        index=1,  # default to "base"
        help=(
            "Bigger models are more accurate but slower. 'base' is a good start. "
            "'large-v3-turbo' is a distilled large-v3: similar accuracy, much "
            "faster — handy when large-v3 feels too slow on CPU."
        ),
    )

    # How to label who-said-what.
    speaker_choice = st.selectbox(
        "Speaker labels",
        options=[
            "Rough guess from pauses (fast)",
            "Real voice recognition (slower, needs extras)",
            "Accurate diarization with pyannote (best, needs HF token)",
            "No speaker labels",
        ],
        index=0,
        help=(
            "Voice recognition needs the resemblyzer/torch extras "
            "(requirements-voice.txt). pyannote is the most accurate option but "
            "needs its extras AND a Hugging Face token (requirements-pyannote.txt)."
        ),
    )
    # Map the friendly label back to the internal mode the pipeline expects.
    speaker_mode = {
        "Rough guess from pauses (fast)": "pause",
        "Real voice recognition (slower, needs extras)": "voice",
        "Accurate diarization with pyannote (best, needs HF token)": "pyannote",
        "No speaker labels": "none",
    }[speaker_choice]

    # Relevant for both voice and pyannote modes: how many people are talking.
    num_speakers = None
    if speaker_mode in ("voice", "pyannote"):
        num_speakers = st.number_input(
            "Number of speakers (0 = auto-detect)",
            min_value=0,
            max_value=20,
            value=0,
            help="Leave at 0 to let the system guess how many people spoke.",
        )
        num_speakers = num_speakers or None  # turn 0 into None (auto-detect)

    # pyannote's models are gated, so it needs a Hugging Face token. Prefer the
    # .env value; otherwise let the user paste one here (same pattern as the
    # Groq key below). We stash it in the environment so the diarizer can read it.
    if speaker_mode == "pyannote":
        hf_env = os.getenv("HF_TOKEN")
        if hf_env:
            st.success("Hugging Face token loaded from .env ✓")
        else:
            hf_token = st.text_input(
                "Hugging Face token",
                type="password",
                help=(
                    "Accept the terms for pyannote/speaker-diarization-community-1, "
                    "then create a token with public-gated-repo read access at "
                    "huggingface.co/settings/tokens."
                ),
            ).strip()
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

    # Optional language hint. Empty = auto-detect.
    language = st.text_input(
        "Language code (optional)",
        value="",
        help="Like 'en' for English. Leave blank to auto-detect.",
    ).strip() or None

    st.divider()

    # API key: prefer the .env value; otherwise let the user paste one here.
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        st.success("Groq API key loaded from .env ✓")
        api_key = env_key
    else:
        st.warning("No GROQ_API_KEY found in .env")
        api_key = st.text_input(
            "Groq API key",
            type="password",
            help="Get a free key at https://console.groq.com/keys",
        ).strip()


# ---------------------------------------------------------------------------
# Main area: upload + generate
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a meeting audio file",
    type=SUPPORTED_EXTENSIONS,
    help="Supported: " + ", ".join(SUPPORTED_EXTENSIONS),
)

# Let the user listen back to confirm they picked the right file.
if uploaded_file is not None:
    st.audio(uploaded_file)

def _transcribe_and_detect(audio_path: str):
    """
    STEP 1 (the slow part): transcribe the audio and, if asked, detect speakers.

    Returns (segments, use_speakers), or None if no speech was found. We run this
    on its own and cache the result so the speaker-naming + LLM step can run
    afterwards WITHOUT re-transcribing.
    """
    with st.status("Transcribing & detecting speakers...", expanded=True) as status:
        status.update(label="Transcribing audio (this can take a while)...")
        segments = transcribe_audio(audio_path, model_size=model_size, language=language)
        if not segments:
            status.update(label="No speech detected.", state="error")
            return None
        st.write(f"✓ Transcribed {len(segments)} segments.")

        if speaker_mode == "voice":
            status.update(label="Recognizing speakers by voice...")
            # Imported here so the page still loads without the heavy extras.
            from mom_generator.diarizer import diarize_audio, assign_voice_speakers
            turns = diarize_audio(audio_path, num_speakers=num_speakers)
            segments = assign_voice_speakers(segments, turns)
            use_speakers = True
            st.write("✓ Added speaker labels from real voice recognition.")
        elif speaker_mode == "pyannote":
            status.update(label="Diarizing speakers with pyannote...")
            from mom_generator.diarizer import assign_voice_speakers
            from mom_generator.diarizer_pyannote import diarize_audio_pyannote
            turns = diarize_audio_pyannote(audio_path, num_speakers=num_speakers)
            segments = assign_voice_speakers(segments, turns)
            use_speakers = True
            st.write("✓ Added speaker labels from pyannote diarization.")
        elif speaker_mode == "pause":
            status.update(label="Labeling speakers (rough guess)...")
            segments = assign_basic_speakers(segments)
            use_speakers = True
            st.write("✓ Added approximate speaker labels.")
        else:  # "none"
            use_speakers = False
            st.write("• Skipping speaker labels.")

        status.update(label="Done — name the speakers (optional), then generate.",
                      state="complete")
    return segments, use_speakers


def _build_minutes(segments, use_speakers, name_map, output_path):
    """
    STEP 2: apply the speaker name_map, structure the minutes with the LLM, and
    write the .docx. Works on a COPY of the cached segments, so re-generating with
    different names always starts from the original "Speaker N" labels.
    """
    import copy
    segs = copy.deepcopy(segments)
    if use_speakers and name_map:
        rename_speakers(segs, name_map)
    transcript = format_transcript(segs, with_speakers=use_speakers)

    with st.status("Structuring the minutes...", expanded=True) as status:
        status.update(label="Structuring minutes with the LLM...")
        mom = build_mom(transcript, api_key=api_key)
        st.write("✓ Minutes structured.")
        status.update(label="Writing the Word document...")
        export_to_docx(mom, output_path)
        st.write("✓ Word document created.")
        status.update(label="Done! 🎉", state="complete")
    return mom


def show_preview(mom: dict):
    """Show a quick on-page preview of the minutes so the user sees the result."""
    st.subheader(mom.get("meeting_title", "Meeting Minutes"))

    if mom.get("date_time"):
        st.caption(f"🗓️ {mom['date_time']}")
    if mom.get("attendees"):
        st.write("**Attendees:** " + ", ".join(mom["attendees"]))

    def bullets(title, items):
        if items:
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {item}")

    bullets("Agenda", mom.get("agenda_items"))
    bullets("Key Discussion Points", mom.get("key_discussion_points"))
    bullets("Decisions Made", mom.get("decisions_made"))

    action_items = mom.get("action_items") or []
    if action_items:
        st.markdown("**Action Items**")
        st.table(
            [
                {
                    "Task": a.get("task", ""),
                    "Owner": a.get("owner", "") or "(unassigned)",
                    "Deadline": a.get("deadline", "") or "(no deadline)",
                }
                for a in action_items
            ]
        )

    if mom.get("next_meeting"):
        st.markdown(f"**Next Meeting:** {mom['next_meeting']}")


# ---------------------------------------------------------------------------
# Two-step flow:
#   1) Transcribe + detect speakers (the slow part; cached in session_state)
#   2) Name the speakers (optional), then generate the minutes
# Splitting it lets you label "Speaker 1/2" with real names BEFORE the minutes
# are written, and without paying for transcription twice.
# ---------------------------------------------------------------------------
if st.button(
    "1 · Transcribe & detect speakers",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
):
    # Write the upload to a temp file (the pipeline works on paths), transcribe +
    # diarize, cache the result, then delete the temp file — step 2 needs only
    # the segments, not the audio.
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_audio_path = tmp.name
    try:
        result = _transcribe_and_detect(temp_audio_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        st.error(f"Something went wrong while transcribing: {exc}")
        result = None
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if result is None:
        st.session_state.pop("detected", None)
        st.error("No speech was detected in the audio. Nothing to do.")
    else:
        segments, use_speakers = result
        st.session_state["detected"] = {
            "segments": segments,
            "use_speakers": use_speakers,
            "samples": speaker_samples(segments) if use_speakers else {},
            "base_name": os.path.splitext(uploaded_file.name)[0],
        }

# Once we have a cached detection, offer optional speaker naming + the generate
# button. This block re-renders on every interaction (e.g. typing a name).
detected = st.session_state.get("detected")
if detected:
    name_map = {}
    if detected["use_speakers"] and detected["samples"]:
        st.divider()
        st.subheader("Name the speakers (optional)")
        st.caption(
            "Diarization tells the voices apart but not who they are. Add real "
            "names to use them in the minutes; leave blank to keep the labels."
        )
        for label, sample in detected["samples"].items():
            value = st.text_input(
                f'{label} — said: "{sample}"', key=f"name_{label}"
            ).strip()
            if value:
                name_map[label] = value

    if st.button("2 · Generate minutes", type="primary", use_container_width=True):
        if not api_key:
            st.error(
                "No Groq API key. Paste one in the sidebar, or add GROQ_API_KEY "
                "to a .env file. Get a free key at https://console.groq.com/keys"
            )
            st.stop()
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"{detected['base_name']}_MoM.docx")
        try:
            mom = _build_minutes(
                detected["segments"], detected["use_speakers"], name_map, output_path
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Something went wrong: {exc}")
            mom = None

        if mom:
            st.success(f"Minutes saved to: {output_path}")
            with open(output_path, "rb") as f:
                docx_bytes = f.read()
            st.download_button(
                label="⬇️ Download minutes (.docx)",
                data=docx_bytes,
                file_name=f"{detected['base_name']}_MoM.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
            st.divider()
            st.caption("Preview")
            show_preview(mom)
