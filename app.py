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
from mom_generator.transcriber import assign_basic_speakers


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
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=1,  # default to "base"
        help="Bigger models are more accurate but slower. 'base' is a good start.",
    )

    # How to label who-said-what.
    speaker_choice = st.selectbox(
        "Speaker labels",
        options=[
            "Rough guess from pauses (fast)",
            "Real voice recognition (slower, needs extras)",
            "No speaker labels",
        ],
        index=0,
        help=(
            "Voice recognition needs the optional resemblyzer/torch extras "
            "(see requirements-voice.txt)."
        ),
    )
    # Map the friendly label back to the internal mode the pipeline expects.
    speaker_mode = {
        "Rough guess from pauses (fast)": "pause",
        "Real voice recognition (slower, needs extras)": "voice",
        "No speaker labels": "none",
    }[speaker_choice]

    # Only relevant for voice mode: how many people are talking.
    num_speakers = None
    if speaker_mode == "voice":
        num_speakers = st.number_input(
            "Number of speakers (0 = auto-detect)",
            min_value=0,
            max_value=20,
            value=0,
            help="Leave at 0 to let the system guess how many people spoke.",
        )
        num_speakers = num_speakers or None  # turn 0 into None (auto-detect)

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

generate_clicked = st.button(
    "✨ Generate Minutes",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
)


def run_pipeline(audio_path: str, output_path: str):
    """
    Run the four pipeline steps and report progress in the UI.

    This is the web equivalent of main() in project.py. It returns the
    structured `mom` dict so we can preview it on the page afterwards.
    """
    # st.status gives a collapsible progress box with a spinner. We update its
    # label as each step finishes so the user can see what's happening — handy
    # because transcription can take a while and a frozen page looks broken.
    with st.status("Working on your minutes...", expanded=True) as status:

        # --- Step 1: Transcribe ------------------------------------------
        status.update(label="Step 1/4 · Transcribing audio (this can take a while)...")
        segments = transcribe_audio(audio_path, model_size=model_size, language=language)
        if not segments:
            status.update(label="No speech detected.", state="error")
            st.error("No speech was detected in the audio. Nothing to do.")
            return None
        st.write(f"✓ Transcribed {len(segments)} segments.")

        # --- Step 2: Speaker labels (optional) ---------------------------
        if speaker_mode == "voice":
            status.update(label="Step 2/4 · Recognizing speakers by voice...")
            # Imported here so the page still loads without the heavy extras.
            from mom_generator.diarizer import diarize_audio, assign_voice_speakers
            turns = diarize_audio(audio_path, num_speakers=num_speakers)
            segments = assign_voice_speakers(segments, turns)
            use_speakers = True
            st.write("✓ Added speaker labels from real voice recognition.")
        elif speaker_mode == "pause":
            status.update(label="Step 2/4 · Labeling speakers (rough guess)...")
            segments = assign_basic_speakers(segments)
            use_speakers = True
            st.write("✓ Added approximate speaker labels.")
        else:  # "none"
            use_speakers = False
            st.write("• Skipping speaker labels.")

        transcript = format_transcript(segments, with_speakers=use_speakers)

        # --- Step 3: Build structured minutes with Groq ------------------
        status.update(label="Step 3/4 · Structuring minutes with the LLM...")
        mom = build_mom(transcript, api_key=api_key)
        st.write("✓ Minutes structured.")

        # --- Step 4: Write the .docx -------------------------------------
        status.update(label="Step 4/4 · Writing the Word document...")
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
# When the button is clicked: validate, run, then offer a download.
# ---------------------------------------------------------------------------
if generate_clicked:
    if not api_key:
        st.error(
            "No Groq API key. Paste one in the sidebar, or add GROQ_API_KEY to "
            "a .env file. Get a free key at https://console.groq.com/keys"
        )
        st.stop()

    # Streamlit hands us the upload as in-memory bytes, but the pipeline works
    # on file *paths*. So we write the upload to a temporary file on disk and
    # point the pipeline at that. The temp file is cleaned up at the end.
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_audio_path = tmp.name

    base_name = os.path.splitext(uploaded_file.name)[0]
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"{base_name}_MoM.docx")

    try:
        mom = run_pipeline(temp_audio_path, output_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        st.error(f"Something went wrong: {exc}")
        mom = None
    finally:
        # Clean up the temporary audio file no matter what happened.
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if mom:
        st.success(f"Minutes saved to: {output_path}")

        # Read the finished .docx back as bytes so the browser can download it.
        with open(output_path, "rb") as f:
            docx_bytes = f.read()
        st.download_button(
            label="⬇️ Download minutes (.docx)",
            data=docx_bytes,
            file_name=f"{base_name}_MoM.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

        st.divider()
        st.caption("Preview")
        show_preview(mom)
