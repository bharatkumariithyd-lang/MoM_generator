"""
mom_builder.py
==============

Step 2 of the pipeline: send the transcript to the Groq LLM and get back a
clean, structured Minutes of Meeting.

We ask the model (llama-3.3-70b-versatile) to return its answer as strict JSON
so the next step (writing the .docx) can rely on a predictable shape instead of
parsing free-form text.
"""

import json
import os


# This is the exact shape of JSON we ask the model to return. Keeping it here
# (and describing it in the prompt) means docx_exporter.py knows what to expect.
MOM_SCHEMA_DESCRIPTION = """
Return ONLY a valid JSON object with EXACTLY these keys:

{
  "meeting_title": "string - a short inferred title for the meeting",
  "date_time": "string - the date/time if clearly mentioned, else empty string",
  "attendees": ["list of attendee names or roles mentioned, else empty list"],
  "agenda_items": ["list of agenda / topics that were meant to be covered"],
  "key_discussion_points": ["list of the main points discussed"],
  "decisions_made": ["list of concrete decisions the group agreed on"],
  "action_items": [
    {
      "task": "string - what needs to be done",
      "owner": "string - who is responsible, or empty string if not said",
      "deadline": "string - due date if mentioned, or empty string"
    }
  ],
  "next_meeting": "string - details of the next meeting if mentioned, else empty string"
}
"""


def _build_prompt(transcript: str) -> str:
    """Assemble the instruction text we send to the model."""
    return f"""You are an expert executive assistant who writes precise,
professional Minutes of Meeting from raw meeting transcripts.

The transcript below was produced by automatic speech-to-text, so it may
contain small errors. Speaker labels (e.g. "Speaker 1") are approximate guesses
based on pauses, NOT verified identities — use them only as a loose hint and
prefer real names if people introduce themselves in the text.

Rules:
- Base everything ONLY on what is actually in the transcript. Do not invent
  attendees, dates, decisions, or action items.
- If something is not mentioned, use an empty string "" or an empty list [].
- Keep each point concise and written in clear, professional English.

{MOM_SCHEMA_DESCRIPTION}

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"
"""


def build_mom(
    transcript: str,
    api_key: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Send the transcript to Groq and return the Minutes of Meeting as a dict.

    Parameters
    ----------
    transcript : str
        The formatted transcript text from transcriber.format_transcript().
    api_key : str | None
        Your Groq API key. If None, we read it from the GROQ_API_KEY
        environment variable (which python-dotenv loads from your .env file).
    model : str
        The Groq model to use. The project standard is
        "llama-3.3-70b-versatile".

    Returns
    -------
    dict  (matching MOM_SCHEMA_DESCRIPTION)
    """
    # Imported here so importing this module does not require the SDK to exist.
    from groq import Groq

    # Resolve the API key: explicit argument first, then environment variable.
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "No Groq API key found. Create a .env file with a line like:\n"
            "    GROQ_API_KEY=your_key_here\n"
            "You can copy .env.example to .env to get started."
        )

    client = Groq(api_key=key)

    print(f"[mom_builder] Sending transcript to Groq model '{model}'...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You output only valid JSON. No markdown, no extra text.",
            },
            {"role": "user", "content": _build_prompt(transcript)},
        ],
        # Ask Groq to guarantee the reply is a JSON object. This avoids the
        # common headache of the model wrapping JSON in ```code fences```.
        response_format={"type": "json_object"},
        temperature=0.2,  # low temperature -> factual, less "creative"
    )

    raw_json = response.choices[0].message.content
    print("[mom_builder] Received structured response from Groq.")

    # Convert the JSON text into a Python dictionary.
    try:
        mom = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Groq returned text that was not valid JSON: {exc}\n"
            f"Raw response:\n{raw_json}"
        )

    return _ensure_all_keys(mom)


def _ensure_all_keys(mom: dict) -> dict:
    """
    Defensive cleanup: make sure every expected key exists so the docx writer
    never crashes on a missing field, even if the model omits one.
    """
    defaults = {
        "meeting_title": "Meeting Minutes",
        "date_time": "",
        "attendees": [],
        "agenda_items": [],
        "key_discussion_points": [],
        "decisions_made": [],
        "action_items": [],
        "next_meeting": "",
    }
    for key, default_value in defaults.items():
        mom.setdefault(key, default_value)
    return mom
