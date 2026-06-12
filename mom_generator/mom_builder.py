"""
mom_builder.py
==============

Step 3 of the pipeline: send the transcript to the Groq LLM and get back a
clean, structured Minutes of Meeting.

We ask the model (llama-3.3-70b-versatile) to return its answer as strict JSON
so the next step (writing the .docx) can rely on a predictable shape instead of
parsing free-form text.

LONG MEETINGS (automatic chunking)
----------------------------------
A very long transcript can be too big for one request (free-tier token caps)
and, even when it fits, models tend to "lose the middle" and quietly drop
details. So build_mom() picks a strategy based on length:

    short transcript -> one LLM call
    long  transcript -> map-reduce:
        1. split into overlapping chunks            (_split_transcript)
        2. MAP:    extract a partial MoM per chunk   (_extract_from_text)
        3. REDUCE: merge the partials and dedupe     (_merge_partials)
        4. one final LLM cleanup pass                (_consolidate_with_llm)

The public function build_mom() keeps the same signature either way, so nothing
downstream (project.py, docx_exporter) needs to know which path ran.
"""

import json
import os
import time


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


# --- Chunking knobs --------------------------------------------------------
# We estimate tokens cheaply as characters / 4 (good enough, and avoids needing
# a real tokenizer). These thresholds decide single-call vs. map-reduce.
CHARS_PER_TOKEN = 4
SINGLE_CALL_TOKEN_LIMIT = 10000   # transcripts under this go in one request
CHUNK_TARGET_TOKENS = 6000        # aim for chunks roughly this big
CHUNK_OVERLAP_LINES = 2           # carry this many lines into the next chunk


def _estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer: ~4 characters per token."""
    return len(text) // CHARS_PER_TOKEN


def _build_prompt(transcript: str, part_info: tuple | None = None) -> str:
    """
    Assemble the instruction text we send to the model.

    If `part_info` is (i, n), we tell the model it is only seeing PART i of n of
    a longer meeting, so it extracts what's present in this slice and does not
    invent an overall title or a "next meeting" for every chunk.
    """
    if part_info:
        i, n = part_info
        part_note = (
            f"NOTE: This is PART {i} of {n} of a longer meeting transcript. "
            f"Extract only what actually appears in this part. If the meeting "
            f"title, date, or next-meeting details are not stated in THIS part, "
            f"leave them as empty strings — a later step combines the parts.\n\n"
        )
    else:
        part_note = ""

    return f"""You are an expert executive assistant who writes precise,
professional Minutes of Meeting from raw meeting transcripts.

The transcript below was produced by automatic speech-to-text, so it may
contain small errors. Speaker labels (e.g. "Speaker 1") are approximate guesses
based on pauses, NOT verified identities — use them only as a loose hint and
prefer real names if people introduce themselves in the text.

{part_note}Rules:
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

    Automatically uses a single call for short transcripts and a map-reduce
    strategy for long ones (see the module docstring). The return shape is the
    same either way: a dict matching MOM_SCHEMA_DESCRIPTION.

    Parameters
    ----------
    transcript : str
        The formatted transcript text from transcriber.format_transcript().
    api_key : str | None
        Your Groq API key. If None, read from the GROQ_API_KEY environment
        variable (which python-dotenv loads from your .env file).
    model : str
        The Groq model to use. Project standard: "llama-3.3-70b-versatile".
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

    if _estimate_tokens(transcript) <= SINGLE_CALL_TOKEN_LIMIT:
        # Short meeting: the whole transcript fits comfortably in one request.
        print(f"[mom_builder] Sending transcript to Groq model '{model}'...")
        mom = _extract_from_text(client, model, transcript)
        print("[mom_builder] Received structured response from Groq.")
    else:
        # Long meeting: split, extract per chunk, then merge (prints its own progress).
        mom = _build_mom_chunked(client, model, transcript)

    return _ensure_all_keys(mom)


def _groq_chat(client, model, messages, max_retries: int = 5) -> str:
    """
    Make one Groq chat call (JSON mode) and return the raw reply text.

    Retries with exponential backoff on ANY error — most importantly the 429
    rate-limit that free tiers return when we send several chunks quickly. This
    is what makes the long-meeting path survive on a free plan.
    """
    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # Ask Groq to guarantee a JSON object (no ```code fences```).
                response_format={"type": "json_object"},
                temperature=0.2,  # low temperature -> factual, less "creative"
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything
            if attempt == max_retries:
                raise
            print(f"[mom_builder]   API error ({type(exc).__name__}); "
                  f"retry {attempt}/{max_retries} in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2  # 4s, 8s, 16s, 32s ...


def _extract_from_text(client, model, transcript, part_info=None) -> dict:
    """
    One LLM call: send `transcript` (a whole meeting OR a single chunk) and parse
    the JSON reply into a dict. Shared by the single-call and map-reduce paths.
    """
    messages = [
        {"role": "system",
         "content": "You output only valid JSON. No markdown, no extra text."},
        {"role": "user", "content": _build_prompt(transcript, part_info)},
    ]
    raw_json = _groq_chat(client, model, messages)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Groq returned text that was not valid JSON: {exc}\n"
            f"Raw response:\n{raw_json}"
        )


# ---------------------------------------------------------------------------
# Long-meeting path: split -> map -> reduce
# ---------------------------------------------------------------------------
def _build_mom_chunked(client, model, transcript) -> dict:
    """Map-reduce a long transcript into one MoM (see the module docstring)."""
    chunks = _split_transcript(transcript)
    print(f"[mom_builder] Long meeting (~{_estimate_tokens(transcript)} tokens) "
          f"-> processing in {len(chunks)} chunks...")

    # MAP: extract a partial MoM from each chunk.
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[mom_builder]   Extracting chunk {i}/{len(chunks)}...")
        try:
            partials.append(
                _extract_from_text(client, model, chunk, part_info=(i, len(chunks)))
            )
        except Exception as exc:  # noqa: BLE001
            # One bad chunk shouldn't sink the whole meeting — skip and continue.
            print(f"[mom_builder]   (chunk {i} failed, skipping: {exc})")
        time.sleep(0.5)  # be gentle with free-tier rate limits

    if not partials:
        raise RuntimeError("Every chunk failed to process; cannot build minutes.")

    # REDUCE: merge in code, then one LLM pass to consolidate and tidy.
    merged = _merge_partials(partials)
    print("[mom_builder]   Merged chunks; running final consolidation pass...")
    final = _consolidate_with_llm(client, model, merged)
    print("[mom_builder] Received structured response from Groq.")
    return final


def _split_transcript(transcript, max_tokens=CHUNK_TARGET_TOKENS,
                      overlap_lines=CHUNK_OVERLAP_LINES) -> list:
    """
    Cut the transcript into chunks of roughly `max_tokens`, breaking only on
    whole lines (never mid-sentence) and carrying `overlap_lines` lines into the
    next chunk so context that straddles a boundary isn't lost.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    lines = [ln for ln in transcript.split("\n") if ln.strip()]

    chunks, current, current_chars = [], [], 0
    for line in lines:
        current.append(line)
        current_chars += len(line) + 1
        if current_chars >= max_chars:
            chunks.append("\n".join(current))
            # Start the next chunk with a little overlap for continuity.
            current = current[-overlap_lines:] if overlap_lines else []
            current_chars = sum(len(l) + 1 for l in current)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _merge_partials(partials) -> dict:
    """
    Combine the per-chunk MoMs into one (the REDUCE step).

    Lists are concatenated then de-duplicated (case/space-insensitive). For the
    scalar fields we keep the first non-empty value any chunk reported. We start
    from EMPTY defaults (not _ensure_all_keys, whose title default would block
    the "first non-empty" logic).
    """
    list_fields = ["attendees", "agenda_items", "key_discussion_points",
                   "decisions_made"]
    merged = {
        "meeting_title": "", "date_time": "",
        "attendees": [], "agenda_items": [], "key_discussion_points": [],
        "decisions_made": [], "action_items": [], "next_meeting": "",
    }

    for partial in partials:
        for field in list_fields:
            values = partial.get(field) or []
            if isinstance(values, list):
                merged[field].extend(values)
        action_items = partial.get("action_items") or []
        if isinstance(action_items, list):
            merged["action_items"].extend(action_items)
        for field in ["meeting_title", "date_time", "next_meeting"]:
            value = partial.get(field) or ""
            if not merged[field] and value:
                merged[field] = value

    for field in list_fields:
        merged[field] = _dedupe_strings(merged[field])
    merged["action_items"] = _dedupe_action_items(merged["action_items"])
    return merged


def _normalize(text) -> str:
    """Lowercase + collapse whitespace, for comparing 'the same' text."""
    return " ".join(str(text).lower().split())


def _dedupe_strings(items) -> list:
    """Drop duplicate strings, keeping first occurrence and its original wording."""
    seen, result = set(), []
    for item in items:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedupe_action_items(items) -> list:
    """
    Drop duplicate action items (matched by task text). When two chunks report
    the same task, keep whichever copy actually has an owner / deadline.
    """
    by_task, order = {}, []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _normalize(item.get("task", ""))
        if not key:
            continue
        if key not in by_task:
            by_task[key] = dict(item)
            order.append(key)
        else:
            existing = by_task[key]
            for field in ["owner", "deadline"]:
                if not existing.get(field) and item.get(field):
                    existing[field] = item[field]
    return [by_task[key] for key in order]


def _consolidate_with_llm(client, model, merged) -> dict:
    """
    Final cleanup pass: hand the merged notes back to the model to tidy wording,
    merge near-duplicates the string match missed, and choose one good title.

    Falls back to the code-merged result if this call fails — so the extra
    polish can never make the output worse than the deterministic merge.
    """
    instruction = (
        "You are given draft Minutes of Meeting assembled from several parts of "
        "ONE meeting. Consolidate them: merge true duplicates, fix wording, and "
        "write one clear meeting_title. Do NOT drop any distinct decision or "
        "action item, and do NOT invent new ones. Return the SAME JSON schema.\n\n"
        f"{MOM_SCHEMA_DESCRIPTION}\n\n"
        f"DRAFT MINUTES (JSON):\n{json.dumps(merged, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system",
         "content": "You output only valid JSON. No markdown, no extra text."},
        {"role": "user", "content": instruction},
    ]
    try:
        return json.loads(_groq_chat(client, model, messages))
    except Exception as exc:  # noqa: BLE001
        print(f"[mom_builder]   (consolidation pass skipped: {exc})")
        return merged


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
    # A model can return an explicit empty title; fall back to the placeholder.
    if not mom.get("meeting_title"):
        mom["meeting_title"] = defaults["meeting_title"]
    return mom
