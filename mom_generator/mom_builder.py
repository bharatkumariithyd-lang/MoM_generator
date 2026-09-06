"""
mom_builder.py
==============

Step 3 of the pipeline: send the transcript to the Groq LLM and get back a
clean, structured Minutes of Meeting.

We ask the model (openai/gpt-oss-120b) to return its answer as strict JSON
so the next step (writing the .docx) can rely on a predictable shape instead of
parsing free-form text.

LONG MEETINGS (automatic chunking)
----------------------------------
A very long transcript can be too big for one request — not for the model's
context window (gpt-oss-120b has 131k) but for Groq's free-tier cap of 8,000
tokens per MINUTE. So build_mom() picks a strategy based on length, measuring
the budget at runtime rather than trusting hardcoded thresholds:

    short transcript -> one LLM call
    long  transcript -> map-reduce:
        1. split into overlapping chunks            (_split_transcript)
        2. MAP:    extract a partial MoM per chunk   (_extract_from_text)
        3. REDUCE: merge the partials and dedupe     (_merge_partials)
        4. one final LLM cleanup pass                (_consolidate_with_llm)

The public function build_mom() keeps the same signature either way, so nothing
downstream (project.py, docx_exporter) needs to know which path ran.

Requests are paced against a rolling 60-second token window, an oversized chunk
is split and retried rather than retried unchanged, and a chunk that is lost
anyway stops the run instead of silently yielding minutes for part of the
meeting (--allow-partial overrides that).
"""

import json
import os
import re
import time


# This is the exact shape of JSON we ask the model to return. Keeping it here
# (and describing it in the prompt) means docx_exporter.py knows what to expect.
MOM_SCHEMA_DESCRIPTION = """
Return ONLY a valid JSON object with EXACTLY these keys:

{
  "meeting_title": "string - a short inferred title for the meeting",
  "date_time": "string - the date/time if clearly mentioned, else empty string",
  "attendees": ["people who ACTUALLY PARTICIPATED (spoke, or are addressed as present). Do NOT include people who are only talked ABOUT - absent colleagues, third parties, names merely mentioned. Real names if stated, else roles. Strip honorifics: 'sir', 'madam' and a trailing '-ji' are forms of address, not names, and are never attendees on their own. Merge speech-to-text spelling variants of the same person into one spelling. Empty list if unclear."],
  "agenda_items": ["the meeting's main topics as short headings. If no formal agenda was announced, infer 2-5 high-level topic headings from the discussion (these summarise; key_discussion_points hold the detail)"],
  "key_discussion_points": ["EVERY distinct substantive point discussed - be thorough and specific; do not collapse several different points into one generic bullet"],
  "decisions_made": ["EVERY concrete decision or agreement the meeting actually CONCLUDED - list each separately, including conditional or partial agreements. Anything merely proposed, requested, or left unresolved belongs in key_discussion_points, NOT here"],
  "action_items": [
    {
      "task": "string - what needs to be done",
      "owner": "string - who is responsible, ONLY if the transcript clearly states it by name; otherwise empty string. Do NOT guess, and NEVER use a speaker label such as 'Speaker 1'.",
      "deadline": "string - due date if mentioned, or empty string"
    }
  ],
  "next_meeting": "string - details of the next meeting if mentioned, else empty string"
}
"""


# --- Token budget ----------------------------------------------------------
# Chunking exists for ONE reason: Groq's free tier caps us at a number of tokens
# per MINUTE (8,000). That is a billing limit, not a context limit — the model
# itself has a 131k window, so an entire 90-minute transcript would fit in a
# single request on a paid tier. Raise GROQ_TPM_LIMIT there and the code below
# stops chunking on its own.
TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))

# Leave headroom: the estimate below is approximate, and the model's own reply
# counts against the same per-minute budget.
REQUEST_TOKEN_BUDGET = int(TPM_LIMIT * 0.70)

CHUNK_OVERLAP_TURNS = 1    # speaker turns carried into the next chunk
MIN_CHUNK_TOKENS = 500     # never subdivide below this (stops runaway splitting)
MAX_SPLIT_DEPTH = 3        # how many times an oversized chunk may be halved


def _estimate_tokens(text: str) -> int:
    """
    Rough token count without a tokenizer.

    Measured against real Groq usage on this project's transcripts, English
    prose runs about 3.2 characters per token — not the 4 a generic rule of
    thumb suggests. That 22% shortfall is exactly how oversized requests used to
    slip past this check. Devanagari is far denser (roughly a token per
    character), so a Hinglish transcript has to be counted per script or it gets
    badly under-counted.
    """
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars // 3 + (len(text) - ascii_chars)


def _prompt_overhead_tokens() -> int:
    """
    What the instructions alone cost, before any transcript is added.

    Measured at runtime rather than hardcoded: editing the prompt silently grows
    this, and a stale hardcoded allowance is what pushes requests over the cap.
    """
    return _estimate_tokens(_build_prompt("", part_info=(1, 9)))


def _chunk_budget() -> int:
    """How many transcript tokens fit in one request alongside the prompt."""
    return max(MIN_CHUNK_TOKENS, REQUEST_TOKEN_BUDGET - _prompt_overhead_tokens())


# Sliding window of (timestamp, tokens) for requests sent in the last minute.
_recent_requests: list[tuple[float, int]] = []


def _pace_for_tpm(tokens: int) -> None:
    """
    Wait, if needed, so this request stays inside the per-MINUTE token budget.

    A fixed short sleep between chunks does nothing against a per-minute cap;
    what matters is how many tokens have gone out in the last 60 seconds.
    """
    now = time.monotonic()
    _recent_requests[:] = [(t, n) for t, n in _recent_requests if now - t < 60.0]
    spent = sum(n for _t, n in _recent_requests)

    if _recent_requests and spent + tokens > TPM_LIMIT:
        wait = 60.0 - (now - _recent_requests[0][0]) + 1.0
        if wait > 0:
            print(f"[mom_builder]   Pacing for the {TPM_LIMIT} tokens/min limit: "
                  f"waiting {wait:.0f}s ({spent} tokens used in the last minute)...")
            time.sleep(wait)
            now = time.monotonic()
            _recent_requests[:] = [(t, n) for t, n in _recent_requests
                                   if now - t < 60.0]

    _recent_requests.append((time.monotonic(), tokens))


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
contain small errors. Speaker labels (e.g. "Speaker 1") indicate distinct
speaking turns and are a hint to how many people took part, but they are NOT
verified identities. Where the text names someone or they introduce themselves,
use the real name.

This is an Indian workplace meeting, so the transcript may be "Hinglish" -
mostly English with Hindi words, phrases, or short exchanges mixed in. The
speech-to-text engine may render that Hindi EITHER phonetically in Latin
letters (e.g. "kal tak bhej dena", "thoda time lagega") OR in Devanagari
script. It sometimes even writes ENGLISH speech in Devanagari letters, so sound
out Devanagari before assuming it is Hindi. Do NOT skip, ignore, or treat any
of it as noise or as a transcription failure - work out what was meant and fold
any substantive content into the minutes exactly like the English portions,
rendered in clear, professional English.

{part_note}Rules:
- Base everything ONLY on what is actually in the transcript. Do not invent
  attendees, dates, decisions, or action items.
- ATTENDEES are only the people actually present/speaking — never list someone
  just because they are mentioned or discussed by others. "Sir", "madam" and a
  trailing "-ji" are forms of address, not part of anyone's name.
- NEVER output a bare speaker label ("Speaker 1", "Speaker 2") as an attendee or
  as an action-item owner. Those are turn markers, not identities. Use the names
  people are actually called in the transcript; if a speaker is never named,
  leave them out rather than listing their label.
- Speech-to-text garbles personal names and institution names, and may spell the
  SAME one differently in different places (two spellings a letter or two apart
  are almost always the same person or organisation). Treat obvious variants as
  one and use a single consistent spelling throughout. Where context makes the
  intended proper noun unambiguous, write the correct form - a dropped letter in
  an institute acronym, say. Never invent a name you cannot ground in the
  transcript; where one stays genuinely unclear, use the person's role.
- DECISIONS are only what the meeting actually CONCLUDED. Anything proposed,
  requested, or still open belongs in key_discussion_points instead. If the
  transcript is unclear or self-contradictory about what was decided, record
  that uncertainty rather than confidently picking one version.
- Be THOROUGH on discussion points and decisions: capture every distinct one
  rather than compressing the meeting into a few generic bullets.
- Do NOT guess action-item owners or deadlines; leave them empty unless the
  transcript clearly states them. "Someone should do X" with nobody named has
  an EMPTY owner.
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
    model: str = "openai/gpt-oss-120b",
    allow_partial: bool = False,
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
        The Groq model to use. Project standard: "openai/gpt-oss-120b".
    allow_partial : bool
        Long meetings only. By default, losing any chunk raises rather than
        quietly producing minutes that cover part of the meeting. Set True to
        accept whatever came back.
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

    # One call whenever the transcript plus its instructions fit in the budget.
    # Chunking is a workaround for the per-minute cap, not something we want:
    # raise GROQ_TPM_LIMIT on a paid tier and even a 90-minute meeting takes
    # this path, which avoids the merge step losing detail across boundaries.
    if _estimate_tokens(transcript) + _prompt_overhead_tokens() <= REQUEST_TOKEN_BUDGET:
        print(f"[mom_builder] Sending transcript to Groq model '{model}'...")
        mom = _extract_from_text(client, model, transcript)
        print("[mom_builder] Received structured response from Groq.")
    else:
        # Long meeting: split, extract per chunk, then merge (prints its own progress).
        mom = _build_mom_chunked(client, model, transcript, allow_partial)

    return _strip_speaker_labels(_ensure_all_keys(mom))


class RequestTooLargeError(RuntimeError):
    """The request exceeded the per-request/per-minute token cap."""


def _is_too_large(exc: Exception) -> bool:
    """
    Tell a "request too large" apart from a transient failure.

    This one is deterministic: the same bytes will be rejected every time, so
    retrying it burns a minute of backoff and still fails. The fix is to send
    less, not to wait.
    """
    if getattr(exc, "status_code", None) == 413:
        return True
    text = str(exc).lower()
    return "request_too_large" in text or "too large" in text


def _groq_chat(client, model, messages, max_retries: int = 5) -> str:
    """
    Make one Groq chat call (JSON mode) and return the raw reply text.

    Retries with exponential backoff on transient errors — most importantly the
    429 rate-limit that free tiers return when we send several chunks quickly.
    A "request too large" is NOT transient, so it is raised immediately as
    RequestTooLargeError for the caller to handle by splitting the input.
    """
    _pace_for_tpm(sum(_estimate_tokens(str(m.get("content", ""))) for m in messages))

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
            if _is_too_large(exc):
                raise RequestTooLargeError(str(exc)) from exc
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
def _extract_chunk(client, model, chunk, part_info, depth: int = 0) -> list:
    """
    Extract one chunk, halving it and retrying if the request is too large.

    The size estimate is deliberately conservative but it is still an estimate,
    so an occasional chunk will come back over the limit. Halving it costs one
    extra request; retrying it unchanged (the old behaviour) costs a minute of
    backoff and then loses that slice of the meeting entirely.
    """
    try:
        return [_extract_from_text(client, model, chunk, part_info)]
    except RequestTooLargeError:
        halves = _split_transcript(chunk, max_tokens=max(
            MIN_CHUNK_TOKENS, _estimate_tokens(chunk) // 2), overlap_turns=0)
        if depth >= MAX_SPLIT_DEPTH or len(halves) < 2:
            raise
        print(f"[mom_builder]   Chunk was over the token limit; "
              f"splitting into {len(halves)} and retrying...")
        partials = []
        for half in halves:
            partials.extend(_extract_chunk(client, model, half, part_info, depth + 1))
        return partials


def _build_mom_chunked(client, model, transcript, allow_partial: bool = False) -> dict:
    """Map-reduce a long transcript into one MoM (see the module docstring)."""
    budget = _chunk_budget()
    chunks = _split_transcript(transcript, max_tokens=budget)
    print(f"[mom_builder] Long meeting (~{_estimate_tokens(transcript)} tokens) "
          f"-> {len(chunks)} chunks of up to {budget} tokens each "
          f"(budget: {TPM_LIMIT} tokens/min).")

    # MAP: extract a partial MoM from each chunk, tracking how much of the
    # meeting actually made it through.
    partials, succeeded, lost_chars = [], 0, 0
    total_chars = sum(len(chunk) for chunk in chunks)
    for i, chunk in enumerate(chunks, start=1):
        print(f"[mom_builder]   Extracting chunk {i}/{len(chunks)}...")
        try:
            partials.extend(_extract_chunk(client, model, chunk, (i, len(chunks))))
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            lost_chars += len(chunk)
            print(f"[mom_builder]   !! Chunk {i} FAILED: {exc}")

    coverage = 100.0 * (total_chars - lost_chars) / total_chars if total_chars else 0.0
    print(f"[mom_builder] Processed {succeeded}/{len(chunks)} chunks "
          f"({coverage:.0f}% of the transcript).")

    if not partials:
        raise RuntimeError("Every chunk failed to process; cannot build minutes.")

    # Minutes built from part of a meeting look exactly like minutes built from
    # all of it — nothing in the .docx says the middle is missing. Refuse rather
    # than hand over a document nobody can tell is incomplete.
    if lost_chars and not allow_partial:
        raise RuntimeError(
            f"{len(chunks) - succeeded} of {len(chunks)} chunks failed, so these "
            f"minutes would cover only {coverage:.0f}% of the meeting with no "
            f"indication of it in the document. Re-run with --allow-partial to "
            f"accept that, or retry once the rate limit clears (the transcript "
            f"is already saved if you used --save-transcript)."
        )

    # REDUCE: merge in code, then one LLM pass to consolidate and tidy.
    merged = _merge_partials(partials)
    print("[mom_builder]   Merged chunks; running final consolidation pass...")
    final = _consolidate_with_llm(client, model, merged)
    print("[mom_builder] Received structured response from Groq.")
    return final


def _split_transcript(transcript, max_tokens=None,
                      overlap_turns=CHUNK_OVERLAP_TURNS) -> list:
    """
    Cut the transcript into chunks of roughly `max_tokens`, breaking only
    BETWEEN whole speaker turns and carrying `overlap_turns` turns into the next
    chunk so context that straddles a boundary isn't lost.

    format_transcript() separates each "[Speaker N] (mm:ss)" block with a blank
    line. The previous version filtered those blank lines out and split on any
    line, so a chunk could begin mid-turn with its speaker header stranded in
    the previous chunk — the model then had no idea who was talking.
    """
    if max_tokens is None:
        max_tokens = _chunk_budget()

    # Whole speaker turns are the natural unit. Fall back to single lines for a
    # transcript with no blank-line blocks (--speakers none), and to a hard
    # character cut for the rare single turn that is itself over budget.
    units: list[str] = []
    for block in re.split(r"\n\s*\n", transcript):
        block = block.strip()
        if not block:
            continue
        if _estimate_tokens(block) <= max_tokens:
            units.append(block)
        else:
            units.extend(_split_oversized(block, max_tokens))

    # Measure the JOINED text each time rather than adding up per-unit counts:
    # the separators between turns are real characters, and per-unit rounding
    # accumulates, so a running total drifts a little under the true size —
    # enough to put a chunk over the cap it was built to respect.
    chunks, current = [], []
    for unit in units:
        if current and _estimate_tokens("\n\n".join(current + [unit])) > max_tokens:
            chunks.append("\n\n".join(current))
            carry = current[-overlap_turns:] if overlap_turns else []
            if carry and _estimate_tokens("\n\n".join(carry + [unit])) > max_tokens:
                carry = []  # the overlap alone would push the new chunk over
            current = carry
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_oversized(block: str, max_tokens: int) -> list:
    """Break a single over-budget speaker turn into line groups that fit."""
    pieces, current = [], []
    for line in block.split("\n"):
        # A single line longer than the whole budget: cut it on characters.
        # Shrink the cut until it fits, since Devanagari packs far more tokens
        # into the same number of characters than English does.
        while _estimate_tokens(line) > max_tokens:
            cut = max_tokens * 3
            while cut > 1 and _estimate_tokens(line[:cut]) > max_tokens:
                cut //= 2
            pieces.append(line[:cut])
            line = line[cut:]
        if current and _estimate_tokens("\n".join(current + [line])) > max_tokens:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


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
        "action item, and do NOT invent new ones.\n"
        "The parts were extracted independently, so ALSO reconcile them:\n"
        "- Unify speech-to-text spelling variants of the same person or "
        "institution into ONE consistent name (two spellings a letter or two "
        "apart are almost always the same one).\n"
        "- Strip honorifics from attendee names: 'sir', 'madam' and a trailing "
        "'-ji' are forms of address, never attendees in their own right.\n"
        "- Move anything only proposed, requested, or left unresolved out of "
        "decisions_made and into key_discussion_points.\n"
        "Return the SAME JSON schema.\n\n"
        f"{MOM_SCHEMA_DESCRIPTION}\n\n"
        f"DRAFT MINUTES (JSON):\n{json.dumps(merged, ensure_ascii=False)}"
    )
    # On a long meeting the merged notes can themselves outgrow the per-minute
    # budget. The code merge is already a complete, correct MoM — just less
    # polished — so skip the polish rather than fail or truncate it.
    if _estimate_tokens(instruction) > REQUEST_TOKEN_BUDGET:
        print("[mom_builder]   (consolidation pass skipped: the merged notes "
              "exceed the per-minute token budget; using the merged result)")
        return merged

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


# A bare "Speaker 1" is a turn marker, not a person. The prompt says so, but a
# model at temperature still emits one now and then — observed on a real run
# where two action items came back owned by "Speaker 1". Instructions can't be
# relied on for something this cheap to enforce, so we strip them in code.
_SPEAKER_LABEL = re.compile(r"^\s*speaker\s*\d+\s*$", re.IGNORECASE)


def _strip_speaker_labels(mom: dict) -> dict:
    """Remove bare "Speaker N" values from attendees and action-item owners."""
    mom["attendees"] = [
        person for person in (mom.get("attendees") or [])
        if not _SPEAKER_LABEL.match(str(person))
    ]
    for item in mom.get("action_items") or []:
        if isinstance(item, dict) and _SPEAKER_LABEL.match(str(item.get("owner", ""))):
            item["owner"] = ""
    return mom
