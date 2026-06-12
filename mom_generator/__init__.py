"""
mom_generator
=============

A small package that turns a meeting audio file into a structured
Minutes of Meeting (MoM) Word document.

Pipeline (each step lives in its own module so it is easy to follow):

    audio file
        -> transcriber.transcribe_audio()   (faster-whisper, runs locally)
        -> diarizer.diarize_audio()          (Resemblyzer, real voice labels) [optional]
        -> mom_builder.build_mom()           (Groq LLM, structures the notes)
        -> docx_exporter.export_to_docx()    (python-docx, writes the .docx)

Import the high-level helpers directly from the package:

    from mom_generator import transcribe_audio, build_mom, export_to_docx
"""

from .transcriber import transcribe_audio, format_transcript
from .mom_builder import build_mom
from .docx_exporter import export_to_docx

__all__ = [
    "transcribe_audio",
    "format_transcript",
    "build_mom",
    "export_to_docx",
]

# Real voice-based speaker recognition is optional because it needs the heavier
# resemblyzer/torch stack. Expose it only if those extras are installed, so the
# core pipeline keeps working without them.
try:
    from .diarizer import diarize_audio, assign_voice_speakers  # noqa: F401
    __all__ += ["diarize_audio", "assign_voice_speakers"]
except ImportError:
    pass
