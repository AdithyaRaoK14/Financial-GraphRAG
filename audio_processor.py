"""
audio_processor.py
==================

High-quality earnings call processor.

Features
--------
✓ faster-whisper (large-v3) transcription, biased with a domain
  vocabulary prompt — no Hugging Face token, lower VRAM + faster
  inference than openai-whisper
✓ GPU support
✓ Timestamp preservation
✓ Transcript cleanup (case-preserving — earlier version force-lowercased
  every transcript because normalize_financial_terms lowercased the
  WHOLE string instead of just the matched terms)
✓ Financial terminology + observed-ASR-error corrections
✓ Word-count + sentence-boundary aware merging (not raw character count)
✓ Best-effort speaker-role tagging (Operator/Analyst cues — NOT real
  diarization; see note below)
✓ JSON-ready output
✓ Chunk metadata incl. document_name, source_file_hash, processed_at

Dependencies
------------
pip install faster-whisper

faster-whisper is a CTranslate2 reimplementation of the same Whisper
weights — same model, same transcription quality, no Hugging Face token,
noticeably lower VRAM usage and faster inference than openai-whisper
(useful on smaller GPUs like a GTX 1650). On GPU it needs cuBLAS/cuDNN
available; if those aren't installed, set device to "cpu" or use
compute_type="int8" for CPU-friendly quantization.

Known limitation — no real speaker diarization
------------------------------------------------
`detect_speaker_role()` below is a cue-phrase heuristic (catches operator
hand-offs and "thank you for the opportunity" style analyst openers). It
cannot tell individual speakers apart or reliably label management
answers. For real turn-by-turn speaker labels, swap this processor for
WhisperX with pyannote diarization — that needs a GPU, a HuggingFace
auth token for the pyannote model, and is a bigger infra change than a
one-file edit, so it isn't wired in here. Rough shape of that swap:

    import whisperx
    model = whisperx.load_model(model_size, device)
    result = whisperx.align(model.transcribe(audio_path), ...)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN)
    result = whisperx.assign_word_speakers(diarize_model(audio_path), result)

Each segment in `result["segments"]` then carries a real `speaker` label
you can drop straight into build_chunk() in place of detect_speaker_role().
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Data Classes
# -------------------------------------------------------


@dataclass
class AudioSegment:
    start: float

    end: float

    text: str


# -------------------------------------------------------
# Processor
# -------------------------------------------------------


class AudioProcessor:
    # Fed to Whisper as initial_prompt to bias decoding toward these terms
    # (Option B from the review). Company names come from config so this
    # stays in sync with whatever companies the project is processing.
    DOMAIN_VOCABULARY = (
        ", ".join(getattr(config, "COMPANIES", []))
        + ", Motilal Oswal, EBITDA, GNPA, NNPA, PAT, PPOP, ROA, ROE, CASA, "
        "NIM, AUM, CAGR, EPS, Book Value, PCR, YoY, QoQ, disbursement, "
        "collections, microfinance, delinquencies, provisioning, ALM, "
        "cost of funds, portfolio yield, FY24, FY25"
    )

    def __init__(self, model_size="large-v3"):

        self.model_size = model_size

        self.model = None

    # ---------------------------------------------------

    def load_model(self):

        if self.model is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # int8_float16 keeps large-v3 fitting comfortably on small-VRAM
        # GPUs (e.g. a 4GB GTX 1650) while staying close to float16
        # quality. Drop to "int8" via config if you still hit
        # out-of-memory errors, or "float16"/"float32" if you have more
        # VRAM to spare and want maximum fidelity.
        compute_type = config.WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper %s on %s (compute_type=%s)",
            self.model_size,
            device,
            compute_type,
        )

        self.model = WhisperModel(
            self.model_size, device=device, compute_type=compute_type
        )

    # ---------------------------------------------------

    def transcribe(self, audio_path):

        self.load_model()

        logger.info("Transcribing %s", audio_path)

        segments_iter, _info = self.model.transcribe(
            str(audio_path),
            language="en",
            beam_size=config.WHISPER_BEAM_SIZE,
            best_of=config.WHISPER_BEST_OF,
            temperature=config.WHISPER_TEMPERATURE,
            vad_filter=config.WHISPER_VAD_FILTER,
            condition_on_previous_text=config.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=config.WHISPER_LOG_PROB_THRESHOLD,
            no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
            initial_prompt=self.DOMAIN_VOCABULARY,
        )

        segments = []

        for seg in segments_iter:
            txt = seg.text.strip()

            if not txt:
                continue

            segments.append(AudioSegment(start=seg.start, end=seg.end, text=txt))

        return segments

    # ---------------------------------------------------
    # Cleanup
    # ---------------------------------------------------

    def clean_text(self, text):

        text = re.sub(r"\s+", " ", text)

        text = text.replace(" uh ", " ")

        text = text.replace(" um ", " ")

        text = text.replace(" you know ", " ")

        text = text.replace(" sort of ", " ")

        text = text.replace(" kind of ", " ")

        text = text.replace(" okay ", " ")

        text = text.strip()

        return text

    # ---------------------------------------------------
    # Financial term normalization (Option C — expanded dictionary)
    # ---------------------------------------------------

    FINANCIAL_TERMS = {
        "ebit da": "EBITDA",
        "e bit da": "EBITDA",
        "gross n p a": "Gross NPA",
        "net n p a": "Net NPA",
        "a u m": "AUM",
        "profit after tax": "PAT",
        "year on year": "YoY",
        "quarter on quarter": "QoQ",
        "p p o p": "PPOP",
        "return on assets": "ROA",
        "return on equity": "ROE",
        "c a s a": "CASA",
        "net interest margin": "NIM",
        "c a g r": "CAGR",
        "earnings per share": "EPS",
        "book value": "Book Value",
        "operating margin": "Operating Margin",
        "provision coverage ratio": "PCR",
        "cost of funds": "Cost of Funds",
        "assets under management": "AUM",
        "gross non performing assets": "Gross NPA",
        "net non performing assets": "Net NPA",
    }

    # Corrections for specific ASR errors observed in real transcripts —
    # this list is a starting point, not exhaustive. Extend it as you
    # process more calls and spot new recurring mistranscriptions;
    # treat every entry as a hypothesis to sanity-check against the
    # audio, not ground truth.
    CORRECTIONS = {
        "credit access gramm in": "CreditAccess Grameen's",
        "credit access gramm": "CreditAccess Grameen",
        "motilaal oswal": "Motilal Oswal",
        "motilal orstral": "Motilal Oswal",
        "delicacies": "delinquencies",
        "cut-em-as": "customers",
        "cut income ratio": "cost-to-income ratio",
        "oricial poisoning": "overall provisioning",
        "a&m position": "ALM position",
        "easier coverage": "ECL coverage",
        "easier provisioning": "ECL provisioning",
    }

    def normalize_financial_terms(self, text):
        """
        Case-insensitive term replacement that preserves the casing of
        everything else in the string. The previous version called
        text.lower() on the WHOLE segment before returning it — that's
        why every transcript came back fully lowercase regardless of
        what Whisper actually produced.
        """

        for src, dst in {**self.FINANCIAL_TERMS, **self.CORRECTIONS}.items():
            text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

        return text

    # ---------------------------------------------------
    # Speaker-role heuristic (best-effort — see module docstring)
    # ---------------------------------------------------

    def detect_speaker_role(self, text):

        lowered = text.lower()

        if re.search(r"next question is from the line of", lowered):
            return "Operator"

        if re.search(r"please go ahead|please show a hand", lowered):
            return "Operator"

        if re.search(
            r"ladies and gentlemen|welcome to the|we will now begin the question",
            lowered,
        ):
            return "Operator"

        if re.search(
            r"thank you for the opportunity|thanks for the opportunity|"
            r"thank you for giving me the opportunity|thank you for giving the opportunity",
            lowered,
        ):
            return "Analyst"

        return None  # most likely management commentary, but unconfirmed

    # ---------------------------------------------------
    # Merge tiny segments (word-count target, cut at sentence
    # boundaries instead of accumulating by raw character count)
    # ---------------------------------------------------

    def merge_segments(self, segments, target_words=None, max_words=None):

        target_words = target_words or config.WHISPER_CHUNK_TARGET_WORDS
        max_words = max_words or config.WHISPER_CHUNK_MAX_WORDS

        merged = []

        current = None
        current_words = 0

        for seg in segments:
            seg.text = self.clean_text(seg.text)

            seg.text = self.normalize_financial_terms(seg.text)

            seg_words = len(seg.text.split())

            if current is None:
                current = seg
                current_words = seg_words
                continue

            current.text += " " + seg.text
            current.end = seg.end
            current_words += seg_words

            # Prefer cutting once we've reached the target AND the text
            # ends at a sentence boundary — avoids splitting a thought in
            # the middle just because the word count ticked over. But if
            # sentence-ending punctuation is sparse for a long stretch
            # (long unpunctuated Q&A cross-talk, for example), max_words
            # forces a cut anyway — without this, one segment could grow
            # unboundedly (a real run produced a single ~35-minute,
            # ~4900-word chunk this way).
            ends_at_boundary = current.text.rstrip().endswith((".", "?", "!"))

            if (
                current_words >= target_words and ends_at_boundary
            ) or current_words >= max_words:
                merged.append(current)
                current = None
                current_words = 0

        if current:
            merged.append(current)

        return merged

    # ---------------------------------------------------
    # Chunk ID
    # ---------------------------------------------------

    def chunk_id(self, text, meta):

        # Same fix as chunker.py's Chunker.chunk_id(): processed_at is a
        # fresh timestamp every run and source_file_hash changes on any
        # re-save of the source audio file, even with identical content.
        # Hashing either means reprocessing the exact same audio segment
        # produces a DIFFERENT chunk_id, breaking every downstream
        # reference to the old one (Neo4j AudioChunk nodes, embeddings,
        # ground truth provenance) — this path bypasses chunker.py
        # entirely (see data_processing.py), so that earlier fix never
        # covered audio chunks; this closes the same gap here.
        stable_meta = {
            k: v
            for k, v in meta.items()
            if k not in ("processed_at", "source_file_hash")
        }

        raw = text + json.dumps(stable_meta, sort_keys=True)

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------
    # File hash (for source_file_hash metadata / traceability)
    # ---------------------------------------------------

    def file_hash(self, path):

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)

        return h.hexdigest()[:16]

    # ---------------------------------------------------
    # Build Chunk
    # ---------------------------------------------------

    def build_chunk(
        self,
        segment,
        company,
        year,
        quarter,
        document_name=None,
        source_file_hash=None,
        processed_at=None,
    ):

        embedding_text = segment.text

        meta = {
            "company": company,
            "year": year,
            "quarter": quarter,
            "document_name": document_name,
            "source_type": "audio",
            "chunk_type": "audio",
            "speaker": self.detect_speaker_role(segment.text),
            "source_file_hash": source_file_hash,
            "processed_at": processed_at,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
        }

        return {
            **meta,
            "text": segment.text,
            "embedding_text": embedding_text,
            "chunk_id": self.chunk_id(
                embedding_text,
                meta,
            ),
        }

    # ---------------------------------------------------
    # Semantic Merge
    # ---------------------------------------------------

    TOPIC_WORDS = [
        "revenue",
        "margin",
        "profit",
        "guidance",
        "question",
        "operator",
        "expenses",
        "segment",
        "cash flow",
        "capex",
        "ebitda",
        "npa",
        "aum",
        "outlook",
    ]

    def semantic_merge(self, segments):

        merged = []

        current = None

        for seg in segments:
            if current is None:
                current = seg

                continue

            same_topic = False

            for word in self.TOPIC_WORDS:
                if word in seg.text.lower() and word in current.text.lower():
                    same_topic = True

                    break

            if same_topic:
                current.text += "\n" + seg.text

                current.end = seg.end

            else:
                merged.append(current)

                current = seg

        if current:
            merged.append(current)

        return merged

    # ---------------------------------------------------
    # Remove duplicate sentences
    # ---------------------------------------------------

    def deduplicate(self, segments):

        cleaned = []

        seen = set()

        for seg in segments:
            text = seg.text.strip()

            if text in seen:
                continue

            seen.add(text)

            cleaned.append(seg)

        return cleaned

    # ---------------------------------------------------
    # Build final JSON
    # ---------------------------------------------------

    def process(
        self,
        audio_path,
        company,
        year,
        quarter,
        document_name=None,
    ):

        if document_name is None:
            document_name = Path(audio_path).name

        source_file_hash = self.file_hash(audio_path)
        processed_at = datetime.now(timezone.utc).isoformat()

        segments = self.transcribe(audio_path)

        segments = self.merge_segments(segments)

        segments = self.deduplicate(segments)

        output = []

        for seg in segments:
            output.append(
                self.build_chunk(
                    seg,
                    company,
                    year,
                    quarter,
                    document_name,
                    source_file_hash,
                    processed_at,
                )
            )

        return output

    # ---------------------------------------------------
    # Export
    # ---------------------------------------------------

    def export_json(
        self,
        audio_path,
        company,
        year,
        quarter,
        output_path,
        document_name=None,
    ):

        chunks = self.process(
            audio_path,
            company,
            year,
            quarter,
            document_name,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return chunks


# -------------------------------------------------------
# Singleton
# -------------------------------------------------------

_processor = AudioProcessor(config.WHISPER_MODEL_SIZE)


def extract_audio(
    audio_path,
    company,
    year,
    quarter,
    document_name=None,
):

    return _processor.process(
        audio_path,
        company,
        year,
        quarter,
        document_name,
    )


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("audio")

    parser.add_argument("--company")

    parser.add_argument("--year")

    parser.add_argument("--quarter")

    parser.add_argument("--document-name", default=None)

    args = parser.parse_args()

    chunks = extract_audio(
        args.audio,
        args.company,
        args.year,
        args.quarter,
        args.document_name,
    )

    print(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False,
        )
    )
