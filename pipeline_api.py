import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------- TRANSFORMERS SUPPORT ----------------
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        BartTokenizer,
        BartForConditionalGeneration,
        pipeline as hf_pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# ---------------- CONFIG ----------------
# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

PEGASUS_MODEL = "google/pegasus-xsum"
BART_MODEL = "facebook/bart-large-cnn"
FALLBACK_SUMMARIZER = "sshleifer/distilbart-cnn-12-6"

MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

# New ALLOWED includes text:
ALLOWED = {
    "audio": {"exts": {"mp3", "wav", "m4a", "ogg"}},
    "video": {"exts": {"mp4", "webm", "mov"}},
    "text": {"exts": {"txt", "md"}}
}

ACTION_KEYWORDS = [
    "will", "should", "need to", "needs to", "has to", "have to",
    "to work on", "to complete", "to finish", "to send", "to finalize",
    "to review", "to update", "assigned to", "follow up",
    "next meeting", "schedule", "must", "plan to"
]

SUMMARY_LENGTH_CONFIG = {
    "short": {"max_length": 80, "min_length": 30},
    "medium": {"max_length": 140, "min_length": 60},
    "long": {"max_length": 250, "min_length": 120},
}

nltk.download("punkt", quiet=True)

# ---------------- LAZY MODELS ----------------
MODELS = {
    "whisper": None,
    "pegasus_tokenizer": None,
    "pegasus_model": None,
    "bart_tokenizer": None,
    "bart_model": None,
    "fallback_pipeline": None,
    "device": torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"),
    "transformers_ok": TRANSFORMERS_AVAILABLE
}

# ---------------- FASTAPI ----------------
# ---------------- FASTAPI ----------------
app = FastAPI(title="Summarizer Backend (Audio + Video + Text)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

@app.get("/")
def root():
    f = BASE_DIR / "index.html"
    return FileResponse(f) if f.exists() else {"message": "index.html missing"}


# ---------------- MODEL LOADER ----------------
def _load_models_once():
    """Load Whisper, Pegasus, BART lazily."""
    if MODELS["whisper"] is None:
        print("Loading Whisper...")
        MODELS["whisper"] = whisper.load_model(WHISPER_MODEL)

    if MODELS["transformers_ok"]:

        if MODELS["pegasus_model"] is None:
            try:
                print("Loading Pegasus...")
                MODELS["pegasus_tokenizer"] = AutoTokenizer.from_pretrained(PEGASUS_MODEL)
                MODELS["pegasus_model"] = AutoModelForSeq2SeqLM.from_pretrained(PEGASUS_MODEL).to(MODELS["device"])
            except Exception as e:
                print("Pegasus failed:", e)

        if MODELS["bart_model"] is None:
            try:
                print("Loading BART...")
                MODELS["bart_tokenizer"] = BartTokenizer.from_pretrained(BART_MODEL)
                MODELS["bart_model"] = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(MODELS["device"])
            except Exception as e:
                print("BART failed:", e)

        if MODELS["pegasus_model"] is None and MODELS["bart_model"] is None:
            try:
                print("Loading fallback summarizer...")
                MODELS["fallback_pipeline"] = hf_pipeline(
                    "summarization", model=FALLBACK_SUMMARIZER,
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                print("Fallback summarizer failed.")

async def ensure_loaded():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_models_once)



# ---------------- HELPERS ----------------
async def save_upload(upload: UploadFile, mediaType: str) -> str:
    """Save uploaded file to disk."""
    ext = (upload.filename or "").split(".")[-1].lower()

    if mediaType not in ALLOWED:
        raise HTTPException(400, f"Invalid mediaType: {mediaType}")

    if ext not in ALLOWED[mediaType]["exts"]:
        raise HTTPException(400, f"Unsupported file type for {mediaType}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    size = 0

    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_SIZE_BYTES:
            raise HTTPException(400, "File too large (max 2GB)")
        tmp.write(chunk)

    tmp.close()
    return tmp.name


def transcribe_audio(path: str, language: Optional[str]):
    """Whisper transcription."""
    result = MODELS["whisper"].transcribe(path, language=language)
    transcript = result.get("text", "").strip()

    duration = None
    seg = result.get("segments")
    if seg:
        duration = float(seg[-1].get("end", 0))

    return transcript, duration


def summarize_pegasus(text: str, length: str) -> Optional[str]:
    tok = MODELS["pegasus_tokenizer"]
    model = MODELS["pegasus_model"]
    if not tok or not model:
        return None

    cfg = SUMMARY_LENGTH_CONFIG[length]
    inp = tok(text, truncation=True, padding="longest", return_tensors="pt").to(MODELS["device"])
    ids = model.generate(
        inp["input_ids"],
        max_length=cfg["max_length"], min_length=cfg["min_length"],
        num_beams=4, no_repeat_ngram_size=3
    )
    return tok.decode(ids[0], skip_special_tokens=True)


def summarize_bart(text: str, length: str) -> Optional[str]:
    tok = MODELS["bart_tokenizer"]
    model = MODELS["bart_model"]
    if not tok or not model:
        return None

    cfg = SUMMARY_LENGTH_CONFIG[length]
    inp = tok(text, truncation=True, padding="longest", return_tensors="pt").to(MODELS["device"])
    ids = model.generate(
        inp["input_ids"],
        max_length=cfg["max_length"], min_length=cfg["min_length"],
        num_beams=4, no_repeat_ngram_size=3
    )
    return tok.decode(ids[0], skip_special_tokens=True)


def summarize_fallback(text: str, length: str):
    pip = MODELS["fallback_pipeline"]
    cfg = SUMMARY_LENGTH_CONFIG[length]

    if pip:
        out = pip(text, max_length=cfg["max_length"], min_length=cfg["min_length"], truncation=True)
        return out[0]["summary_text"]

    # basic extractive fallback
    sent = nltk.sent_tokenize(text)
    count = {"short": 1, "medium": 2, "long": 4}[length]
    return " ".join(sent[:count])


def extract_actions(text: str) -> List[str]:
    out = []
    for s in nltk.sent_tokenize(text):
        if any(k in s.lower() for k in ACTION_KEYWORDS):
            out.append(s)
    return list(dict.fromkeys(out))[:50]


# ---------------- RESPONSE MODEL ----------------
class AnalyzeResponse(BaseModel):
    pegasus_summary: Optional[str]
    bart_summary: Optional[str]
    summary: str
    action_items: List[str]
    transcript: str
    duration_seconds: Optional[float]


# ---------------- MAIN ENDPOINT ----------------
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    mediaType: str = Form(...),
    summaryLength: str = Form("short"),
    extractActions: str = Form("true"),
    language: str = Form("en")
):
    await ensure_loaded()

    if summaryLength not in SUMMARY_LENGTH_CONFIG:
        summaryLength = "short"

    # Save file
    temp_path = await save_upload(file, mediaType)

    # Handle TEXT files separately -------------------
    if mediaType == "text":
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            transcript = f.read().strip()
        os.remove(temp_path)
        duration = None

    else:
        # Audio/video -> Whisper
        transcript, duration = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_audio, temp_path, language
        )
        os.remove(temp_path)

    if not transcript.strip():
        raise HTTPException(400, "Empty transcript")

    # Summaries -------------------------------------
    loop = asyncio.get_event_loop()
    peg = await loop.run_in_executor(None, summarize_pegasus, transcript, summaryLength)
    bart = await loop.run_in_executor(None, summarize_bart, transcript, summaryLength)

    if peg is None:
        peg = await loop.run_in_executor(None, summarize_fallback, transcript, summaryLength)
    if bart is None:
        bart = await loop.run_in_executor(None, summarize_fallback, transcript, summaryLength)

    default = peg or bart or summarize_fallback(transcript, summaryLength)

    actions = extract_actions(transcript) if extractActions.lower() == "true" else []

    return AnalyzeResponse(
        pegasus_summary=peg,
        bart_summary=bart,
        summary=default,
        action_items=actions,
        transcript=transcript,
        duration_seconds=duration
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper_loaded": MODELS["whisper"] is not None,
        "pegasus_loaded": MODELS["pegasus_model"] is not None,
        "bart_loaded": MODELS["bart_model"] is not None,
        "fallback_loaded": MODELS["fallback_pipeline"] is not None,
        "device": str(MODELS["device"])
    }