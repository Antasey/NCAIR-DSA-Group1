"""
Keyword extraction from patient speech — identifies clinical entities
(symptoms, duration, severity, anatomical mentions) to highlight in the
structured note for doctor review.

This version adds:
  1. `transcribe_audio()` — turns a patient recording (audio file) into text,
     using a pluggable backend, so the rest of the pipeline can run on any
     recording, not just text you already have.
  2. `extract_open_keywords()` — pulls out clinically-relevant terms that
     AREN'T in the curated SYMPTOM/SEVERITY/ANATOMY lists, using generic
     frequency + stopword filtering. This catches things like medication
     names, unusual complaints, or phrasing the fixed lists don't cover.
  3. `process_patient_recording()` — one call that goes from an audio file
     (or a raw transcript string) straight to the full keyword dict.
"""
import re
import os
import string
from collections import Counter
from typing import List, Dict, Optional, Union

# ── Clinical keywords by category (expand this based on domain knowledge) ──
SYMPTOM_KEYWORDS = [
    "pain", "fever", "cough", "vomiting", "nausea", "headache", "dizziness",
    "fatigue", "weakness", "rash", "itching", "swelling", "difficulty breathing",
    "chest pain", "abdominal pain", "back pain", "joint pain"
]
DURATION_PATTERNS = [
    r"(\d+\s*(?:day|week|month|year|hour)s?)",
    r"(since\s+(?:yesterday|today|last\s+\w+))",
    r"(for\s+\d+\s*(?:day|week|month)s?)"
]
SEVERITY_KEYWORDS = [
    "mild", "moderate", "severe", "unbearable", "manageable",
    "very bad", "really bad", "not too bad", "little bit"
]
ANATOMICAL_SITES = [
    "head", "chest", "stomach", "abdomen", "back", "arm", "leg",
    "throat", "eye", "ear", "joint", "knee", "shoulder"
]

# Common filler / stopwords to ignore when doing OPEN (non-curated) extraction.
# Kept intentionally small and generic — extend for your patient population.
_STOPWORDS = set("""
a an the this that these those i you he she it we they me him her us them
my your his its our their is am are was were be been being have has had
do does did doing will would shall should may might must can could
and or but if then so because as until while of at by for with about
against between into through during before after above below to from
up down in out on off over under again further here there when where why
how all any both each few more most other some such no nor not only own
same than too very s t just don now um uh yeah okay ok like really kind
sort feel feeling feels felt been going get got gonna
""".split())


def extract_keywords(transcript: str) -> Dict[str, List[str]]:
    """
    Extract clinical keywords from patient transcript.
    Returns dict with categories: symptoms, duration, severity, anatomy.
    """
    transcript_lower = transcript.lower()

    extracted = {
        "symptoms": [],
        "duration": [],
        "severity": [],
        "anatomical_sites": []
    }

    # Extract symptoms
    for symptom in SYMPTOM_KEYWORDS:
        if symptom in transcript_lower:
            extracted["symptoms"].append(symptom)

    # Extract duration mentions
    for pattern in DURATION_PATTERNS:
        matches = re.findall(pattern, transcript_lower, re.IGNORECASE)
        extracted["duration"].extend(matches)

    # Extract severity
    for severity in SEVERITY_KEYWORDS:
        if severity in transcript_lower:
            extracted["severity"].append(severity)

    # Extract anatomical sites
    for site in ANATOMICAL_SITES:
        if site in transcript_lower:
            extracted["anatomical_sites"].append(site)

    # Remove duplicates
    for key in extracted:
        extracted[key] = list(set(extracted[key]))

    return extracted


def extract_open_keywords(
    transcript: str,
    top_n: int = 15,
    min_word_len: int = 3,
    already_found: Optional[List[str]] = None
) -> List[str]:
    """
    Generic, non-curated keyword extraction: surfaces notable words/phrases
    from the transcript that AREN'T already covered by the fixed clinical
    lists (symptoms/severity/anatomy). Useful for catching things the
    curated lists miss — e.g. a medication name, a device, an unusual
    complaint — so a doctor can still see it flagged.

    This is intentionally lightweight (stdlib only, no NLP model download),
    using frequency + stopword filtering + simple bigram detection. It is
    NOT a substitute for the curated lists — pair the two.
    """
    already_found = set(w.lower() for w in (already_found or []))

    # Tokenize: strip punctuation, lowercase
    cleaned = transcript.lower().translate(str.maketrans("", "", string.punctuation))
    words = [w for w in cleaned.split() if len(w) >= min_word_len and w not in _STOPWORDS]

    if not words:
        return []

    # Unigram frequency
    freq = Counter(words)

    # Simple bigram candidates (adjacent non-stopword pairs), which often
    # capture things like "blood pressure", "left knee", "throwing up"
    bigrams = Counter()
    raw_tokens = cleaned.split()
    for i in range(len(raw_tokens) - 1):
        w1, w2 = raw_tokens[i], raw_tokens[i + 1]
        if (len(w1) >= min_word_len and len(w2) >= min_word_len
                and w1 not in _STOPWORDS and w2 not in _STOPWORDS):
            bigrams[f"{w1} {w2}"] += 1

    candidates = Counter()
    candidates.update(freq)
    candidates.update(bigrams)

    # Drop anything already captured by the curated extractors
    ranked = [
        term for term, _ in candidates.most_common()
        if term not in already_found
    ]

    return ranked[:top_n]


# ─────────────────────────── Audio transcription ───────────────────────────

def transcribe_audio(
    audio_path: str,
    backend: str = "auto",
    language: str = "en-US"
) -> str:
    """
    Transcribe a patient recording (e.g. .wav, .flac, .aiff) to text.

    `backend` controls which speech-to-text engine is used:
      - "sphinx"       : fully offline (CMU Sphinx), lower accuracy, no
                         network/API key needed. Good default for a
                         hospital environment with strict data policies.
      - "google_cloud" : Google Cloud Speech-to-Text, needs
                         GOOGLE_APPLICATION_CREDENTIALS configured and
                         network access. Higher accuracy.
      - "whisper"      : OpenAI Whisper (local model via the `whisper`
                         package), fully offline, best accuracy of the
                         three, needs more compute.
      - "auto"         : try whisper -> sphinx -> google_cloud, using
                         whichever is installed/configured.

    Raises RuntimeError with clear setup instructions if no backend is
    available in the current environment — since transcribing audio
    generally needs either a local model or a configured cloud service,
    neither of which this sandbox has installed by default.

    IMPORTANT (hospital/PHI context): if you use a cloud backend, confirm
    it's covered by your institution's BAA (HIPAA business associate
    agreement) before sending patient audio to it. Offline backends
    (sphinx, whisper) avoid that question entirely by not sending audio
    anywhere.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    tried = []

    def _try_whisper():
        import whisper  # pip install -U openai-whisper (also needs ffmpeg)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language=language.split("-")[0])
        return result["text"].strip()

    def _try_sphinx():
        import speech_recognition as sr  # pip install SpeechRecognition pocketsphinx
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        return r.recognize_sphinx(audio, language=language)

    def _try_google_cloud():
        import speech_recognition as sr  # pip install SpeechRecognition google-cloud-speech
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        return r.recognize_google_cloud(audio, language=language)

    backends = {
        "whisper": _try_whisper,
        "sphinx": _try_sphinx,
        "google_cloud": _try_google_cloud,
    }

    order = [backend] if backend != "auto" else ["whisper", "sphinx", "google_cloud"]

    for name in order:
        fn = backends.get(name)
        if fn is None:
            continue
        try:
            return fn()
        except ImportError as e:
            tried.append(f"{name}: not installed ({e})")
        except Exception as e:
            tried.append(f"{name}: failed ({e})")

    raise RuntimeError(
        "No working speech-to-text backend available. Tried:\n  "
        + "\n  ".join(tried)
        + "\n\nTo enable transcription, install one of:\n"
        "  pip install -U openai-whisper           # offline, best accuracy, needs ffmpeg\n"
        "  pip install SpeechRecognition pocketsphinx  # offline, lower accuracy\n"
        "  pip install SpeechRecognition google-cloud-speech  # cloud, needs credentials + BAA\n"
    )


def process_patient_recording(
    source: str,
    is_audio: Optional[bool] = None,
    transcribe_backend: str = "auto",
    include_open_keywords: bool = True
) -> Dict[str, Union[str, Dict[str, List[str]], List[str]]]:
    """
    End-to-end: given either an audio file path or a raw transcript string,
    return the transcript plus the full keyword breakdown.

    `is_audio`: force interpretation of `source`. If None, it's inferred
    from whether `source` looks like a path to an existing audio file.
    """
    audio_extensions = (".wav", ".flac", ".aiff", ".mp3", ".m4a", ".ogg")

    if is_audio is None:
        is_audio = os.path.exists(source) and source.lower().endswith(audio_extensions)

    if is_audio:
        transcript = transcribe_audio(source, backend=transcribe_backend)
    else:
        transcript = source

    keywords = extract_keywords(transcript)

    result = {
        "transcript": transcript,
        "keywords": keywords,
    }

    if include_open_keywords:
        already = (
            keywords["symptoms"] + keywords["severity"] + keywords["anatomical_sites"]
        )
        result["open_keywords"] = extract_open_keywords(transcript, already_found=already)

    return result


# ─────────────────────────────── Rendering ──────────────────────────────────

def highlight_keywords(note: Dict, keywords: Dict[str, List[str]]) -> str:
    """
    Render the structured note as HTML with keywords highlighted.
    Doctor sees the note with color-coded keywords for quick scanning.
    """
    all_keywords = []
    for category, items in keywords.items():
        all_keywords.extend(items)

    html = "<div style='font-family: Arial; line-height: 1.6;'>"

    # Chief Complaint
    chief = note.get("chief_complaint", "")
    html += f"<h4>Chief Complaint</h4>"
    html += f"<p>{_highlight_text(chief, all_keywords)}</p>"

    # Duration
    duration = note.get("duration", "")
    html += f"<h4>Duration</h4>"
    html += f"<p>{_highlight_text(duration, keywords.get('duration', []))}</p>"

    # Severity
    severity = note.get("severity", "")
    html += f"<h4>Severity</h4>"
    html += f"<p>{_highlight_text(severity, keywords.get('severity', []))}</p>"

    # History
    history = note.get("history", "")
    html += f"<h4>Relevant History</h4>"
    html += f"<p>{_highlight_text(history, all_keywords)}</p>"

    # Open / uncategorized keywords (only if present)
    open_kw = keywords.get("open_keywords", [])
    if open_kw:
        html += "<h4>Other Notable Terms</h4>"
        html += "<p>" + ", ".join(
            f'<mark style="background-color: #FFD580;">{kw}</mark>' for kw in open_kw
        ) + "</p>"

    html += "</div>"
    return html


def _highlight_text(text: str, keywords: List[str]) -> str:
    """
    Internal: highlight keywords in text with yellow background.
    """
    if not text:
        return text

    highlighted = text
    for keyword in set(keywords):  # avoid duplicate highlighting
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark style="background-color: #FFFF00;">{keyword}</mark>',
            highlighted
        )
    return highlighted


if __name__ == "__main__":
    # Example: text-only path (no audio file needed to test the logic)
    sample_transcript = (
        "I've had this really bad headache for 3 days, and my left knee "
        "has been swelling up since yesterday. I'm also taking ibuprofen "
        "but it's not helping much."
    )
    result = process_patient_recording(sample_transcript, is_audio=False)
    print(result)
