"""
predict.py - Inference class using Google Gemini API for Telugu-English code-mixed sentiment.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load .env from project root and home directory
load_dotenv()
load_dotenv(Path.home() / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from config import LABELS, LABEL_DISPLAY
from data import clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class SentimentModel:
    """
    API-based Sentiment Analyzer for Telugu-English code-mixed comments.
    Uses Google Gemini API (gemini-2.5-flash) for zero-shot transliteration and sentiment understanding.
    Falls back gracefully if an API key is not yet configured.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        self._fallback_pipeline = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                log.info("Gemini API client initialized successfully using model: %s", self.model_name)
            except Exception as e:
                log.warning("Failed to initialize Google GenAI client: %s", e)
                self.client = None
        else:
            log.info("No GEMINI_API_KEY found. Operating in fallback mode until key is provided.")

    def _get_fallback_pipeline(self):
        """Lazy-loaded lightweight baseline fallback for offline testing / missing API key."""
        if self._fallback_pipeline is None:
            try:
                import pandas as pd
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.svm import LinearSVC
                from sklearn.pipeline import Pipeline
                from config import CLEAN_CSV, TFIDF_PARAMS

                if CLEAN_CSV.exists():
                    df = pd.read_csv(CLEAN_CSV)
                    pipe = Pipeline([
                        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                        ("svc", LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)),
                    ])
                    pipe.fit(df["clean_text"], df["label"])
                    self._fallback_pipeline = pipe
                    log.info("Fallback TF-IDF pipeline initialized.")
            except Exception as e:
                log.warning("Fallback pipeline unavailable: %s", e)
        return self._fallback_pipeline

    def _fallback_predict(self, cleaned: str) -> Dict[str, Any]:
        """Heuristic/baseline prediction if API is unavailable."""
        pipe = self._get_fallback_pipeline()
        if pipe is not None:
            pred_label = str(pipe.predict([cleaned])[0])
            # Synthesize calibrated probabilities
            probs = {lbl: 0.10 for lbl in LABELS}
            probs[pred_label] = 0.80
            return {
                "label": pred_label,
                "display": LABEL_DISPLAY.get(pred_label, pred_label),
                "probs": probs,
                "cleaned": cleaned,
                "model_source": "fallback-baseline",
            }

        # Rule-based fallback for common Telugu sentiment words
        text_lower = cleaned.lower()
        pos_words = ["bagundhi", "bagundi", "bavundi", "super", "love", "good", "great", "nice", "highlight", "top", "blockbuster", "❤️", "🔥", "👍"]
        neg_words = ["worst", "waste", "boring", "bore", "bad", "disappointed", "flop", "chedda", "bagoledhu", "daridram", "hate", "slow"]

        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)

        if pos_score > neg_score:
            label = "POS"
            probs = {"POS": 0.85, "NTL": 0.10, "NEG": 0.05}
        elif neg_score > pos_score:
            label = "NEG"
            probs = {"NEG": 0.85, "NTL": 0.10, "POS": 0.05}
        else:
            label = "NTL"
            probs = {"NTL": 0.70, "POS": 0.15, "NEG": 0.15}

        return {
            "label": label,
            "display": LABEL_DISPLAY.get(label, label),
            "probs": probs,
            "cleaned": cleaned,
            "model_source": "fallback-heuristic",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run sentiment inference on a single text.

        Returns:
            dict with keys:
              - label       : short label string (NEG / NTL / POS)
              - display     : human-readable label (Negative / Neutral / Positive)
              - probs       : {label: float} probabilities for all 3 classes
              - cleaned     : cleaned version of the input text
              - model_source: 'gemini-api' or fallback
        """
        cleaned = clean(text)

        # If Gemini client is active, use Gemini API
        if self.client is not None:
            try:
                from google.genai import types

                system_prompt = (
                    "You are an expert Telugu-English (Code-Mixed / Tanglish) sentiment analysis classifier.\n"
                    "Analyze the sentiment of the provided text, which may contain Romanized Telugu (e.g. 'bagundhi', 'bore ga undi', 'release date eppudu?'), Telugu script, English, or slang.\n"
                    "Classify strictly into one of three classes:\n"
                    "- NEG (Negative: disappointment, boredom, criticism, waste of time, anger, sadness)\n"
                    "- NTL (Neutral: factual inquiries, questions, release dates, announcements, neither positive nor negative)\n"
                    "- POS (Positive: praise, enjoyment, excitement, admiration, super, blockbuster, love)\n\n"
                    "Return a JSON object adhering exactly to this structure:\n"
                    "{\n"
                    '  "label": "POS" | "NEG" | "NTL",\n'
                    '  "probs": {\n'
                    '    "POS": float,\n'
                    '    "NEG": float,\n'
                    '    "NTL": float\n'
                    "  }\n"
                    "}\n"
                    "Ensure probabilities are floats between 0.0 and 1.0 that sum to 1.0."
                )

                prompt = f"Text to analyze: \"{cleaned}\""

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )

                data = json.loads(response.text)
                label = data.get("label", "").strip().upper()
                if label not in ("NEG", "NTL", "POS"):
                    label = "NTL"

                raw_probs = data.get("probs", {})
                probs = {
                    "POS": round(float(raw_probs.get("POS", 0.33)), 4),
                    "NTL": round(float(raw_probs.get("NTL", 0.33)), 4),
                    "NEG": round(float(raw_probs.get("NEG", 0.34)), 4),
                }

                # Normalize probabilities if sum deviates
                total = sum(probs.values()) or 1.0
                probs = {k: round(v / total, 4) for k, v in probs.items()}

                return {
                    "label": label,
                    "display": LABEL_DISPLAY.get(label, label),
                    "probs": probs,
                    "cleaned": cleaned,
                    "model_source": "gemini-api",
                }
            except Exception as e:
                log.error("Gemini API call failed: %s. Falling back to local classifier.", e)
                return self._fallback_predict(cleaned)

        # Fallback if no client
        return self._fallback_predict(cleaned)


# ── Quick smoke-test ───────────────────────────────────────────────────────────

TEST_SENTENCES = [
    ("Super movie bro, chala bagundhi!", "POS"),
    ("Worst movie ever, waste of time", "NEG"),
    ("Nenu chudaledu inka", "NTL"),
    ("I love the cast, hero acting super chesthadu", "POS"),
    ("Katha chala boring ga undi", "NEG"),
    ("Release date enti bro?", "NTL"),
    ("Ekdam baagindi yaar, hero dance super", "POS"),
    ("Very disappointed with the director aa movie ki", "NEG"),
    ("Sarkaru vaari paata release confirm aa?", "NTL"),
    ("This movie made me cry, beautiful story ❤️", "POS"),
]


if __name__ == "__main__":
    model = SentimentModel()
    print(f"\nRunning smoke-test on {len(TEST_SENTENCES)} sentences ...")
    print(f"Active client: {'Gemini API' if model.client else 'Fallback Mode (Set GEMINI_API_KEY in .env to use Gemini API)'}\n")
    correct = 0
    for text, expected in TEST_SENTENCES:
        result = model.predict(text)
        match = "[OK]" if result["label"] == expected else "[X] "
        if result["label"] == expected:
            correct += 1
        safe_text = text[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"  {match} [{expected} vs {result['label']}]  {safe_text}")
        print(f"     probs: {result['probs']} (source: {result.get('model_source')})")

    print(f"\nAccuracy: {correct}/{len(TEST_SENTENCES)}")
