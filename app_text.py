from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import traceback

# -------------------------------------------------
# Create the FastAPI app
# -------------------------------------------------
app = FastAPI(title="Email Triage & Draft API")

# -------------------------------------------------
# Load summarization model safely at startup
# -------------------------------------------------
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarizer model loaded successfully.")
except Exception as e:
    print("⚠️ Failed to load summarizer model:", e)
    summarizer = None


# -------------------------------------------------
# Request models
# -------------------------------------------------
class DraftRequest(BaseModel):
    email_text: str
    recipient_name: str | None = "there"
    my_name: str | None = "Ankur"
    tone: str | None = "polite and concise"
    max_summary_tokens: int = 150


class ExtractRequest(BaseModel):
    email_text: str


# -------------------------------------------------
# Draft reply endpoint
# -------------------------------------------------
@app.post("/draft_reply")
def draft_reply(req: DraftRequest):
    """Summarize an email and return a polite draft reply."""

    if summarizer is None:
        return {"error": "Summarizer model not available."}

    try:
        # Step 1: Summarize the input email
        summary = summarizer(
            req.email_text,
            max_length=req.max_summary_tokens,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]

        # Step 2: Create a simple draft reply
        draft = f"""
Hi {req.recipient_name},

Thanks for your email. {summary}

I’ll get back to you shortly with the next steps.

Best regards,  
{req.my_name}
"""
        return {"summary": summary.strip(), "draft_reply": draft.strip()}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


# -------------------------------------------------
# Extraction endpoint
# -------------------------------------------------
@app.post("/extract")
def extract(req: ExtractRequest):
    """
    Simulates extraction of key fields from an email.
    (No heavy LLM used — just heuristic example.)
    """
    text = req.email_text.lower()

    urgency = "medium"
    if "urgent" in text or "asap" in text:
        urgency = "high"
    elif "whenever" in text or "no rush" in text:
        urgency = "low"

    extraction = {
        "intent": "follow-up" if "follow" in text else "general inquiry",
        "urgency": urgency,
        "due_date_guess": None,
        "action_items": [
            "reply to sender",
            "analyze request details",
        ],
    }

    return {"extraction": extraction}


# -------------------------------------------------
# Root endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Server running. Use /docs for API documentation.",
    }
