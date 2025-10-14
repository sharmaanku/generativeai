from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

app = FastAPI(title="Email Triage &amp; Draft API")

# --- Load models once (warm start) ---
# Summarizer (small, efficient)
sum_model = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=sum_model)

# Chatty mini-LLM for drafting
chat_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_id)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_id, torch_dtype=None, device_map="auto")

class DraftRequest(BaseModel):
    email_text: str
    recipient_name: str | None = "there"
    my_name: str | None = "Ankur"
    tone: str | None = "polite and concise"
    max_summary_tokens: int = 180

def draft_reply_prompt(summary, recipient_name, my_name, tone):
    system = (
        "You are a helpful assistant that writes short, professional emails. "
        "Keep replies courteous, direct, and formatted as plain text."
    )
    user = f"""
Write a reply email.

Constraints:
- Tone: {tone}
- Start with a friendly greeting.
- 1–2 short paragraphs, then bullet list of next steps if relevant.
- Close with a courteous sign-off and my name.
- Avoid restating obvious details from the thread.

Context summary of the incoming email:
{summary}

Recipient: {recipient_name}
Sender (me): {my_name}
"""
    # Simple chat template for TinyLlama
    prompt = f"&lt;|system|&gt;\n{system}\n&lt;|user|&gt;\n{user}\n&lt;|assistant|&gt;"
    return prompt

@app.post("/draft_reply")
def draft_reply(req: DraftRequest):
    # 1) Summarize
    sum_out = summarizer(
        req.email_text,
        max_length=req.max_summary_tokens,
        min_length=60,
        do_sample=False
        )[0]["summary_text"]

    # 2) Draft
    prompt = draft_reply_prompt(sum_out, req.recipient_name, req.my_name, req.tone)
    input_ids = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    gen_ids = chat_model.generate(
        **input_ids,
        max_new_tokens=350,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        eos_token_id=chat_tokenizer.eos_token_id,
    )
    full = chat_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # Extract assistant part if template tags exist
    reply = full.split("&lt;|assistant|&gt;")[-1].strip()

    return {
        "summary": sum_out.strip(),
        "draft_reply": reply
    }

class ExtractRequest(BaseModel):
    email_text: str

@app.post("/extract")
def extract(req: ExtractRequest):
    """
    Heuristic extraction with the LLM (no external rules):
    Produces a small JSON with intent, urgency, due date guess, and action items.
    """
    system = "You extract structured fields from emails and answer in pure JSON only."
    user = f"""
Email:
\"\"\"
{req.email_text}
\"\"\"

Return JSON with keys:
intent: short phrase
urgency: one of [low, medium, high]
due_date_guess: ISO date if obvious else null
action_items: array of short strings
"""
    prompt = f"&lt;|system|&gt;\n{system}\n&lt;|user|&gt;\n{user}\n&lt;|assistant|&gt;"
    input_ids = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    gen_ids = chat_model.generate(
        **input_ids,
        max_new_tokens=250,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=chat_tokenizer.eos_token_id,
    )
    full = chat_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    payload = full.split("&lt;|assistant|&gt;")[-1].strip()

    # Best-effort: if model adds extra text, try to trim to JSON braces
    import re, json
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {"extraction": data}
        except Exception:
            pass

    return {"extraction_raw": payload}

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /draft_reply or /extract"}
 