from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re

# Force CPU mode
device = torch.device("cpu")

print("✅ Loading summarizer model (facebook/bart-large-cnn)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

print("✅ Loading chat model (microsoft/phi-2)... This may take a few minutes on first run.")
chat_model_id = "microsoft/phi-2"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_id)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_id, torch_dtype=torch.float32)
chat_model.to(device)


def summarize_email(email_text: str, max_tokens: int = 150):
    """Summarize input email text."""
    print("\n🧠 Summarizing email...")
    result = summarizer(
        email_text,
        max_length=max_tokens,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]
    print("✅ Summary complete.")
    return result.strip()


def build_draft_prompt(summary: str, recipient_name: str, my_name: str, tone: str) -> str:
    """Construct prompt for reply drafting."""
    return f"""
You are a helpful assistant that writes professional email replies.

Context Summary:
"{summary}"

Guidelines:
- Tone: {tone}
- Start with a greeting.
- Write 1–2 concise paragraphs.
- Be clear and professional.
- Close politely with "{my_name}".

Recipient: {recipient_name}
"""


def draft_email_reply(email_text: str, recipient_name="there", my_name="Ankur", tone="polite"):
    """Generate draft reply email."""
    summary = summarize_email(email_text)
    prompt = build_draft_prompt(summary, recipient_name, my_name, tone)

    print("\n✍️ Generating reply draft...")
    input_ids = chat_tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = chat_model.generate(
        **input_ids,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=chat_tokenizer.eos_token_id
    )

    reply = chat_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    reply = reply[len(prompt):].strip()

    print("✅ Reply generated.")
    return {"summary": summary, "draft_reply": reply}


def extract_email_metadata(email_text: str):
    """Extract structured info like intent, urgency, due date, and action items."""
    prompt = f"""
You are an assistant that extracts structured info from emails.

Email:
\"\"\"{email_text}\"\"\"

Return JSON:
{{
  "intent": "short phrase",
  "urgency": "low|medium|high",
  "due_date_guess": "YYYY-MM-DD or null",
  "action_items": ["task1", "task2"]
}}
"""

    print("\n🔍 Extracting metadata...")
    input_ids = chat_tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = chat_model.generate(
        **input_ids,
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        pad_token_id=chat_tokenizer.eos_token_id
    )

    payload = chat_tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            print("✅ Metadata extraction complete.")
            return data
        except Exception:
            pass

    print("⚠️ Could not parse structured JSON. Returning raw output.")
    return {"raw_extraction": payload}


if __name__ == "__main__":
    email_text = """
    Hi Ankur, Can you send the Q3 numbers to Tushar Sir by today 3:00 PM?
    We also need a brief explanation of the variance against plan.
    If not possible, please be ready with the detailed explanation.
    Thanks, Privacy Org
    """

    print("\n===============================")
    print("📧 EMAIL ASSISTANT STARTED")
    print("===============================")

    result = draft_email_reply(
        email_text=email_text,
        recipient_name="Tushar",
        my_name="Ankur",
        tone="Harsh and Panic"
    )

    print("\n--- 📨 DRAFT REPLY ---")
    print(json.dumps(result, indent=2))

    extracted = extract_email_metadata(email_text)
    print("\n---  EXTRACTED METADATA ---")
    print(json.dumps(extracted, indent=2))

    print("\n✅ Done.")
 
