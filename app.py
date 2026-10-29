import os, json, time, threading, logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
import kb

log = logging.getLogger("app"); logging.basicConfig(level=logging.INFO)
app = FastAPI(title="PlacidWay Knowledge Chatbot")
groq = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MAIN = os.getenv("GROQ_MAIN_MODEL", "openai/gpt-oss-120b")
FAST = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
CONTACT = os.getenv("CONTACT_URL", "https://www.placidway.com")
DATA = Path("data"); DATA.mkdir(exist_ok=True)
MIN_SCORE = 0.70  # below this, nothing relevant was retrieved -> skip the LLM entirely
refresh_state = {"running": False, "last": None, "stats": {}}

SYSTEM = f"""You are the PlacidWay website assistant. Answer ONLY from the CONTEXT passages provided.
Rules:
- Never use outside knowledge. Never invent prices, clinics, treatments, durations or medical facts.
- Prices: always give the number WITH its currency and the page's wording (e.g. "starting from", what the package includes). If the page shows no price, say the price is not listed and suggest requesting a free quote.
- If the user states a fact (e.g. a price) that the context does not confirm or contradicts, do NOT agree; state what the context actually says.
- Compare only if both sides appear in the context; otherwise say what is missing.
- Medical advice/diagnosis/safety for a personal condition: do not give any; say a doctor must assess it and offer the general info on the pages.
- Requests you cannot do (booking, appointments) or off-topic questions (weather etc.): politely decline; for booking/quotes point to PlacidWay's free quote/contact ({CONTACT}).
- Reply in the user's language. Be concise. Add a one-line medical disclaimer only when the question is clinical, not on every answer.
- Set wants_quote=true only if the user shows interest in getting a quote/booking.
Return only one valid JSON object, with no markdown or explanation:
{{"answer": str, "sources": [urls used, copied exactly from context], "found": bool, "wants_quote": bool}}
found=false when the context does not contain the answer (or the request is out of scope)."""

class Msg(BaseModel):
    role: str; content: str
class ChatIn(BaseModel):
    message: str
    history: list[Msg] = []
class LeadIn(BaseModel):
    name: str; email: str; note: str = ""

def log_jsonl(name, obj):
    with open(DATA / name, "a") as f:
        f.write(json.dumps({"ts": time.time(), **obj}, ensure_ascii=False) + "\n")

def append_lead_to_google_sheet(body: LeadIn):
    """Append a lead to Google Sheets when Sheets integration is configured."""
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    worksheet_name = os.getenv("GOOGLE_SHEET_TAB", "Leads")

    if not service_account_json and not sheet_id:
        return False  # Local JSONL remains the storage when Sheets is not enabled.
    if not service_account_json or not sheet_id:
        raise RuntimeError(
            "Google Sheets requires GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEET_ID"
        )

    try:
        credentials_info = json.loads(service_account_json)
    except json.JSONDecodeError as e:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from e

    credentials = Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    worksheet = gspread.authorize(credentials).open_by_key(sheet_id).worksheet(
        worksheet_name
    )
    worksheet.append_row(
        [
            datetime.now(timezone.utc).isoformat(),
            body.name,
            body.email,
            body.note,
        ],
        value_input_option="USER_ENTERED",
    )
    return True

def standalone(message, history):
    """Rewrite follow-ups ('and recovery time?') into a self-contained search query."""
    if not history: return message
    h = "\n".join(f"{m.role}: {m.content[:400]}" for m in history[-6:])
    r = groq.chat.completions.create(model=FAST, temperature=0, max_tokens=80, messages=[
        {"role": "system", "content": "Rewrite the last user message as one standalone search query in English, using the chat for context. Output only the query."},
        {"role": "user", "content": f"{h}\nuser: {message}"}])
    return r.choices[0].message.content.strip() or message

@app.post("/chat")
def chat(body: ChatIn):
    msg = body.message.strip()[:1000]
    if not msg: raise HTTPException(400, "empty message")
    try:
        hits = kb.search(standalone(msg, body.history))
    except Exception as e:
        log.exception("retrieval failed"); raise HTTPException(503, "Knowledge base unavailable") from e
    good = [h for h in hits if h["score"] >= MIN_SCORE]
    fallback = f"I couldn't find that in PlacidWay's pages I have access to. For an accurate answer, please request a free quote or contact PlacidWay: {CONTACT}"
    if not good:
        log_jsonl("unanswered.jsonl", {"q": msg, "reason": "low_retrieval"})
        return {"answer": fallback, "sources": [], "found": False, "wants_quote": True}
    ctx = "\n\n".join(f"[{i+1}] URL: {h['url']}\n{h['text']}" for i, h in enumerate(good))
    msgs = [{"role": "system", "content": SYSTEM}] + \
           [{"role": m.role, "content": m.content[:1500]} for m in body.history[-6:] if m.role in ("user", "assistant")] + \
           [{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {msg}"}]
    try:
        r = groq.chat.completions.create(model=MAIN, temperature=0, max_tokens=1600,
                                         response_format={"type": "json_object"}, messages=msgs)
        out = json.loads(r.choices[0].message.content)
    except Exception as e:
        log.exception("llm failed"); raise HTTPException(503, "Answer service unavailable") from e
    allowed = {h["url"] for h in good}
    sources = [u for u in dict.fromkeys(out.get("sources", [])) if u in allowed]  # drop hallucinated links
    found = bool(out.get("found")) and bool(sources)
    answer = str(out.get("answer", "")).strip() or fallback
    if not found:
        log_jsonl("unanswered.jsonl", {"q": msg, "reason": "llm_not_found"})
        if CONTACT not in answer: answer += f"\n\nFor a free quote or more help, contact PlacidWay: {CONTACT}"
        sources = []
    return {"answer": answer, "sources": sources, "found": found, "wants_quote": bool(out.get("wants_quote"))}

@app.post("/lead")
def lead(body: LeadIn):
    if "@" not in body.email: raise HTTPException(400, "invalid email")
    log_jsonl("leads.jsonl", body.model_dump())
    try:
        google_sheet = append_lead_to_google_sheet(body)
    except Exception as e:
        log.exception("Google Sheets lead sync failed")
        raise HTTPException(
            503,
            "Lead saved locally, but Google Sheets synchronization failed",
        ) from e
    return {"ok": True, "google_sheet": google_sheet}

def _refresh():
    refresh_state["running"] = True
    try: refresh_state["stats"] = kb.ingest()
    except Exception as e: refresh_state["stats"] = {"error": str(e)}
    finally: refresh_state.update(running=False, last=time.time())

@app.post("/refresh")
def refresh(x_admin_token: str = Header("")):
    if x_admin_token != os.getenv("ADMIN_TOKEN", "change-me") or not os.getenv("ADMIN_TOKEN"):
        raise HTTPException(401, "bad token")
    if refresh_state["running"]: return {"status": "already running"}
    threading.Thread(target=_refresh, daemon=True).start()
    return {"status": "started"}

@app.get("/refresh/status")
def refresh_status(): return refresh_state

@app.get("/health")
def health(): return {"ok": True}

@app.get("/")
def home(): return FileResponse("static/index.html")
