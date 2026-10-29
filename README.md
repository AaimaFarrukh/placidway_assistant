# PlacidWay Website Knowledge Chatbot

RAG chatbot grounded only in 7 PlacidWay URLs. Answers cite sources, say "I don't know" honestly, and resist wrong premises.

## Stack & why
| Part | Choice | Why |
|---|---|---|
| API/UI | FastAPI + single static HTML page | Small, no build step |
| LLM | Groq `llama-3.3-70b-versatile` (answers), `llama-3.1-8b-instant` (follow-up rewrite) | Free, fast, JSON mode |
| Embeddings | Pinecone Inference `multilingual-e5-large` (1024-d) | Free, multilingual (Spanish/Arabic questions over English pages), no extra key |
| Vector DB | Pinecone serverless free tier | As required |
| Hosting | Railway | As required |

## Architecture
1. **Ingest** (`kb.py`): checks robots.txt, custom User-Agent, 1 req/s, strips nav/scripts, keeps headings/lists/table rows, chunks ~220 words w/ 40 overlap, embeds, upserts with `url/title/text` metadata.
2. **Retrieve**: follow-ups rewritten to standalone query (8B model) → embed → top-6. Chunks below similarity 0.70 are dropped; if none remain, the bot answers the fallback **without calling the LLM** (cannot hallucinate).
3. **Generate** (`app.py`): temperature 0, strict system prompt (no outside knowledge, price+currency+context, don't confirm unverified claims, no medical advice, decline booking/off-topic, reply in user's language), JSON output `{answer, sources, found, wants_quote}`.
4. **Guardrails in code**: cited URLs are filtered to those actually retrieved; `found=false` → contact/quote fallback appended; unanswered questions logged to `data/unanswered.jsonl`.
5. **Extras**: multilingual replies, chat history, conditional disclaimer, lead capture (`data/leads.jsonl`), optional Google Sheets lead sync, unanswered-question log.

Groq models are configurable with `GROQ_MAIN_MODEL` and `GROQ_FAST_MODEL`.
The defaults are `openai/gpt-oss-120b` for answers and
`openai/gpt-oss-20b` for follow-up rewriting.

## Setup (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill GROQ_API_KEY, PINECONE_API_KEY, ADMIN_TOKEN
python kb.py             # crawl + index (prints chunks per URL)
uvicorn app:app --reload # http://localhost:8000
python run_tests.py      # writes TEST_RESULTS.md
```

### Authorized crawl override

The crawler respects `robots.txt` by default. If the site owner has explicitly
authorized this assessment to crawl the configured URLs, set:

```text
RESPECT_ROBOTS=false
```

Keep the custom User-Agent and the one-second request delay. If this variable
is omitted or set to `true`, `1`, `yes`, or another non-false value, the
crawler continues to enforce `robots.txt`.

## Google Sheets lead sync

Leads are always written to `data/leads.jsonl`. To also append each lead to a
Google Sheet, configure these environment variables:

```text
GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}
GOOGLE_SHEET_ID=your-spreadsheet-id
GOOGLE_SHEET_TAB=Leads
```

Create a worksheet named `Leads` and add this header row:

```text
Timestamp | Name | Email | Note
```

The Google service account email must have Editor access to the spreadsheet.
The service-account JSON is a secret and must not be committed to source
control. If the Google variables are not configured, the existing local JSONL
lead storage continues to work. If they are configured but synchronization
fails, the lead remains in `data/leads.jsonl` and the API returns an error so
the failed sync is visible.
## Refresh content
`curl -X POST $URL/refresh -H "x-admin-token: $ADMIN_TOKEN"` then `GET /refresh/status`. (Or re-run `python kb.py`.)

## Deploy on Railway
Push to GitHub → Railway "New Project → Deploy from GitHub" → add env vars from `.env.example` → Settings → Networking → Generate Domain. `Procfile` sets the start command. Run `python kb.py` once locally (same Pinecone index) or call `/refresh` after deploy. Keep it up ≥7 days (watch trial credit).
Note: Railway disk is ephemeral, so `data/*.jsonl` logs reset on redeploy (attach a Volume at `/app/data` to persist).

## Known limitations
- Plain `requests` crawl: if a page renders prices via JavaScript, extraction may be empty (ingest reports it). Fix: Playwright fallback.
- Fixed 0.70 similarity threshold needs tuning on real data.
- Groq free-tier rate limits; no auth/rate-limit on `/chat`.
- Tables flattened to `a | b | c` rows; very wide tables may lose context across chunks.

## With another week
Playwright fallback, hybrid BM25+vector search, reranker, eval set with automated pass/fail on the hidden-question styles, streaming responses, rate limiting, persistent DB for logs/leads.
