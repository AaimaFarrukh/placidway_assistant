"""Ingestion + retrieval: crawl the 7 URLs, chunk, embed (Pinecone inference), upsert/query."""
import os, re, time, hashlib, logging
import requests
from dotenv import load_dotenv
from urllib import robotparser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from urls import URLS

load_dotenv(dotenv_path=".env", override=True)
log = logging.getLogger("kb")
UA = "PlacidWayAssessmentBot/1.0 (candidate assessment; respects robots.txt; ~1 req/s)"
EMBED_MODEL, DIM = "multilingual-e5-large", 1024  # multilingual -> ES/AR questions work
CHUNK_WORDS, OVERLAP = 220, 40
RESPECT_ROBOTS = os.getenv("RESPECT_ROBOTS", "true").strip().lower() not in {
    "0", "false", "no"
}

_pc = None
def index():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.getenv("PINECONE_INDEX", "placidway-kb")
    if name not in _pc.list_indexes().names():
        _pc.create_index(name=name, dimension=DIM, metric="cosine",
                         spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        while not _pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return _pc.Index(name)

def embed(texts, kind):
    out = []
    for i in range(0, len(texts), 32):
        r = _pc.inference.embed(model=EMBED_MODEL, inputs=texts[i:i+32],
                                parameters={"input_type": kind, "truncate": "END"})
        out += [d["values"] for d in r]
    return out

def robots_ok(url):
    if not RESPECT_ROBOTS:
        return True
    p = urlparse(url)
    rp = robotparser.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        return True  # robots unreachable -> proceed politely

def fetch(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
        t.decompose()
    title = (soup.title.string or url).strip() if soup.title else url
    root = soup.find("main") or soup.body or soup
    lines = []
    for el in root.find_all(["h1","h2","h3","h4","p","li","td","th","tr","dt","dd"]):
        if el.name in ("td","th") : continue  # handled via tr
        txt = " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["td","th"])) if el.name=="tr" \
              else el.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) > 2 and (not lines or lines[-1] != txt):
            lines.append(("## " if el.name[0]=="h" else "") + txt)
    return title, "\n".join(lines)

def chunk(text):
    words, chunks, step = text.split(), [], CHUNK_WORDS - OVERLAP
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i+CHUNK_WORDS])
        if len(piece) > 40: chunks.append(piece)
        if i + CHUNK_WORDS >= len(words): break
    return chunks

def ingest():
    idx = index()
    stats = {}
    pending = []
    for url in URLS:
        time.sleep(1)  # ~1 request/second
        if not robots_ok(url):
            stats[url] = "blocked by robots.txt"; continue
        try:
            title, text = fetch(url)
        except Exception as e:
            stats[url] = f"error: {e}"; log.error("fetch %s: %s", url, e); continue
        chunks = chunk(text)
        if not chunks:
            stats[url] = "no text extracted (page may be JS-rendered)"; continue
        pending.append((url, title, chunks))
        stats[url] = f"{len(chunks)} chunks"
    if not pending:
        log.error("ingestion produced no content; preserving existing index")
        return stats
    index_stats = idx.describe_index_stats()
    namespaces = getattr(index_stats, "namespaces", None)
    if namespaces is None:
        namespaces = index_stats.get("namespaces", {})
    namespace_stats = namespaces.get("", namespaces.get("__default__"))
    vector_count = (
        getattr(namespace_stats, "vector_count", 0)
        if namespace_stats is not None
        else 0
    )
    if vector_count:
        idx.delete(delete_all=True)
    for url, title, chunks in pending:
        vecs = embed([f"{title}\n{c}" for c in chunks], "passage")
        idx.upsert(vectors=[
            {
                "id": hashlib.md5(f"{url}{i}".encode()).hexdigest(),
                "values": v,
                "metadata": {"url": url, "title": title, "text": c},
            }
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ])
    return stats

def search(query, k=6):
    idx = index()
    v = embed([query], "query")[0]
    res = idx.query(vector=v, top_k=k, include_metadata=True)
    return [{"score": m.score, **m.metadata} for m in res.matches]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for u, s in ingest().items(): print(s, "<-", u)
