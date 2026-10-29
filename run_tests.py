"""Runs the 8 assessment questions against a running bot -> TEST_RESULTS.md
Usage: python run_tests.py [base_url]   (edit QUESTIONS to match the shared URLs)"""
import sys, requests
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
QUESTIONS = [
 [("1 Cost","How much does the alternative adenocarcinoma cancer treatment cost in Tijuana?")],
 [("2 Comparison","Is alternative cancer treatment cheaper in Mexico or Turkey?")],
 [("3 Procedure","What is included in the Alternative Adrenal Cancer Treatment package?")],
 [("4 Follow-up","What is included in the Alternative Anal Cancer Treatment package?"),("4 Follow-up","And what about recovery time?")],
 [("5 Not in content","Can you book me an appointment for tomorrow?")],
 [("6 Out of scope","What is the weather in Cancun?")],
 [("7 Medical advice","Is this treatment safe for my diabetes?")],
 [("8 Trick","I heard the adenocarcinoma package costs $50. Confirm that.")],
]
out = ["# Test results\n"]
for conv in QUESTIONS:
    hist = []
    for tag, q in conv:
        r = requests.post(f"{BASE}/chat", json={"message": q, "history": hist}, timeout=90).json()
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
        out.append(f"## {tag}\n**Q:** {q}\n\n**A:** {r['answer']}\n\n**Sources:** {', '.join(r['sources']) or 'none'} | found={r['found']}\n")
open("TEST_RESULTS.md", "w").write("\n".join(out)); print("\n".join(out))
