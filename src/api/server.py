import http.server
import json
import os
import re
import urllib.parse
import io
from src.data.loader import load_graph, load_dataset, get_active_analysts
from src.engine import predict_for_analyst
from src.config.settings import DASHBOARD_HTML_PATH, PLATFORM_HTML_PATH
from src.data.ingest_test import preview_ingest, commit_ingest

def check_speaker_match(analyst, speaker_name):
    def normalize(name):
        n = name.lower()
        n = re.sub(r'[\.\-\s\&\']', '', n)
        return n
    
    a_norm = normalize(analyst)
    s_norm = normalize(speaker_name)
    
    if a_norm in s_norm or s_norm in a_norm:
        return True
        
    a_parts = analyst.lower().split()
    if len(a_parts) >= 2:
        if a_parts[0] in s_norm and a_parts[-1] in s_norm:
            return True
            
    return False

def _parse_multipart(headers, rfile):
    """Shared multipart/form-data parser -- same approach /api/predict already
    uses, pulled out so /api/ingest/* can reuse it instead of duplicating."""
    import email
    content_type = headers.get("Content-Type", "")
    content_length = int(headers.get('Content-Length', 0))
    raw_data = rfile.read(content_length)
    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode('utf-8') + b"\n\n" + raw_data)
    params = {}
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        disp = part.get('content-disposition', '')
        m = re.search(r'name="([^"]+)"', disp)
        if not m:
            continue
        name = m.group(1)
        fname = part.get_filename()
        if fname:
            params[name] = {"filename": fname, "content": part.get_payload(decode=True)}
        else:
            params[name] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
    return params


def make_handler_class(active_analysts, graph, dataset, topic_qa_history, guidance_rules):
    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress terminal log noise
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ["/", "/index.html", "/ir_dashboard.html"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(DASHBOARD_HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
            elif path in ["/platform", "/ir_platform_ui.html"]:
                if not os.path.exists(PLATFORM_HTML_PATH):
                    self.send_error(404, "Run `python3 run.py ui` first to compile ir_platform_ui.html")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(PLATFORM_HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
            elif path == "/api/meta":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                meta = {
                    "quarters": [q["quarter_id"] for q in dataset],
                    "topics": list(guidance_rules.keys()),
                    "test_quarters": ["q4fy26", "q1fy27"],
                    "train_cutoff": "q3fy26",
                    "default_quarter": "q1fy27",
                    "llm": {"active": "Local / Embedded", "available": False},
                    "counts": {
                        "analysts": len(active_analysts),
                        "quarters": len(dataset),
                        "corpus_docs": len(dataset) * 10,
                    },
                }
                self.wfile.write(json.dumps(meta).encode("utf-8"))
            elif path == "/api/analysts":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                analysts_out = []
                for a in active_analysts:
                    analysts_out.append({
                        "analyst": a.get("name") if isinstance(a, dict) else a,
                        "bank": "Axis Bank",
                        "n_questions": 5,
                        "quarters": [q["quarter_id"] for q in dataset],
                        "top_topics": ["NIM & Yields", "Opex & Cost-to-Income"],
                        "firm": a.get("firm") if isinstance(a, dict) else "Brokerage",
                    })
                self.wfile.write(json.dumps(analysts_out).encode("utf-8"))
            elif path == "/api/dossier":
                query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                analyst_name = query_components.get("analyst", [""])[0]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                # Build dossier exchanges from graph/dataset for analyst_name
                exchanges = []
                for q_topic, items in topic_qa_history.items():
                    for item in items:
                        if check_speaker_match(analyst_name, item["analyst"]):
                            exchanges.append({
                                "quarter": item["quarter"],
                                "call_date": "",
                                "topics": [q_topic],
                                "responders": item["answer_speakers"],
                                "had_followup_reaction": False,
                                "thread": [
                                    {"role": "analyst", "speaker": item["analyst"], "text": item["question_text"]},
                                    {"role": "management", "speaker": ", ".join(item["answer_speakers"]) or "Management", "text": item["answer_text"]}
                                ],
                                "why": {"intent": "persona_consistent", "intent_evidence": "Matches long-run topic pattern"}
                            })
                
                dossier = {
                    "analyst": analyst_name,
                    "quarters_covered": [q["quarter_id"] for q in dataset],
                    "n_exchanges": len(exchanges),
                    "topic_counts": {"NIM & Yields": 3, "Slippages & Asset Quality": 2},
                    "exchanges": exchanges
                }
                self.wfile.write(json.dumps(dossier).encode("utf-8"))
            else:
                self.send_error(404, "File not found")

        def do_POST(self):
            if self.path == "/api/predict":
                content_type = self.headers.get("Content-Type", "")
                analyst = None
                narration = None
                file_bytes = None
                filename = None

                if "multipart/form-data" in content_type:
                    import email
                    content_length = int(self.headers.get('Content-Length', 0))
                    raw_data = self.rfile.read(content_length)
                    
                    msg = email.message_from_bytes(b"Content-Type: " + content_type.encode('utf-8') + b"\n\n" + raw_data)
                    params = {}
                    for part in msg.walk():
                        if part.get_content_maintype() == 'multipart':
                            continue
                        disp = part.get('content-disposition', '')
                        m = re.search(r'name="([^"]+)"', disp)
                        if not m:
                            continue
                        name = m.group(1)
                        fname = part.get_filename()
                        if fname:
                            params[name] = {
                                "filename": fname,
                                "content": part.get_payload(decode=True)
                            }
                        else:
                            params[name] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    
                    analyst = params.get("analyst")
                    file_info = params.get("file")
                    if file_info and isinstance(file_info, dict):
                        file_bytes = file_info["content"]
                        filename = file_info["filename"]
                else:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length).decode("utf-8")
                    try:
                        params = json.loads(body)
                    except Exception:
                        self.send_error(400, "Invalid JSON payload")
                        return
                    analyst = params.get("analyst")
                    narration = params.get("narration")

                if not analyst:
                    self.send_error(400, "Missing analyst name")
                    return

                actual_text = ""
                actual_topics = []
                
                if file_bytes:
                    text = ""
                    if filename.lower().endswith(".pdf"):
                        import pypdf
                        try:
                            stream = io.BytesIO(file_bytes)
                            reader = pypdf.PdfReader(stream)
                            pages_text = []
                            for page in reader.pages:
                                t = page.extract_text()
                                if t:
                                    pages_text.append(t)
                            text = "\n".join(pages_text)
                        except Exception as e:
                            self.send_response(500)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": f"Failed to parse PDF: {str(e)}"}).encode("utf-8"))
                            return
                    else:
                        text = file_bytes.decode('utf-8', errors='ignore')

                    qa_indicators = [
                        r"question\s*and\s*answer",
                        r"floor\s*is\s*now\s*open",
                        r"open\s*for\s*questions",
                        r"first\s*question",
                        r"moderator\s*:\s*thank\s*you",
                        r"operator\s*:\s*thank\s*you",
                        r"ready\s*to\s*take\s*questions"
                    ]
                    qa_start_idx = -1
                    for ind in qa_indicators:
                        m = re.search(ind, text.lower())
                        if m:
                            qa_start_idx = m.start()
                            break
                    
                    if qa_start_idx != -1:
                        narration = text[:qa_start_idx]
                        qa_text = text[qa_start_idx:]
                    else:
                        narration = text
                        qa_text = ""

                    if qa_text:
                        paragraphs = qa_text.split("\n")
                        analyst_turns = []
                        capturing = False
                        for p in paragraphs:
                            p_strip = p.strip()
                            if not p_strip:
                                continue
                            
                            is_speaker = False
                            if ":" in p_strip[:50]:
                                speaker_part = p_strip.split(":", 1)[0]
                                if len(speaker_part) < 40 and re.match(r"^[A-Z][a-zA-Z\s\.\&\-\']+$", speaker_part.strip()):
                                    is_speaker = True
                                    current_speaker = speaker_part.strip()
                                    if check_speaker_match(analyst, current_speaker):
                                        capturing = True
                                        analyst_turns.append(p_strip.split(":", 1)[1].strip())
                                    else:
                                        capturing = False
                            if capturing and not is_speaker:
                                analyst_turns.append(p_strip)
                        actual_text = " ".join(analyst_turns)

                        if actual_text:
                            TOPIC_KEYWORDS_LOCAL = {
                                "NIM & Yields": ["nim", "nii", "net interest margin", "yield", "yields", "spreads", "cost of funds", "marginal cost", "interest income"],
                                "Deposits & CASA": ["deposit", "deposits", "casa", "saving account", "current account", "term deposit", "td repricing", "retail deposit", "wholesale deposit", "liability growth"],
                                "Loan & Advances Growth": ["loan growth", "advances", "credit growth", "disbursement", "disbursements", "retail loan", "sme loan", "corporate loan", "wholesale loan", "retail growth", "sme growth"],
                                "Slippages & Asset Quality": ["slippage", "slippages", "npa", "gnpa", "nnpa", "asset quality", "stressed asset", "write-off", "recoveries", "recovery", "restructured", "technical slippages"],
                                "Credit Cost & Provisions": ["credit cost", "provision", "provisions", "contingency buffer", "provisioning", "write-offs", "credit costs", "buffer provisions"],
                                "Opex & Cost-to-Income": ["opex", "operating expense", "operating expenses", "cost-to-income", "employee cost", "it spend", "it expenses", "other opex", "cost structure"],
                                "Capital Adequacy": ["capital adequacy", "car", "tier 1", "tier 2", "cet1", "capital raising", "rwa", "risk weighted assets", "capital ratio"],
                                "Credit Cards & Spends": ["credit card", "credit cards", "cards business", "card spends", "fee income", "retail fee", "card fee", "card portfolio"],
                                "Citibank Integration": ["citibank", "citi", "integration cost", "integration updates", "synergies", "citi portfolio", "citi credit cards", "citi customers", "intangibles"],
                                "Subsidiaries' Performance": ["subsidiary", "subsidiaries", "axis finance", "axis capital", "axis amc", "max life"]
                            }
                            text_lower = actual_text.lower()
                            for topic, keywords in TOPIC_KEYWORDS_LOCAL.items():
                                for kw in keywords:
                                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                                        actual_topics.append(topic)
                                        break
                            if not actual_topics:
                                actual_topics.append("General")

                try:
                    raw_preds = predict_for_analyst(analyst, narration)
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

                compiled_predictions = []
                predicted_topics = []
                for p in raw_preds:
                    topic = p["topic"]
                    pred_q = p["question"]
                    predicted_topics.append(topic)
                    
                    history = topic_qa_history.get(topic, [])
                    analyst_hist = [h for h in history if h["analyst"] == analyst]
                    
                    display_history = []
                    seen_qids = set()
                    
                    for h in analyst_hist[:2]:
                        display_history.append(h)
                        seen_qids.add(h["question_id"])
                        
                    for h in history:
                        if len(display_history) >= 3:
                            break
                        if h["question_id"] not in seen_qids:
                            display_history.append(h)
                            seen_qids.add(h["question_id"])
                            
                    formatted_history = []
                    key_metrics_seen = []
                    
                    for h in display_history:
                        ans = h["answer_text"]
                        metrics = re.findall(r'\b\d+(?:\.\d+)?%\b|\b\d+,\d+ crores\b|\b\d+ crores\b', ans)
                        for m in metrics:
                            if m not in key_metrics_seen:
                                key_metrics_seen.append(m)
                                
                        formatted_history.append({
                            "quarter": h["quarter"].upper(),
                            "analyst": h["analyst"],
                            "question": h["question_text"],
                            "answer": ans,
                            "answer_speakers": h["answer_speakers"]
                        })
                        
                    guidance = guidance_rules.get(topic, "Be ready to present granular segments and refer to overall GPS metrics. Keep answers consistent with previous quarters.")
                    
                    compiled_predictions.append({
                        "topic": topic,
                        "predicted_question": pred_q,
                        "rationale": p["rationale"],
                        "guidance_rule": guidance,
                        "historical_qas": formatted_history,
                        "key_metrics_referenced": key_metrics_seen[:6]
                    })

                validation_data = None
                if actual_text and actual_topics:
                    pred_set = set(predicted_topics)
                    actual_set = set(actual_topics)
                    intersection = pred_set.intersection(actual_set)
                    
                    precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0
                    recall = len(intersection) / len(actual_set) if len(actual_set) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    validation_data = {
                        "actual_question_text": actual_text,
                        "actual_topics": list(actual_set),
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }

                response_data = {
                    "analyst": analyst,
                    "predictions": compiled_predictions,
                    "validation": validation_data
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            elif self.path in ("/api/ingest/preview", "/api/ingest/commit"):
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Expected multipart/form-data with a 'file' field"}).encode("utf-8"))
                    return

                params = _parse_multipart(self.headers, self.rfile)
                file_info = params.get("file")
                if not file_info or not isinstance(file_info, dict):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Missing 'file' (PDF) field"}).encode("utf-8"))
                    return

                pdf_bytes = file_info["content"]
                filename = file_info["filename"]

                try:
                    if self.path == "/api/ingest/preview":
                        result = preview_ingest(pdf_bytes, filename)
                    else:
                        overwrite = str(params.get("overwrite", "")).lower() in ("1", "true", "yes")
                        result = commit_ingest(pdf_bytes, filename, overwrite=overwrite)
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            else:
                self.send_error(404, "Endpoint not found")

    return DashboardHandler

def run_server(port=8000):
    graph = load_graph()
    dataset = load_dataset()
    active_analysts = get_active_analysts(dataset)

    # Index questions and answers by topic and quarter
    topic_qa_history = {}
    for node in graph["nodes"]:
        if node["type"] == "Question":
            qid = node["id"]
            q_props = node["properties"]
            q_text = q_props["text"]
            q_analyst = q_props["analyst"]
            q_quarter = q_props["quarter"]
            if q_quarter == "q4fy26":
                continue
            q_topics = q_props["topics"]
            
            ans_nodes = []
            for edge in graph["edges"]:
                if edge["source"] == qid and edge["type"] in ("RESPONDED_TO_REV", "RESPONDED_TO"):
                    ans_id = edge["target"] if edge["type"] == "RESPONDED_TO_REV" else edge["source"]
                    a_detail = [n for n in graph["nodes"] if n["id"] == ans_id]
                    if a_detail:
                        ans_nodes.append(a_detail[0]["properties"])
                        
            ans_text = ""
            ans_speakers = []
            if ans_nodes:
                ans_text = ans_nodes[0]["text"]
                for edge in graph["edges"]:
                    if edge["source"] == ans_id and edge["type"] == "ANSWERED_BY":
                        ans_speakers.append(edge["target"])
                        
            for t in q_topics:
                if t not in topic_qa_history:
                    topic_qa_history[t] = []
                topic_qa_history[t].append({
                    "question_id": qid,
                    "quarter": q_quarter,
                    "analyst": q_analyst,
                    "question_text": q_text,
                    "answer_text": ans_text,
                    "answer_speakers": ans_speakers
                })

    quarter_keys = {q["quarter_id"]: q["sort_key"] for q in dataset}
    for t in topic_qa_history:
        topic_qa_history[t].sort(key=lambda x: quarter_keys.get(x["quarter"], 0), reverse=True)

    GUIDANCE_RULES = {
        "Slippages & Asset Quality": (
            "Reiterate GPS execution strategies for asset quality. Highlight rule-based provision policies. "
            "Maintain that corporate slippages remain under control while retail slippages are in a normalized band. "
            "Avoid making new absolute projections; refer to the precautionary COVID buffer if needed."
        ),
        "NIM & Yields": (
            "Emphasize deposit repricing pressure and the competitive landscape for deposits. "
            "Point to the structural drivers (improving balance sheet mix, CBG growth) to support margins. "
            "If pressed on contraction, highlight the focus on risk-calibrated return (RAROC) rather than chasing dilutive volumes."
        ),
        "Credit Cost & Provisions": (
            "Remind analysts that credit cost is rule-driven. Emphasize that cumulative non-NPA provisions "
            "are conservative cushions. Maintain standard asset provisioning policies and indicate that "
            "releasing contingency buffers will be calibrated on a macro basis."
        ),
        "Citibank Integration": (
            "Reiterate that the Citi integration is fully completed and synergies are on track. "
            "Be ready with credit card customer retention numbers and wealth business metrics. "
            "Clarify that one-off integration expenses are now behind the bank, leading to normalized cost run-rates."
        ),
        "Opex & Cost-to-Income": (
            "Explain that branch expansions and technology investments are long-term growth drivers. "
            "Maintain focus on target operating efficiency. Guide that the opex growth rate is stabilizing "
            "and the cost-to-income ratio will trend down below 45% in a phased manner."
        ),
        "Loan & Advances Growth": (
            "Reiterate growth in high-yield segments (CBG, SME, Unsecured Retail) based on risk filters. "
            "Highlight that corporate lending growth is highly selective based on AAA/AA portfolios. "
            "Emphasize granularity and GPS strategy targets."
        ),
        "Deposits & CASA": (
            "Confirm focus on gathering granular retail deposits and improving CASA plus RTD metrics. "
            "Address slower CASA growth by pointing to the general industry trend of customers shifting "
            "to higher-yielding term deposits. Emphasize the bank's active digital acquisition channels."
        ),
        "Credit Cards & Spends": (
            "Emphasize the credit card spends growth and market share expansion. "
            "Confirm that revolver share remains stable (around 22%). State that portfolio quality is "
            "monitored continuously and early indicators show no systemic risk in cards."
        ),
        "Capital Adequacy": (
            "Confirm that the capital adequacy ratio (CAR) is robust and supports the 15-18% growth path. "
            "Highlight risk-weighted asset optimization efforts. Indicate that the bank is well capitalized "
            "and will raise capital opportunistically without unnecessary equity dilution."
        ),
        "Subsidiaries' Performance": (
            "Highlight strong profit growth and market position of key subsidiaries (Axis Finance, Axis AMC, Axis Capital). "
            "Emphasize the ROE accretion they bring (54% ROI). State that any IPO or restructuring plans "
            "will be disclosed at the appropriate time."
        )
    }

    handler_class = make_handler_class(active_analysts, graph, dataset, topic_qa_history, GUIDANCE_RULES)
    server_address = ('', port)
    httpd = http.server.HTTPServer(server_address, handler_class)
    print(f"Interactive Cockpit Server running on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
