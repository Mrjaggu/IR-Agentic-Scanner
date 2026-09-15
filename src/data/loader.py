import json
import os
import re
from src.config.settings import (
    DATASET_PATH,
    GRAPH_PATH,
    ANALYST_ALIASES,
    VAL_QUARTER,
    PREP_SHEET_PATH,
    PREDICTIONS_PATH,
    PERSONA_DERIVED_PATH,
)

def load_dataset(canonicalize=True, dataset_path=None):
    # dataset_path: optional override so a caller can load a SPECIFIC bank's
    # dataset.json (e.g. graphs/compiler.py compiling a non-default bank)
    # without this function itself needing full bank_id threading yet --
    # that's a later phase (see src/config/settings.py::paths_for). Defaults
    # to the existing DATASET_PATH shim (currently axis) so every unmodified
    # caller behaves exactly as before.
    dataset_path = dataset_path or DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset.json not found at {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    dataset.sort(key=lambda x: x["sort_key"])
    
    if canonicalize:
        for q in dataset:
            for a in q.get("analysts", []):
                if a["name"] in ANALYST_ALIASES:
                    a["name"] = ANALYST_ALIASES[a["name"]]
    return dataset

def load_graph(canonicalize=True):
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"graph.json not found at {GRAPH_PATH}")
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)
        
    if canonicalize:
        for n in graph["nodes"]:
            if n["type"] == "Question" and n["properties"].get("analyst") in ANALYST_ALIASES:
                n["properties"]["analyst"] = ANALYST_ALIASES[n["properties"]["analyst"]]
    return graph

def get_active_analysts(dataset, val_quarter=VAL_QUARTER):
    # Active analysts — anyone who appeared in the last 6 training quarters
    _RECENT_WINDOW = ["q2fy25", "q3fy25", "q4fy25", "q1fy26", "q2fy26", "q3fy26"]
    active = set()
    for q in dataset:
        if q["quarter_id"] in _RECENT_WINDOW:
            for a in q["analysts"]:
                if a["name"] not in ("Moderator", "Operator"):
                    active.add(a["name"])
                    
    # Also include validation quarter participants so we can validate them
    q4_data = [q for q in dataset if q["quarter_id"] == val_quarter]
    if q4_data:
        for a in q4_data[0]["analysts"]:
            if a["name"] not in ("Moderator", "Operator"):
                active.add(a["name"])
                
    return sorted(list(active))

def generate_prep_sheet():
    print("Compiling preparation sheet from predictions and historical graph...")
    
    with open(PREDICTIONS_PATH, "r") as f:
        pred_data = json.load(f)
        
    graph = load_graph()
    dataset = load_dataset()
    
    predictions = pred_data["predictions"]
    
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
            
            # find corresponding answer
            ans_nodes = []
            ans_id = None
            for edge in graph["edges"]:
                if edge["target"] == qid and edge["type"] == "RESPONDED_TO":
                    ans_id = edge["source"]
                    a_detail = [n for n in graph["nodes"] if n["id"] == ans_id]
                    if a_detail:
                        ans_nodes.append(a_detail[0]["properties"])
                        
            # Backwards compatibility check: try RESPONDED_TO_REV if RESPONDED_TO is empty
            if not ans_nodes:
                for edge in graph["edges"]:
                    if edge["source"] == qid and edge["type"] == "RESPONDED_TO_REV":
                        ans_id = edge["target"]
                        a_detail = [n for n in graph["nodes"] if n["id"] == ans_id]
                        if a_detail:
                            ans_nodes.append(a_detail[0]["properties"])
                        
            ans_text = ""
            ans_speakers = []
            if ans_nodes:
                ans_text = ans_nodes[0]["text"]
                # find who answered
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
            "Reiterate GP S execution strategies for asset quality. Highlight rule-based provision policies. "
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

    persona_derived = {}
    if os.path.exists(PERSONA_DERIVED_PATH):
        with open(PERSONA_DERIVED_PATH) as f:
            persona_derived = json.load(f)

    prep_sheet = {}
    for analyst, items in predictions.items():
        style = persona_derived.get(analyst)
        prep_sheet[analyst] = {
            "analyst_style": style["style_note"] if style else None,
            "predictions": [],
        }
        for p in items:
            topic = p["topic"]
            q_pred = p["question"]
            rationale = p["rationale"]
            
            history = topic_qa_history.get(topic, [])
            rule = GUIDANCE_RULES.get(topic, "Follow standard bank guidance parameters for this topic.")
            
            # Find relevant historical Q&A for this analyst
            relevant_qa = []
            for item in history:
                if item["analyst"] == analyst:
                    relevant_qa.append(item)
                    
            # Fallback to other analysts' recent Q&A for this topic
            peers_qa = []
            if len(relevant_qa) < 3:
                for item in history:
                    if item["analyst"] != analyst:
                        peers_qa.append(item)
                        if len(relevant_qa) + len(peers_qa) >= 5:
                            break
                            
            prep_sheet[analyst]["predictions"].append({
                "topic": topic,
                "predicted_question": q_pred,
                "rationale": rationale,
                "guidance_rule": rule,
                "analyst_history": relevant_qa[:3],
                "peer_history": peers_qa[:2]
            })
            
    with open(PREP_SHEET_PATH, "w") as f:
        json.dump(prep_sheet, f, indent=2)
        
    print(f"IR Prep Sheet compiled successfully!\nFile saved to {PREP_SHEET_PATH}")
