import json
import os
import re
from src.config.settings import ANALYST_ALIASES, GRAPH_PATH, paths_for
from src.config.banks import DEFAULT_BANK, get_bank
from src.data.loader import load_dataset

# Thematic topics and keywords
TOPICS = {
    "Slippages & Asset Quality": [
        "slippage", "npa", "gnpa", "nnpa", "provision coverage", "restructured", "write-off", "write off",
        "stressed asset", "delinquency", "delinquencies", "bad loan", "sma", "recovery", "recoveries",
        "restructure", "stress book", "credit quality"
    ],
    "NIM & Yields": [
        "nim", "margin", "margins", "yield", "yields", "cost of fund", "cost of funds", "cost of deposit",
        "cost of deposits", "interest income", "net interest income", "pricing", "liability pricing",
        "repricing", "funding cost", "cost of liabilities"
    ],
    "Credit Cost & Provisions": [
        "credit cost", "credit costs", "provision", "provisions", "provisioning", "contingency provision",
        "contingency provisions", "buffer provision", "standard asset provision", "pcr", "covid provision",
        "write-off cost", "provisions write back", "write-backs"
    ],
    "Citibank Integration": [
        "citibank", "citi", "acquisition", "integration", "synergies", "integration cost", "goodwill",
        "citi book", "citi customer", "merger cost"
    ],
    "Opex & Cost-to-Income": [
        "opex", "operating expense", "operating expenses", "cost to income", "cost-to-income",
        "employee cost", "employee expenses", "branch expansion", "tech spend", "operating cost", "operating costs"
    ],
    "Loan & Advances Growth": [
        "loan growth", "advances growth", "credit growth", "retail lending", "wholesale lending",
        "corporate loan", "corporate lending", "cbg", "sme growth", "unsecured loan", "unsecured lending",
        "personal loan", "home loan", "loan book"
    ],
    "Deposits & CASA": [
        "deposit", "deposits", "casa", "current account", "savings account", "term deposit", "term deposits",
        "retail deposit", "retail deposits", "casa ratio", "deposit growth", "liability franchise", "saving deposit"
    ],
    "Credit Cards & Spends": [
        "credit card", "credit cards", "cards", "card spends", "revolver", "revolvers", "revolver share",
        "spends", "fee income", "interchange", "card acquisition", "card book"
    ],
    "Capital Adequacy": [
        "capital adequacy", "tier 1", "tier-1", "car", "risk weighted assets", "capital raising", "rwa",
        "capital ratio", "capital ratios", "tier one"
    ],
    "Subsidiaries' Performance": [
        # Generic vocabulary only -- a bank's OWN subsidiary names (e.g. Axis's
        # "axis amc"/"max life") come from src.config.banks.BankConfig.subsidiary_keywords
        # and get merged in by classify_topics(), so each bank is graded on its
        # own subsidiaries rather than everyone being graded on Axis's.
        "subsidiary", "subsidiaries", "subsidiary profit", "subsidiary performance"
    ],
    "Profitability & Returns": [
        "roe", "roa", "raroc", "rarocs", "return on equity", "return on assets", "return on asset",
        "return ratio", "return ratios", "profitability", "roce", "exit roe", "return profile",
        "consolidated roe", "sustainable roe"
    ],
    "Strategy & Competitive Positioning": [
        "market share", "competitive", "competition", "peer banks", "versus peers", "vs peers",
        "compared to peers", "relative to peers", "faster than peers", "than your peers", "than peers",
        "gaining share", "gain share", "positioning", "gps strategy", "faster than the industry",
        "faster than industry", "industry growth", "franchise build"
    ]
}

def classify_topics(text, subsidiary_keywords=None):
    if not text:
        return []
    text_lower = text.lower()
    matched = []
    for topic, keywords in TOPICS.items():
        if topic == "Subsidiaries' Performance" and subsidiary_keywords:
            keywords = keywords + subsidiary_keywords
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                matched.append(topic)
                break
    return matched

def main(bank_id: str = DEFAULT_BANK):
    bank = get_bank(bank_id)
    paths = paths_for(bank_id)
    graph_path = paths.graph_path
    subsidiary_keywords = bank.subsidiary_keywords

    print(f"Compiling document graph.json from dataset.json for {bank.display_name}...")
    dataset = load_dataset(dataset_path=paths.dataset_path, canonicalize=False) # Load raw, compiler handles canonicalization

    nodes = []
    edges = []
    node_ids = set()

    def add_node(node_id, node_type, properties):
        if node_id not in node_ids:
            nodes.append({
                "id": node_id,
                "type": node_type,
                "properties": properties
            })
            node_ids.add(node_id)

    def add_edge(source, target, edge_type):
        edges.append({
            "source": source,
            "target": target,
            "type": edge_type
        })

    # Add Topic Nodes
    for topic in TOPICS.keys():
        add_node(topic, "Topic", {"name": topic})

    analyst_topic_history = {}
    global_topic_history = {}

    for q_idx, q_data in enumerate(dataset):
        qid = q_data["quarter_id"]
        
        # Add Quarter Node
        add_node(qid, "Quarter", {"name": qid.upper(), "sort_key": q_data["sort_key"]})
        
        # Process Narration
        narration_turns = q_data["narration"]
        for turn_idx, turn in enumerate(narration_turns):
            spk = turn["speaker"]
            text = turn["text"]
            
            nid = f"{qid}_narr_{turn_idx}"
            topics = classify_topics(text, subsidiary_keywords=subsidiary_keywords)
            
            add_node(nid, "NarrationSegment", {
                "speaker": spk,
                "text": text,
                "quarter": qid,
                "topics": topics
            })
            add_edge(nid, qid, "IN_QUARTER")
            
            for topic in topics:
                add_edge(nid, topic, "DISCUSSES")
                
        # Process Q&A turns
        qa_turns = q_data["qa"]
        
        qa_blocks = []
        current_analyst = None
        current_firm = None
        current_questions = []
        current_answers = []
        
        for turn in qa_turns:
            role = turn["role"]
            spk = turn["speaker"]
            text = turn["text"]
            a_name = ANALYST_ALIASES.get(turn["analyst_name"], turn["analyst_name"])
            a_firm = turn["analyst_firm"]
            
            if role == "Moderator":
                if current_analyst and (current_questions or current_answers):
                    qa_blocks.append({
                        "analyst_name": current_analyst,
                        "analyst_firm": current_firm,
                        "questions": current_questions,
                        "answers": current_answers
                    })
                current_analyst = a_name
                current_firm = a_firm
                current_questions = []
                current_answers = []
            elif role == "Analyst":
                current_questions.append({"speaker": spk, "text": text})
            elif role == "Management":
                current_answers.append({"speaker": spk, "text": text})
                
        if current_analyst and (current_questions or current_answers):
            qa_blocks.append({
                "analyst_name": current_analyst,
                "analyst_firm": current_firm,
                "questions": current_questions,
                "answers": current_answers
            })
            
        for block_idx, block in enumerate(qa_blocks):
            analyst = block["analyst_name"]
            firm = block["analyst_firm"]
            questions = block["questions"]
            answers = block["answers"]
            
            if not analyst:
                continue
                
            add_node(analyst, "Analyst", {"name": analyst, "firm": firm})
            
            q_text = " ".join([q["text"] for q in questions])
            a_text = " ".join([f"{a['speaker']}: {a['text']}" for a in answers])
            
            if not q_text.strip():
                continue
                
            q_nid = f"{qid}_q_{block_idx}"
            a_nid = f"{qid}_a_{block_idx}"
            
            topics = classify_topics(q_text, subsidiary_keywords=subsidiary_keywords)
            if not topics:
                topics = ["General"]
                add_node("General", "Topic", {"name": "General"})
                
            add_node(q_nid, "Question", {
                "text": q_text,
                "quarter": qid,
                "analyst": analyst,
                "topics": topics,
                "block_index": block_idx
            })
            add_edge(q_nid, analyst, "ASKED_BY")
            add_edge(q_nid, qid, "IN_QUARTER")
            
            for topic in topics:
                add_edge(q_nid, topic, "DISCUSSES")
                
            if a_text.strip():
                add_node(a_nid, "Answer", {
                    "text": a_text,
                    "quarter": qid,
                    "topics": topics,
                    "block_index": block_idx
                })
                add_edge(a_nid, qid, "IN_QUARTER")
                add_edge(a_nid, q_nid, "RESPONDED_TO")
                
                for a in answers:
                    m_spk = a["speaker"]
                    add_node(m_spk, "Management", {"name": m_spk})
                    add_edge(a_nid, m_spk, "ANSWERED_BY")
                    
            if analyst not in analyst_topic_history:
                analyst_topic_history[analyst] = {}
                
            for topic in topics:
                if topic != "General":
                    if topic in analyst_topic_history[analyst]:
                        for past_q_nid in analyst_topic_history[analyst][topic]:
                            add_edge(q_nid, past_q_nid, "FOLLOWS_UP_ON")
                    if topic not in analyst_topic_history[analyst]:
                        analyst_topic_history[analyst][topic] = []
                    analyst_topic_history[analyst][topic].append(q_nid)
                    
            if qid not in global_topic_history:
                global_topic_history[qid] = {}
                
            for topic in topics:
                if topic not in global_topic_history[qid]:
                    global_topic_history[qid][topic] = []
                global_topic_history[qid][topic].append({
                    "node_id": q_nid,
                    "analyst": analyst,
                    "block_idx": block_idx
                })
                
            for topic in topics:
                if topic == "General":
                    continue
                same_q_others = global_topic_history[qid].get(topic, [])
                for item in same_q_others:
                    if item["analyst"] != analyst and item["block_idx"] < block_idx:
                        add_edge(q_nid, item["node_id"], "CROSS_FOLLOWS_SAME_CALL")
                        
                if q_idx > 0:
                    prev_qid = dataset[q_idx - 1]["quarter_id"]
                    if prev_qid in global_topic_history and topic in global_topic_history[prev_qid]:
                        for item in global_topic_history[prev_qid][topic]:
                            add_edge(q_nid, item["node_id"], "CROSS_FOLLOWS_PREV_CALL")

    graph_data = {
        "nodes": nodes,
        "edges": edges
    }
    
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
        
    print(f"\nGraph compiled successfully for {bank.display_name}!")
    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")
    print(f"File saved to {graph_path}")
