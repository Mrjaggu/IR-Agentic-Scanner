import os
import pypdf
import re
import json
from src.config.settings import DATASET_PATH, EARNINGS_TRANSCRIPT_DIR, paths_for
from src.config.banks import DEFAULT_BANK, get_bank

def get_quarter_info(filename):
    m = re.search(r'(q[1-4]fy\d{2})', filename, re.IGNORECASE)
    if m:
        qid = m.group(1).lower()
        # sort key: year * 10 + quarter
        year = int(qid[4:6])
        qtr = int(qid[1])
        return qid, year * 10 + qtr
    return filename, 0

def clean_name(name):
    if not name:
        return None
    name_lower = name.lower()
    guidance_words = ["please", "go", "ahead", "thank", "you", "sir"]
    first_idx = len(name)
    for word in guidance_words:
        m = re.search(r'\b' + word + r'\b', name_lower)
        if m and m.start() < first_idx:
            first_idx = m.start()
    name = name[:first_idx]
    name = re.sub(r'^\s*[\.\,\:\;\-\_]+', '', name)
    name = re.sub(r'[\.\,\:\;\-\_]+\s*$', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Standardize name variations to canonical forms
    nl = name.lower()
    if "mahrukh" in nl or "maruk" in nl:
        return "Mahrukh Adajania"
    if "kunal" in nl:
        return "Kunal Shah"
    if "mahesh" in nl:
        return "MB Mahesh"
    if "nitin" in nl:
        return "Nitin Aggarwal"
    if "param" in nl or "p subramanian" in nl:
        return "Param Subramanian"
    if "suresh" in nl:
        return "Suresh Ganapathy"
    if "abhishek" in nl:
        return "Abhishek Murarka"
    if "adarsh" in nl:
        return "Adarsh Parasrampuria"
    if "prakhar" in nl:
        return "Prakhar Agarwal"
    if "rahul" in nl:
        return "Rahul Jain"
    if "saurabh" in nl:
        return "Saurabh Kumar"
    if "sumeet" in nl:
        return "Sumeet Kariwala"
    if "chintan" in nl:
        return "Chintan Joshi"
    if "rikin" in nl:
        return "Rikin Shah"
    if "jay" in nl or "jai" in nl:
        return "Jai Mundhra"
    if "pranav" in nl:
        return "Pranav Gundlapalle"
        
    return name

def clean_firm(firm):
    if not firm:
        return None
    firm_lower = firm.lower()
    guidance_words = ["please", "go", "ahead", "thank", "you", "sir"]
    first_idx = len(firm)
    for word in guidance_words:
        m = re.search(r'\b' + word + r'\b', firm_lower)
        if m and m.start() < first_idx:
            first_idx = m.start()
    firm = firm[:first_idx]
    firm = re.sub(r'^\s*[\.\,\:\;\-\_]+', '', firm)
    firm = re.sub(r'[\.\,\:\;\-\_]+\s*$', '', firm)
    return re.sub(r'\s+', ' ', firm).strip()

def extract_analyst_info_from_text(text):
    text_clean = re.sub(r'\s+', ' ', text)
    
    # 1. line of <name> from <firm>
    m = re.search(r'line of\s+([A-Za-z\s\.\-\’\‘]+?)\s+from\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))
        
    # 2. line of <name>, <firm>
    m = re.search(r'line of\s+([A-Za-z\s\.\-\’\‘]+?)\s*,\s*([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))
        
    # 3. line of <name> of <firm>
    m = re.search(r'line of\s+([A-Za-z\s\.\-\’\‘]+?)\s+of\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))

    # 3b. line of <name> with <firm>
    m = re.search(r'line of\s+([A-Za-z\s\.\-\’\‘]+?)\s+with\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))

    # 4. next question is from <name> from <firm>
    m = re.search(r'question is from\s+([A-Za-z\s\.\-\’\‘]+?)\s+from\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))

    # 5. next question from <name> from <firm>
    m = re.search(r'question from\s+([A-Za-z\s\.\-\’\‘]+?)\s+from\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))

    # 6. next question from <name> of <firm>
    m = re.search(r'question from\s+([A-Za-z\s\.\-\’\‘]+?)\s+of\s+([A-Za-z0-9\s\.\-\’\‘\&]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), clean_firm(m.group(2))

    # 7. question from <name>
    m = re.search(r'question from\s+([A-Za-z\s\.\-\’\‘]+)', text_clean, re.IGNORECASE)
    if m:
        return clean_name(m.group(1)), None
        
    return None, None

DATE_PATTERN = re.compile(
    r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,\s+\d{4}'
)

def parse_transcript(pdf_path, legal_name="Axis Bank Limited"):
    # legal_name: the transcript's own running header/footer text to scrub (e.g.
    # "Kotak Mahindra Bank Limited") -- see src.config.banks.BankConfig.legal_name.
    # Defaults to Axis's legal name so any EXISTING caller that doesn't pass this
    # explicitly (peer_signal.py calls parse_transcript(path) with no legal_name)
    # behaves exactly as before: a no-op scrub for non-Axis transcripts, same as
    # when this regex was hardcoded inline.
    reader = pypdf.PdfReader(pdf_path)
    lines = []
    call_date = None
    for page in reader.pages:
        for line in page.extract_text().split('\n'):
            line_str = line.strip()
            if not line_str:
                continue
            if re.search(r'Page \d+ of \d+', line_str):
                continue
            if re.search(re.escape(legal_name), line_str, re.IGNORECASE):
                continue
            m = DATE_PATTERN.search(line_str)
            if m:
                if call_date is None:
                    call_date = m.group(0)
                continue
            lines.append(line_str)
            
    turns = []
    current_speaker = None
    current_text_parts = []
    speaker_pattern = re.compile(r'^([A-Z][a-zA-Z\s\.\-\’\‘]+)\s*:\s*(.*)$')
    
    for line in lines:
        m = speaker_pattern.match(line)
        if m:
            if current_speaker:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text_parts)
                })
            current_speaker = m.group(1).strip()
            current_text_parts = [m.group(2).strip()]
        else:
            if current_speaker:
                current_text_parts.append(line)
                
    if current_speaker:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_text_parts)
        })
        
    # Find the Q&A start index
    in_qa = False
    qa_start_idx = len(turns)
    for idx, turn in enumerate(turns):
        speaker = turn["speaker"]
        text = turn["text"]
        if speaker == "Moderator":
            if any(x in text.lower() for x in ["begin the question", "begin with the question", "open the floor", "questions and answers", "question-and-answer", "first question"]):
                in_qa = True
                qa_start_idx = idx
                break
            name, firm = extract_analyst_info_from_text(text)
            if name:
                in_qa = True
                qa_start_idx = idx
                break
                
    narration_turns = turns[:qa_start_idx]
    qa_raw_turns = turns[qa_start_idx:]
    
    blocks = []
    current_block = []
    
    for turn in qa_raw_turns:
        speaker = turn["speaker"]
        text = turn["text"]
        
        is_intro = False
        if speaker == "Moderator":
            name, firm = extract_analyst_info_from_text(text)
            if name:
                is_intro = True
                if current_block:
                    blocks.append(current_block)
                current_block = [{
                    "turn": turn,
                    "is_intro": True,
                    "analyst_name": name,
                    "analyst_firm": firm
                }]
                
        if not is_intro:
            if current_block:
                current_block.append({
                    "turn": turn,
                    "is_intro": False
                })
                
    if current_block:
        blocks.append(current_block)
        
    qa_dialogues = []
    for block in blocks:
        intro_turn = block[0]
        analyst_name = intro_turn["analyst_name"]
        analyst_firm = intro_turn["analyst_firm"]
        
        block_speakers = set()
        for b in block[1:]:
            spk = b["turn"]["speaker"]
            if spk not in ["Moderator", "Operator"]:
                block_speakers.add(spk)
                
        analyst_speakers = set()
        
        for spk in block_speakers:
            spk_clean = spk.lower()
            name_parts = analyst_name.lower().split()
            first_name = name_parts[0] if name_parts else ""
            
            is_analyst = False
            if spk_clean == analyst_name.lower():
                is_analyst = True
            elif first_name and first_name in spk_clean:
                is_analyst = True
            elif any(part in spk_clean for part in name_parts):
                is_analyst = True
                
            if is_analyst:
                analyst_speakers.add(spk)
                
        if not analyst_speakers and block_speakers:
            for b in block[1:]:
                spk = b["turn"]["speaker"]
                if spk not in ["Moderator", "Operator"]:
                    analyst_speakers.add(spk)
                    break
                    
        for b in block:
            turn = b["turn"]
            spk = turn["speaker"]
            text = turn["text"]
            
            role = "Management"
            if spk in ["Moderator", "Operator"]:
                role = "Moderator"
            elif spk in analyst_speakers:
                role = "Analyst"
                
            qa_dialogues.append({
                "speaker": spk,
                "role": role,
                "text": text,
                "analyst_name": analyst_name,
                "analyst_firm": analyst_firm
            })
            
    return narration_turns, qa_dialogues, call_date

def main(bank_id: str = DEFAULT_BANK):
    bank = get_bank(bank_id)
    paths = paths_for(bank_id)
    directory = paths.earnings_transcript_dir
    dataset_path = paths.dataset_path
    if not os.path.exists(directory):
        print(f"Error: directory {directory} not found.")
        return
        
    files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    
    sorted_files = []
    for file in files:
        qid, skey = get_quarter_info(file)
        sorted_files.append((file, qid, skey))
        
    sorted_files.sort(key=lambda x: x[2])
    
    analyst_to_firm_global = {}
    
    print("Pass 1: Compiling analyst-firm relationships...")
    for file, qid, skey in sorted_files:
        filepath = os.path.join(directory, file)
        _, qa, _ = parse_transcript(filepath, legal_name=bank.legal_name)
        for turn in qa:
            name = turn["analyst_name"]
            firm = turn["analyst_firm"]
            if name and firm:
                analyst_to_firm_global[name.lower()] = firm
                
    def resolve_firm(name):
        if not name:
            return None
        name_lower = name.lower()
        if name_lower in analyst_to_firm_global:
            return analyst_to_firm_global[name_lower]
        for k, v in analyst_to_firm_global.items():
            if k in name_lower or name_lower in k:
                return v
        return None

    dataset = []
    
    print("Pass 2: Extracting dialogues and building dataset...")
    for file, qid, skey in sorted_files:
        print(f"Processing {qid} ({file})...")
        filepath = os.path.join(directory, file)
        narr, qa, call_date = parse_transcript(filepath, legal_name=bank.legal_name)

        for turn in qa:
            if turn["analyst_name"] and not turn["analyst_firm"]:
                turn["analyst_firm"] = resolve_firm(turn["analyst_name"])
                
        q_analysts = set()
        for turn in qa:
            if turn["analyst_name"] and turn["analyst_name"] not in ["Moderator", "Operator"]:
                q_analysts.add((turn["analyst_name"], turn["analyst_firm"]))
                
        dataset.append({
            "quarter_id": qid,
            "sort_key": skey,
            "filename": file,
            "call_date": call_date,
            "narration": narr,
            "qa": qa,
            "analysts": [{"name": n, "firm": f} for n, f in q_analysts]
        })
        
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\nDataset compiled successfully for {bank.display_name}! Total quarters: {len(dataset)}")
    print(f"File saved to {dataset_path}")
