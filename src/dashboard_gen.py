import json
import os
from src.config.settings import (
    PREDICTIONS_PATH,
    PREP_SHEET_PATH,
    DATASET_PATH,
    CROSSVAL_PATH,
    SEMANTIC_EVAL_PATH,
    DASHBOARD_HTML_PATH,
)

def main():
    print("Compiling interactive HTML dashboard...")
    
    # Load files
    with open(PREDICTIONS_PATH, "r") as f:
        pred_data = json.load(f)
        
    with open(PREP_SHEET_PATH, "r") as f:
        prep_data = json.load(f)
        
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)
        
    cv_data = {}
    if os.path.exists(CROSSVAL_PATH):
        with open(CROSSVAL_PATH) as f:
            cv_data = json.load(f)
            
    sem_data = {}
    # Check both semantic_eval_full.json and semantic_eval.json for compatibility
    sem_eval_paths = [SEMANTIC_EVAL_PATH, os.path.join(os.path.dirname(SEMANTIC_EVAL_PATH), "semantic_eval.json")]
    for path in sem_eval_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    sem_data = json.load(f)
                break
            except Exception:
                pass

    pred_json_str = json.dumps(pred_data, indent=2)
    prep_json_str = json.dumps(prep_data, indent=2)
    cv_json_str   = json.dumps(cv_data,   indent=2)
    sem_json_str  = json.dumps(sem_data,  indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Axis Bank IR - Analyst Question Predictor & Prep Sheet</title>
    <meta name="description" content="Predictive Graph RAG dashboard to forecast analyst questions and prepare response strategies for Axis Bank earnings calls.">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0B0F19;
            --bg-card: rgba(21, 29, 48, 0.7);
            --bg-card-hover: rgba(27, 38, 62, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --axis-burgundy: #971B49;
            --axis-burgundy-light: #C02C63;
            --axis-burgundy-glow: rgba(151, 27, 73, 0.25);
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
            --accent-blue: #3F8CFF;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
            --shadow-premium: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --font-display: 'Outfit', sans-serif;
            --font-sans: 'Plus Jakarta Sans', sans-serif;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            scroll-behavior: smooth;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-family: var(--font-sans);
            line-height: 1.6;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(151, 27, 73, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(63, 140, 255, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }}

        .app-container {{
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            max-width: 1440px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Header Styles */
        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 30px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            backdrop-filter: blur(12px);
            margin-bottom: 25px;
            box-shadow: var(--shadow-premium);
        }}

        .brand-logo-container {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .brand-logo {{
            width: 45px;
            height: 45px;
            background: linear-gradient(135deg, var(--axis-burgundy) 0%, var(--axis-burgundy-light) 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 24px;
            color: #fff;
            box-shadow: 0 0 20px var(--axis-burgundy-glow);
            font-family: var(--font-display);
        }}

        .brand-text h1 {{
            font-family: var(--font-display);
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #FFF, #D1D5DB);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-text p {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}

        .engine-badge {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 30px;
            font-size: 13px;
        }}

        .engine-status-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-green);
        }}

        /* Navigation Tab Styles */
        .tabs-nav {{
            display: flex;
            gap: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 6px;
            margin-bottom: 25px;
            width: fit-content;
        }}

        .tab-btn {{
            padding: 10px 24px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-display);
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
        }}

        .tab-btn.active {{
            background: var(--axis-burgundy);
            color: #fff;
            box-shadow: 0 4px 15px var(--axis-burgundy-glow);
        }}

        /* Overview Metric Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}

        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-premium);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(151, 27, 73, 0.3);
        }}

        .metric-card p {{
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}

        .metric-val {{
            font-family: var(--font-display);
            font-size: 38px;
            font-weight: 800;
            background: linear-gradient(135deg, #FFF 30%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 8px 0;
        }}

        .metric-desc {{
            font-size: 12px;
            color: var(--text-secondary);
        }}

        .metric-card.burgundy-tint {{
            position: relative;
            overflow: hidden;
        }}
        .metric-card.burgundy-tint::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--axis-burgundy);
        }}

        /* Content Sections */
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.4s ease forwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Layout Grid */
        .layout-grid {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 25px;
        }}

        @media (max-width: 900px) {{
            .layout-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Filter Panel Styles */
        .filter-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 24px;
            height: fit-content;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-premium);
        }}

        .filter-title {{
            font-family: var(--font-display);
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}

        .search-box {{
            width: 100%;
            padding: 12px 16px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: #fff;
            font-family: var(--font-sans);
            font-size: 14px;
            margin-bottom: 20px;
            transition: border-color 0.3s ease;
        }}

        .search-box:focus {{
            outline: none;
            border-color: var(--axis-burgundy);
        }}

        .filter-group-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .analyst-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 400px;
            overflow-y: auto;
            padding-right: 5px;
        }}

        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        .analyst-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .analyst-item:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .analyst-item.active {{
            background: var(--axis-burgundy-glow);
            border-color: var(--axis-burgundy);
        }}

        .analyst-info {{
            display: flex;
            flex-direction: column;
        }}

        .analyst-name {{
            font-size: 14px;
            font-weight: 600;
        }}

        .analyst-firm-lbl {{
            font-size: 11px;
            color: var(--text-secondary);
        }}

        .status-badge-q4 {{
            width: 8px;
            height: 8px;
            background-color: var(--text-secondary);
            border-radius: 50%;
        }}
        .status-badge-q4.participated {{
            background-color: var(--accent-blue);
            box-shadow: 0 0 8px var(--accent-blue);
        }}

        /* Interactive Prep Cards */
        .prep-cards-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        .prep-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-premium);
            overflow: hidden;
            transition: border-color 0.3s ease;
        }}

        .prep-card:hover {{
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .prep-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 24px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
        }}

        .prep-header-left {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .topic-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            color: var(--accent-blue);
            background: rgba(63, 140, 255, 0.1);
            padding: 4px 10px;
            border-radius: 20px;
            width: fit-content;
        }}

        .predicted-q-title {{
            font-family: var(--font-display);
            font-size: 18px;
            font-weight: 700;
            margin-top: 5px;
        }}

        .prep-body {{
            padding: 24px;
        }}

        /* Rationale Alert */
        .rationale-box {{
            background: rgba(255, 255, 255, 0.03);
            border-left: 3px solid var(--text-secondary);
            padding: 15px 20px;
            border-radius: 0 10px 10px 0;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }}

        /* Response Guidance Alert */
        .guidance-box {{
            background: var(--axis-burgundy-glow);
            border-left: 3px solid var(--axis-burgundy);
            padding: 15px 20px;
            border-radius: 0 10px 10px 0;
            font-size: 14px;
            margin-bottom: 25px;
        }}
        .guidance-box strong {{
            color: var(--axis-burgundy-light);
            font-family: var(--font-display);
            display: block;
            margin-bottom: 4px;
        }}

        /* Past Q&A Threads */
        .history-section-title {{
            font-family: var(--font-display);
            font-size: 15px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 15px;
            color: var(--text-primary);
        }}

        .history-timeline {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}

        .history-item {{
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
        }}

        .history-meta {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            padding-bottom: 8px;
        }}

        .hist-q-text {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
            display: flex;
            gap: 8px;
        }}
        .hist-q-text::before {{
            content: 'Q:';
            color: var(--accent-blue);
            font-weight: 800;
        }}

        .hist-a-text {{
            font-size: 13px;
            color: var(--text-secondary);
            display: flex;
            gap: 8px;
        }}
        .hist-a-text::before {{
            content: 'A:';
            color: var(--accent-green);
            font-weight: 800;
        }}

        .hist-speaker-badges {{
            display: flex;
            gap: 5px;
            margin-top: 8px;
        }}
        .hist-speaker-badge {{
            font-size: 10px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}

        /* Key Metrics Grid inside card */
        .metrics-bubble-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }}
        .metric-bubble {{
            font-size: 12px;
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
        }}

        /* Validation Table Styles */
        .validation-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow-premium);
        }}

        .validation-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        .validation-header h2 {{
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 700;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            text-align: left;
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-family: var(--font-display);
        }}

        td {{
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .val-analyst-cell {{
            display: flex;
            flex-direction: column;
        }}
        .val-analyst-name {{
            font-weight: 600;
        }}
        .val-analyst-firm {{
            font-size: 11px;
            color: var(--text-secondary);
        }}

        .tag-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}

        .tag {{
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
        }}
        .tag.match {{
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
        }}
        .tag.miss {{
            background: rgba(239, 68, 68, 0.05);
            border-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
        }}

        .metric-badge-table {{
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 14px;
        }}
        .metric-badge-table.success-text {{
            color: var(--accent-green);
        }}
        .metric-badge-table.warning-text {{
            color: var(--accent-amber);
        }}
        .metric-badge-table.danger-text {{
            color: #EF4444;
        }}

        /* Empty state styling */
        .empty-state {{
            padding: 40px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 15px;
        }}

        /* Sandbox styles */
        .sandbox-container {{
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .sandbox-sidebar {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .sandbox-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
        }}
        .sandbox-card h3 {{
            color: var(--text-main);
            margin: 0 0 15px 0;
            font-size: 1.1rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        .form-group {{
            margin-bottom: 15px;
        }}
        .form-group label {{
            display: block;
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}
        .form-control {{
            width: 100%;
            background: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-main);
            padding: 8px 12px;
            font-size: 0.9rem;
            font-family: inherit;
            box-sizing: border-box;
        }}
        .form-control:focus {{
            border-color: var(--accent-blue);
            outline: none;
        }}
        textarea.form-control {{
            resize: vertical;
        }}
        .btn {{
            flex: 1;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
        }}
        .btn-primary:hover {{
            opacity: 0.9;
        }}
        .btn-secondary {{
            background: #232D3F;
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }}
        .btn-secondary:hover {{
            background: #2D3A4F;
        }}
        .sandbox-results {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            min-height: 400px;
        }}
        .loading-spinner-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            gap: 10px;
            padding: 40px 0;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

    </style>
</head>
<body>

    <div class="app-container">
        
        <!-- Header -->
        <header>
            <div class="brand-logo-container">
                <div class="brand-logo">A</div>
                <div class="brand-text">
                    <h1>Axis Bank Investor Relations</h1>
                    <p>Analyst Call Predictor Console</p>
                </div>
            </div>
            <div class="engine-badge">
                <div class="engine-status-dot"></div>
                <span id="engine-mode-lbl">Predictor: Fallback</span>
            </div>
        </header>

        <!-- Tabs Navigation -->
        <div class="tabs-nav">
            <button class="tab-btn active" onclick="switchTab('prep-tab')">IR Response Prep Sheet</button>
            <button class="tab-btn" onclick="switchTab('val-tab')">Validation & Evaluation</button>
            <button class="tab-btn" onclick="switchTab('sandbox-tab')">Interactive Sandbox</button>
        </div>

        <!-- Metrics Overview Panel -->
        <div class="metrics-grid">
            <div class="metric-card burgundy-tint">
                <p>Validated Call</p>
                <div class="metric-val">Q4FY26</div>
                <span class="metric-desc">Upcoming Validation Target</span>
            </div>
            <div class="metric-card">
                <p>Topic Match F1-Score</p>
                <div class="metric-val" id="overall-f1-lbl">--</div>
                <span class="metric-desc" id="overall-f1-desc">Weighted Topic Prediction Accuracy</span>
            </div>
            <div class="metric-card">
                <p>Average Precision</p>
                <div class="metric-val" id="overall-prec-lbl">--</div>
                <span class="metric-desc">Precision on predicted topics</span>
            </div>
            <div class="metric-card">
                <p>Average Recall</p>
                <div class="metric-val" id="overall-rec-lbl">--</div>
                <span class="metric-desc">Recall on actual topics</span>
            </div>
        </div>

        <!-- PREP TAB CONTENT -->
        <div id="prep-tab" class="tab-content active">
            <div class="layout-grid">
                <!-- Sidebar filters -->
                <div class="filter-panel">
                    <h3 class="filter-title">Active Analysts</h3>
                    <input type="text" id="analyst-search" class="search-box" placeholder="Search analyst or firm..." onkeyup="filterAnalysts()">
                    
                    <div class="filter-group-label">Analysts List</div>
                    <div class="analyst-list" id="analysts-menu-list">
                        <!-- Dynamic list of analysts -->
                    </div>
                </div>

                <!-- Main Content Pane -->
                <div class="prep-main-pane">
                    <div class="prep-cards-container" id="prep-cards-container">
                        <!-- Dynamic prep cards for selected analyst -->
                    </div>
                </div>
            </div>
        </div>

        <!-- VALIDATION TAB CONTENT -->
        <div id="val-tab" class="tab-content">
            <!-- Accuracy Guide Banner -->
            <div class="validation-card" style="margin-bottom:20px; background: linear-gradient(135deg, rgba(21,29,48,0.9) 0%, rgba(27,38,62,0.9) 100%); border: 1px solid rgba(63,140,255,0.25);">
                <div class="validation-header" style="border-bottom: 1px solid rgba(63,140,255,0.15);">
                    <h2 style="color: var(--accent-blue);">📊 Accuracy Benchmarks</h2>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 20px;">
                    <div style="text-align:center; padding:16px; background:rgba(16,185,129,0.1); border-radius:10px; border:1px solid rgba(16,185,129,0.2);">
                        <div style="font-size:2rem; font-weight:800; color:#10B981;" id="cv-f1-lbl">—</div>
                        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">Cross-Val Topic-F1</div>
                        <div style="font-size:0.7rem; color:#6EE7B7; margin-top:2px;">14 quarters, 92 analyst-calls</div>
                    </div>
                    <div style="text-align:center; padding:16px; background:rgba(63,140,255,0.1); border-radius:10px; border:1px solid rgba(63,140,255,0.2);">
                        <div style="font-size:2rem; font-weight:800; color:var(--accent-blue);" id="cv-boost-lbl">—</div>
                        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">Recall-Boost F1</div>
                        <div style="font-size:0.7rem; color:#93C5FD; margin-top:2px;">predict N+1 topics</div>
                    </div>
                    <div style="text-align:center; padding:16px; background:rgba(245,158,11,0.1); border-radius:10px; border:1px solid rgba(245,158,11,0.2);">
                        <div style="font-size:2rem; font-weight:800; color:var(--accent-amber);" id="sem-f1-lbl">—</div>
                        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">Semantic-F1 (Q4FY26)</div>
                        <div style="font-size:0.7rem; color:#FCD34D; margin-top:2px;">LLM-judge / keyword method</div>
                    </div>
                    <div style="text-align:center; padding:16px; background:rgba(151,27,73,0.1); border-radius:10px; border:1px solid rgba(151,27,73,0.25);">
                        <div style="font-size:2rem; font-weight:800; color:#C02C63;" id="q4-f1-lbl">—</div>
                        <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:4px;">Q4FY26 Blind Topic-F1</div>
                        <div style="font-size:0.7rem; color:#F9A8D4; margin-top:2px;">strict category match</div>
                    </div>
                </div>
                <div style="padding: 0 20px 16px; font-size:0.8rem; color:var(--text-secondary); line-height:1.6;">
                    <strong style="color:var(--text-primary);">Reading this table:</strong>
                    Topic-F1 is the strictest metric — predictions must hit the exact topic label. Semantic-F1 is realistic — questions covering the same concern score as a match even if labelled differently.
                    Cross-val across 14 quarters gives the robust confidence interval; Q4FY26 is the zero-leakage blind test.
                </div>
            </div>

            <!-- Q4FY26 Analyst Table -->
            <div class="validation-card" style="margin-bottom:20px;">
                <div class="validation-header">
                    <h2>Q4FY26 Blind Test — Analyst-Level Breakdown</h2>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Analyst</th>
                                <th>Predicted Topics</th>
                                <th>Actual Topics Asked</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>Topic-F1</th>
                            </tr>
                        </thead>
                        <tbody id="validation-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Cross-validation per-quarter heatmap -->
            <div class="validation-card" style="margin-bottom:20px;">
                <div class="validation-header">
                    <h2>Cross-Validation — Per-Quarter Topic-F1 Heatmap</h2>
                </div>
                <div id="cv-heatmap" style="padding:16px; display:flex; flex-wrap:wrap; gap:10px;">
                </div>
                <div style="padding:0 16px 16px; font-size:0.78rem; color:var(--text-secondary);">
                    Each cell = average F1 across all analysts validated in that quarter. Engine trained on all prior quarters.
                </div>
            </div>

            <!-- Per-analyst lifetime accuracy -->
            <div class="validation-card">
                <div class="validation-header">
                    <h2>Per-Analyst Lifetime Accuracy (across all validation quarters)</h2>
                </div>
                <div id="analyst-lifetime-bars" style="padding:20px;">
                </div>
            </div>
        </div>

        <!-- SANDBOX TAB CONTENT -->
        <div id="sandbox-tab" class="tab-content">
            <div class="sandbox-container">
                <div class="sandbox-sidebar">
                    <div class="sandbox-card">
                        <h3>Prediction Parameters</h3>
                        <div class="form-group">
                            <label for="sandbox-analyst-select">Select Analyst Persona</label>
                            <select id="sandbox-analyst-select" class="form-control">
                                <!-- Dynamic analyst options -->
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="sandbox-file-upload">Upload Earnings Call Transcript (.pdf, .txt)</label>
                            <input type="file" id="sandbox-file-upload" class="form-control" accept=".pdf,.txt" onchange="updateUploadStatus(this)">
                            <div id="file-upload-status" style="font-size: 0.8rem; color: var(--accent-green); margin-top: 6px; display:none;">
                                <i class="fas fa-file-alt"></i> <span id="file-upload-name"></span> loaded.
                            </div>
                        </div>
                        <div class="form-group">
                            <button id="predict-btn" class="btn btn-primary" style="width: 100%;" onclick="runSandboxPrediction()">Predict &amp; Verify Questions</button>
                        </div>
                    </div>
                </div>
                <div class="sandbox-results" id="sandbox-results-panel">
                    <div class="empty-state">
                        <i class="fas fa-file-upload" style="font-size: 3rem; margin-bottom: 1rem; color: var(--accent-blue);"></i>
                        <p>Upload a transcript PDF/TXT, select an analyst from the dropdown, and click <strong>Predict &amp; Verify Questions</strong>.</p>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 5px;">The engine will automatically segment the text, extract presentation details, run Graph RAG predictions, and evaluate them against the actual questions in the call to compute your match score.</p>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- Data Injection -->
    <script>
        const PREDICTION_DATA = {pred_json_str};
        const PREP_SHEET_DATA = {prep_json_str};
        const CV_DATA         = {cv_json_str};
        const SEM_DATA        = {sem_json_str};
    </script>

    <!-- UI Logic Scripts -->
    <script>
        // Init UI
        let selectedAnalyst = "";

        document.addEventListener("DOMContentLoaded", () => {{
            // Set engine mode
            document.getElementById("engine-mode-lbl").innerText = `Predictor: ${{PREDICTION_DATA.api_mode}}`;

            // Set validation metrics
            const metrics = PREDICTION_DATA.validation_summary || {{}};
            const avg_f1 = metrics.average_f1 !== undefined ? metrics.average_f1 : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.average_f1_score : 0.0);
            const avg_p = metrics.average_precision !== undefined ? metrics.average_precision : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.average_precision : 0.0);
            const avg_r = metrics.average_recall !== undefined ? metrics.average_recall : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.average_recall : 0.0);
            const count = metrics.analysts_count !== undefined ? metrics.analysts_count : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.validated_analysts_count : 0);

            document.getElementById("overall-f1-lbl").innerText = `${{(avg_f1 * 100).toFixed(0)}}%`;
            document.getElementById("overall-f1-desc").innerText = `Avg across ${{count}} validated analysts`;
            document.getElementById("overall-prec-lbl").innerText = `${{(avg_p * 100).toFixed(0)}}%`;
            document.getElementById("overall-rec-lbl").innerText = `${{(avg_r * 100).toFixed(0)}}%`;

            // Render Validation Table
            renderValidationTable();

            // Render cross-validation panels
            renderAccuracyBenchmarks();
            renderCVHeatmap();
            renderAnalystLifetimeBars();

            // Populate Analysts Menu
            populateAnalystsMenu();

            // Select first analyst in menu by default
            const firstItem = document.querySelector(".analyst-item");
            if (firstItem) {{
                firstItem.click();
            }}

            // Load Sandbox dropdown analysts
            loadSandboxAnalysts();
        }});

        function switchTab(tabId) {{
            document.querySelectorAll(".tab-content").forEach(tab => {{
                tab.classList.remove("active");
            }});
            document.querySelectorAll(".tab-btn").forEach(btn => {{
                btn.classList.remove("active");
            }});

            document.getElementById(tabId).classList.add("active");
            
            let btnIdx = 0;
            if (tabId === 'val-tab') btnIdx = 1;
            if (tabId === 'sandbox-tab') btnIdx = 2;
            document.querySelectorAll(".tab-btn")[btnIdx].classList.add("active");
        }}

        function populateAnalystsMenu() {{
            const list = document.getElementById("analysts-menu-list");
            list.innerHTML = "";

            const sortedAnalysts = Object.keys(PREP_SHEET_DATA).sort();
            
            const reports = PREDICTION_DATA.validation_summary ? PREDICTION_DATA.validation_summary.reports : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.analyst_reports : []);
            const q4Participants = new Set(reports.map(r => r.analyst));

            sortedAnalysts.forEach(a => {{
                const predList = (PREP_SHEET_DATA[a] && PREP_SHEET_DATA[a].predictions) || [];
                if (predList.length === 0) return;

                const isQ4 = q4Participants.has(a);

                const div = document.createElement("div");
                div.className = `analyst-item ${{selectedAnalyst === a ? 'active' : ''}}`;
                div.id = `menu-item-${{a.replace(/\s+/g, '-')}}`;
                div.onclick = () => selectAnalyst(a);
                
                div.innerHTML = `
                    <div class="analyst-info">
                        <span class="analyst-name">${{a}}</span>
                    </div>
                    <span class="status-badge-q4 ${{isQ4 ? 'participated' : ''}}" title="${{isQ4 ? 'Participated in Q4FY26' : 'Active Analyst'}}"></span>
                `;
                list.appendChild(div);
            }});
        }}

        function selectAnalyst(analystName) {{
            selectedAnalyst = analystName;
            
            document.querySelectorAll(".analyst-item").forEach(item => {{
                item.classList.remove("active");
            }});
            const activeItem = document.getElementById(`menu-item-${{analystName.replace(/\s+/g, '-')}}`);
            if (activeItem) {{
                activeItem.classList.add("active");
            }}

            renderPrepCards(analystName);
        }}

        function filterAnalysts() {{
            const query = document.getElementById("analyst-search").value.toLowerCase();
            document.querySelectorAll(".analyst-item").forEach(item => {{
                const name = item.querySelector(".analyst-name").innerText.toLowerCase();
                if (name.includes(query)) {{
                    item.style.display = "flex";
                }} else {{
                    item.style.display = "none";
                }}
            }});
        }}

        function renderPrepCards(analystName) {{
            const container = document.getElementById("prep-cards-container");
            container.innerHTML = "";

            const entry = PREP_SHEET_DATA[analystName];
            const predList = (entry && entry.predictions) || [];
            if (predList.length === 0) {{
                container.innerHTML = '<div class="empty-state">No predictions available for this analyst.</div>';
                return;
            }}

            if (entry.analyst_style) {{
                const styleBanner = document.createElement("div");
                styleBanner.className = "guidance-box";
                styleBanner.style.cssText = "background: rgba(151, 27, 73, 0.05); border-left-color: var(--accent-blue); margin-bottom: 16px;";
                styleBanner.innerHTML = `<strong style="color: var(--accent-blue);">ANALYST STYLE (from transcript-derived history)</strong> ${{entry.analyst_style}}`;
                container.appendChild(styleBanner);
            }}

            predList.forEach((pred, idx) => {{
                const card = document.createElement("div");
                card.className = "prep-card";

                // Historical Q&As formatting
                const histQAs = pred.analyst_history || pred.historical_qas || [];
                const peerQAs = pred.peer_history || [];
                const combinedQAs = [...histQAs, ...peerQAs];

                let historyHtml = "";
                if (combinedQAs.length > 0) {{
                    historyHtml = `
                        <div class="history-section-title">Historical Dialogues &amp; Precedents</div>
                        <div class="history-timeline">
                            ${{combinedQAs.slice(0, 3).map(h => `
                                <div class="history-item">
                                    <div class="history-meta">
                                        <span>Quarter: <strong>${{h.quarter.toUpperCase()}}</strong></span>
                                        <span>Analyst: <strong>${{h.analyst}}</strong></span>
                                    </div>
                                    <div class="hist-q-text">${{h.question_text || h.question}}</div>
                                    <div class="hist-a-text">${{h.answer_text || h.answer}}</div>
                                    <div class="hist-speaker-badges">
                                        ${{(h.answer_speakers || h.speakers || []).map(s => `<span class="hist-speaker-badge">${{s}}</span>`).join('')}}
                                    </div>
                                </div>
                            `).join('')}}
                        </div>
                    `;
                }}

                card.innerHTML = `
                    <div class="prep-header">
                        <div class="prep-header-left">
                            <span class="topic-label">${{pred.topic}}</span>
                            <h3 class="predicted-q-title">Predicted Question #${{idx+1}}</h3>
                        </div>
                    </div>
                    <div class="prep-body">
                        <div class="guidance-box" style="background: rgba(151, 27, 73, 0.05); border-left-color: var(--accent-blue); padding: 15px 20px;">
                            <strong style="color: var(--accent-blue);">PREDICTED QUESTION FORECAST</strong>
                            "${{pred.predicted_question || pred.question}}"
                        </div>

                        <div class="rationale-box">
                            <strong>RATIONALE:</strong> ${{pred.rationale}}
                        </div>

                        <div class="guidance-box">
                            <strong>ANSWER GUIDANCE STRATEGY</strong>
                            ${{pred.guidance_rule}}
                        </div>

                        ${{historyHtml}}
                    </div>
                `;
                container.appendChild(card);
            }});
        }}

        function renderAccuracyBenchmarks() {{
            const cv = CV_DATA;
            const sem = SEM_DATA;
            const vm = PREDICTION_DATA.validation_summary || PREDICTION_DATA.validation_metrics || {{}};

            const cvF1 = cv.overall_f1_base !== undefined ? cv.overall_f1_base : (cv.overall_f1 || 0.0);
            const cvBoost = cv.overall_f1_boost !== undefined ? cv.overall_f1_boost : 0.0;
            const semF1 = sem.q4fy26 ? sem.q4fy26.avg_semantic_f1 : (sem.crossval ? sem.crossval.semantic_f1 : 0.0);
            const q4F1 = vm.average_f1 !== undefined ? vm.average_f1 : (vm.average_f1_score || 0.0);

            document.getElementById("cv-f1-lbl").innerText    = (cvF1 * 100).toFixed(0) + "%";
            document.getElementById("cv-boost-lbl").innerText = (cvBoost * 100).toFixed(0) + "%";
            document.getElementById("sem-f1-lbl").innerText    = (semF1 * 100).toFixed(0) + "%";
            document.getElementById("q4-f1-lbl").innerText     = (q4F1 * 100).toFixed(0) + "%";
        }}

        function renderCVHeatmap() {{
            const cv = CV_DATA;
            const container = document.getElementById("cv-heatmap");
            container.innerHTML = "";
            const qdata = cv.per_quarter_details || cv.results_by_quarter;
            if (!qdata) return;

            const allF1s = Object.values(qdata).map(q => q.f1_base || q.f1 || 0);
            const maxF1 = Math.max(...allF1s, 0.01);

            Object.entries(qdata).forEach(([qid, stats]) => {{
                const f1 = stats.f1_base || stats.f1 || 0;
                const pct = f1 / maxF1;
                const r = Math.round(151 + (16 - 151) * pct);
                const g = Math.round(27  + (185 - 27) * pct);
                const b = Math.round(73  + (129 - 73) * pct);
                const cell = document.createElement("div");
                cell.style.cssText = `
                    background: rgba(${{r}},${{g}},${{b}},0.25);
                    border: 1px solid rgba(${{r}},${{g}},${{b}},0.5);
                    border-radius: 8px; padding: 10px 12px; min-width: 100px;
                    text-align: center; cursor: default;
                `;
                cell.innerHTML = `
                    <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.05em;">${{qid.replace('fy','FY').replace('q','Q')}}</div>
                    <div style="font-size:1.3rem; font-weight:700; color:rgb(${{r}},${{g}},${{b}}); margin:4px 0;">${{(f1*100).toFixed(0)}}%</div>
                    <div style="font-size:0.7rem; color:var(--text-secondary);">${{stats.n}} analysts</div>
                `;
                container.appendChild(cell);
            }});
        }}

        function renderAnalystLifetimeBars() {{
            const cv = CV_DATA;
            const container = document.getElementById("analyst-lifetime-bars");
            container.innerHTML = "";
            const lifetime = cv.lifetime || cv.analyst_lifetime;
            if (!lifetime) return;

            const sorted = Object.entries(lifetime)
                .sort((a, b) => b[1].avg_f1 - a[1].avg_f1);

            sorted.slice(0, 15).forEach(([analyst, stats]) => {{
                const pct = (stats.avg_f1 * 100).toFixed(0);
                const color = stats.avg_f1 >= 0.6 ? "#10B981" : stats.avg_f1 >= 0.35 ? "#F59E0B" : "#EF4444";
                const row = document.createElement("div");
                row.style.cssText = "margin-bottom: 10px;";
                row.innerHTML = `
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:0.82rem; color:var(--text-primary);">${{analyst}}</span>
                        <span style="font-size:0.82rem; font-weight:600; color:${{color}};">${{pct}}% <span style="color:var(--text-secondary); font-weight:400;">(${{stats.calls}} calls)</span></span>
                    </div>
                    <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px; overflow:hidden;">
                        <div style="width:${{pct}}%; background:${{color}}; height:100%; border-radius:4px; transition:width 0.8s;"></div>
                    </div>
                `;
                container.appendChild(row);
            }});
        }}

        function renderValidationTable() {{
            const tbody = document.getElementById("validation-table-body");
            tbody.innerHTML = "";

            const reports = PREDICTION_DATA.validation_summary ? PREDICTION_DATA.validation_summary.reports : (PREDICTION_DATA.validation_metrics ? PREDICTION_DATA.validation_metrics.analyst_reports : []);
            if (!reports || reports.length === 0) {{
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No validation data available.</td></tr>';
                return;
            }}

            reports.forEach(r => {{
                const tr = document.createElement("tr");

                const f1 = r.f1_score !== undefined ? r.f1_score : r.f1 || 0.0;
                const prec = r.precision !== undefined ? r.precision : 0.0;
                const rec = r.recall !== undefined ? r.recall : 0.0;
                const pred_topics = r.predicted_topics || r.predicted || [];
                const actual_topics = r.actual_topics || r.actual || [];

                let f1Class = "danger-text";
                if (f1 >= 0.75) f1Class = "success-text";
                else if (f1 >= 0.4) f1Class = "warning-text";

                const predTags = pred_topics.map(topic => {{
                    const isMatch = actual_topics.includes(topic);
                    return `<span class="tag ${{isMatch ? 'match' : ''}}">${{topic}}</span>`;
                }}).join(' ');

                const actualTags = actual_topics.map(topic => {{
                    const isMatch = pred_topics.includes(topic);
                    return `<span class="tag ${{isMatch ? 'match' : 'miss'}}">${{topic}}</span>`;
                }}).join(' ');

                tr.innerHTML = `
                    <td>
                        <div class="val-analyst-cell">
                            <span class="val-analyst-name">${{r.analyst}}</span>
                        </div>
                    </td>
                    <td><div class="tag-list">${{predTags}}</div></td>
                    <td><div class="tag-list">${{actualTags}}</div></td>
                    <td><span class="metric-badge-table ${{prec >= 0.5 ? 'success-text' : 'danger-text'}}">${{(prec * 100).toFixed(0)}}%</span></td>
                    <td><span class="metric-badge-table ${{rec >= 0.5 ? 'success-text' : 'danger-text'}}">${{(rec * 100).toFixed(0)}}%</span></td>
                    <td><span class="metric-badge-table ${{f1Class}}">${{(f1 * 100).toFixed(0)}}%</span></td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function updateUploadStatus(input) {{
            const statusDiv = document.getElementById("file-upload-status");
            const nameSpan = document.getElementById("file-upload-name");
            if (input.files && input.files[0]) {{
                statusDiv.style.display = "block";
                nameSpan.innerText = input.files[0].name;
            }} else {{
                statusDiv.style.display = "none";
            }}
        }}

        async function loadSandboxAnalysts() {{
            try {{
                const res = await fetch("/api/analysts");
                const analysts = await res.json();
                const select = document.getElementById("sandbox-analyst-select");
                select.innerHTML = "";
                analysts.forEach(a => {{
                    const opt = document.createElement("option");
                    opt.value = a;
                    opt.innerText = a;
                    select.appendChild(opt);
                }});
            }} catch (err) {{
                console.error("Failed to load sandbox analysts:", err);
            }}
        }}

        async function runSandboxPrediction() {{
            const analyst = document.getElementById("sandbox-analyst-select").value;
            const fileInput = document.getElementById("sandbox-file-upload");
            const resultsPanel = document.getElementById("sandbox-results-panel");
            
            if (!analyst) {{
                alert("Please select an analyst.");
                return;
            }}
            
            const file = fileInput.files[0];
            if (!file) {{
                alert("Please upload a transcript file (.pdf or .txt) first.");
                return;
            }}
            
            resultsPanel.innerHTML = `
                <div class="loading-spinner-container">
                    <div class="spinner"></div>
                    <p style="font-weight:600; color:var(--text-main);">Parsing transcript &amp; extracting narration...</p>
                    <p style="font-size: 0.8rem; opacity: 0.7; color:var(--text-secondary);">Running Graph RAG + LLM Predictor...</p>
                </div>
            `;
            
            try {{
                const formData = new FormData();
                formData.append("analyst", analyst);
                formData.append("file", file);
                
                const res = await fetch("/api/predict", {{
                    method: "POST",
                    body: formData
                }});
                
                if (!res.ok) {{
                    throw new Error("Prediction API returned error " + res.status);
                }}
                
                const data = await res.json();
                renderSandboxResults(data);
            }} catch (err) {{
                resultsPanel.innerHTML = `
                    <div class="empty-state" style="color: #FF5A5A;">
                        <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                        <p>Prediction Failed: ${{err.message}}</p>
                        <p style="font-size: 0.85rem; margin-top: 5px; color:var(--text-secondary);">Make sure the dashboard server is running and your API keys are valid.</p>
                    </div>
                `;
            }}
        }}

        function renderSandboxResults(data) {{
            const resultsPanel = document.getElementById("sandbox-results-panel");
            resultsPanel.innerHTML = "";
            
            const {{ analyst, predictions, validation }} = data;
            
            const predContainer = document.createElement("div");
            predContainer.style.display = "flex";
            predContainer.style.flexDirection = "column";
            predContainer.style.gap = "15px";
            
            const headerTitle = document.createElement("h2");
            headerTitle.innerText = `Predictions for ${{analyst}}`;
            headerTitle.style.color = "var(--text-main)";
            headerTitle.style.margin = "0 0 10px 0";
            headerTitle.style.fontSize = "1.2rem";
            predContainer.appendChild(headerTitle);
            
            predictions.forEach((p, idx) => {{
                const card = document.createElement("div");
                card.className = "analyst-card active";
                
                let histHtml = "";
                if (p.historical_qas && p.historical_qas.length > 0) {{
                    p.historical_qas.forEach(h => {{
                        histHtml += `
                            <div class="qa-item" style="border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px; margin-bottom:10px;">
                                <div class="qa-meta" style="font-size:0.75rem; color:var(--accent-purple); font-weight:600; margin-bottom:4px;">${{h.quarter}} | Asked by: ${{h.analyst}}</div>
                                <div class="qa-text" style="font-size:0.85rem; color:var(--text-main); line-height:1.4;"><strong>Q:</strong> ${{h.question}}</div>
                                <div class="qa-text" style="font-size:0.85rem; color:var(--text-secondary); line-height:1.4; margin-top: 5px; background:rgba(0,0,0,0.2); padding:8px; border-radius:4px;"><strong>A:</strong> ${{h.answer}}</div>
                            </div>
                        `;
                    }});
                }} else {{
                    histHtml = `<div class="empty-state" style="padding: 15px; font-size:0.8rem;">No matching historical dialogue found on this topic.</div>`;
                }}
                
                let metricsHtml = "";
                if (p.key_metrics_referenced && p.key_metrics_referenced.length > 0) {{
                    metricsHtml = `
                        <div class="metrics-ref-panel" style="margin-bottom:15px;">
                            <span style="font-size: 0.75rem; color: var(--text-secondary); font-weight: 600;">KEY HISTORICAL METRICS:</span>
                            <div class="metrics-grid" style="margin-top: 5px; display: flex; gap: 8px; flex-wrap: wrap;">
                                ${{p.key_metrics_referenced.map(m => `<span class="metric-pill" style="background:#232D3F; border:1px solid var(--border-color); color:var(--accent-blue); padding:3px 8px; border-radius:12px; font-size:0.75rem;">${{m}}</span>`).join("")}}
                            </div>
                        </div>
                    `;
                }}
                
                card.innerHTML = `
                    <div class="analyst-header" onclick="this.nextElementSibling.classList.toggle('collapsed')" style="display:flex; justify-content:space-between; align-items:center; cursor:pointer; background:#1b2436; padding:12px 20px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="topic-tag" style="background:var(--accent-blue); color:white; padding:3px 8px; border-radius:4px; font-size:0.75rem; font-weight:600;">${{p.topic}}</span>
                            <h3 style="margin: 0; font-size: 1rem; color: var(--text-main);">Forecast Question ${{idx + 1}}</h3>
                        </div>
                        <span style="font-size: 0.8rem; color: var(--accent-blue);">Expand Details &amp; History <i class="fas fa-chevron-down"></i></span>
                    </div>
                    <div class="analyst-body" style="padding:20px; background:var(--bg-card); border: 1px solid var(--border-color); border-top:none;">
                        <div style="background: rgba(11, 240, 187, 0.05); border-left: 3px solid var(--accent-green); padding: 12px; border-radius: 4px; margin-bottom: 15px;">
                            <strong style="color: var(--text-main); font-size: 0.85rem; display:block; margin-bottom:4px;">PREDICTED QUESTION:</strong>
                            <p style="margin: 0; font-style: italic; color: var(--text-main); line-height: 1.4; font-size:0.95rem;">"${{p.predicted_question}}"</p>
                        </div>
                        
                        <div style="margin-bottom: 15px;">
                            <strong style="color: var(--text-main); font-size: 0.85rem; display: block; margin-bottom: 4px;">PREDICTOR RATIONALE:</strong>
                            <p style="margin: 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.4;">${{p.rationale}}</p>
                        </div>
                        
                        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid var(--border-color); padding: 12px; border-radius: 4px; margin-bottom: 15px;">
                            <strong style="color: var(--accent-blue); font-size: 0.85rem; display: block; margin-bottom: 4px;">MANAGEMENT PREP STRATEGY:</strong>
                            <p style="margin: 0; font-size: 0.85rem; color: var(--text-main); line-height: 1.4;">${{p.guidance_rule}}</p>
                        </div>
                        
                        ${{metricsHtml}}
                        
                        <div style="margin-top: 15px; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px;">
                            <strong style="color: var(--text-main); font-size: 0.85rem; display: block; margin-bottom: 8px;">HISTORICAL CONTEXT (PREVIOUS DIALOGUE):</strong>
                            <div class="qa-timeline" style="display: flex; flex-direction: column; gap: 10px;">
                                ${{histHtml}}
                            </div>
                        </div>
                    </div>
                `;
                predContainer.appendChild(card);
            }});
            
            resultsPanel.appendChild(predContainer);
            
            if (validation) {{
                const verCard = document.createElement("div");
                verCard.className = "sandbox-card";
                verCard.style.marginTop = "20px";
                verCard.style.border = "1px solid var(--accent-blue)";
                verCard.style.background = "rgba(10, 25, 47, 0.4)";
                
                const actualTags = validation.actual_topics.map(topic => {{
                    const isMatch = predictions.some(p => p.topic === topic);
                    return `<span class="tag ${{isMatch ? 'match' : 'miss'}}" style="margin-right:5px; margin-bottom:5px;">${{topic}}</span>`;
                }}).join(' ');
                
                const f1Pct = (validation.f1_score * 100).toFixed(0);
                const precPct = (validation.precision * 100).toFixed(0);
                const recPct = (validation.recall * 100).toFixed(0);
                
                let scoreColor = "var(--accent-green)";
                if (validation.f1_score < 0.5) scoreColor = "#EF4444";
                else if (validation.f1_score < 0.8) scoreColor = "var(--accent-amber)";
                
                verCard.innerHTML = `
                    <h3 style="color: var(--accent-blue); border-bottom:1px solid var(--border-color); padding-bottom:8px; margin-top:0;">
                        <i class="fas fa-check-circle"></i> Transcript Verification &amp; Match Score
                    </h3>
                    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:15px; margin-bottom:20px; text-align:center;">
                        <div style="background:var(--bg-main); border:1px solid var(--border-color); padding:10px; border-radius:6px;">
                            <span style="font-size:0.75rem; color:var(--text-secondary); display:block; margin-bottom:5px;">TOPIC MATCH F1-SCORE</span>
                            <strong style="font-size:1.8rem; color:${{scoreColor}};">${{f1Pct}}%</strong>
                        </div>
                        <div style="background:var(--bg-main); border:1px solid var(--border-color); padding:10px; border-radius:6px;">
                            <span style="font-size:0.75rem; color:var(--text-secondary); display:block; margin-bottom:5px;">PRECISION</span>
                            <strong style="font-size:1.5rem; color:var(--text-main);">${{precPct}}%</strong>
                        </div>
                        <div style="background:var(--bg-main); border:1px solid var(--border-color); padding:10px; border-radius:6px;">
                            <span style="font-size:0.75rem; color:var(--text-secondary); display:block; margin-bottom:5px;">RECALL</span>
                            <strong style="font-size:1.5rem; color:var(--text-main);">${{recPct}}%</strong>
                        </div>
                    </div>
                    
                    <div style="margin-bottom:15px;">
                        <strong style="color:var(--text-main); font-size:0.85rem; display:block; margin-bottom:5px;">ACTUAL TOPICS ASKED IN CALL:</strong>
                        <div class="tag-list" style="display:flex; flex-wrap:wrap;">${{actualTags}}</div>
                    </div>
                    
                    <div>
                        <strong style="color:var(--text-main); font-size:0.85rem; display:block; margin-bottom:5px;">ACTUAL QUESTION TEXT PARSED FROM CALL:</strong>
                        <div style="background:rgba(0,0,0,0.2); border:1px solid var(--border-color); padding:12px; border-radius:4px; font-size:0.85rem; color:var(--text-main); line-height:1.5; font-style:italic; max-height: 250px; overflow-y: auto;">
                            "${{validation.actual_question_text || 'No speech turns found for this analyst in the uploaded transcript.'}}"
                        </div>
                    </div>
                `;
                resultsPanel.appendChild(verCard);
            }}
        }}
        
        window.updateUploadStatus = updateUploadStatus;
        window.runSandboxPrediction = runSandboxPrediction;
        window.loadSandboxAnalysts = loadSandboxAnalysts;
        window.renderSandboxResults = renderSandboxResults;
        window.switchTab = switchTab;
    </script>
</body>
</html>
"""

    with open(DASHBOARD_HTML_PATH, "w") as f:
        f.write(html_content)

    print("Interactive HTML Dashboard compiled successfully!")
    print(f"File saved to {DASHBOARD_HTML_PATH}")
