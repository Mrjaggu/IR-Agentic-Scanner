#!/usr/bin/env python3
import sys
import os
import argparse

# Put src/ into path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="Axis Bank Investor Relations — Decision Cockpit CLI tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # 1. Predict
    predict_parser = subparsers.add_parser("predict", help="Run topic prediction engine for active analysts")

    # 2. Evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluations and backtests")
    evaluate_group = evaluate_parser.add_mutually_exclusive_group(required=True)
    evaluate_group.add_argument("--semantic", action="store_true", help="Run Q4FY26 semantic validation using LLM-as-a-Judge")
    evaluate_group.add_argument("--cross-val", action="store_true", help="Run leave-one-quarter-out cross-validation backtest")

    # 3. Server
    server_parser = subparsers.add_parser("server", help="Start the interactive dashboard local API/sandbox server")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to run the HTTP server on (default: 8000)")

    # 2b. FastAPI live app (ingestion + overall/analyst-specific runs + search/chat, one dashboard)
    app_parser = subparsers.add_parser(
        "app",
        help="Start the live FastAPI dashboard (ingest, run Overall/Analyst-Specific/full prediction, "
             "search + chat over the transcript archive) at http://localhost:<port>/")
    app_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")

    # 4. Prep
    prep_parser = subparsers.add_parser("prep", help="Compile the IR response prep sheet JSON")

    # 5. Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Compile and generate the interactive HTML dashboard")

    # 6. Novelty (LLM narration-novelty signal)
    novelty_parser = subparsers.add_parser(
        "novelty",
        help="LLM pass: flag NEW themes in this quarter's narration vs prior quarters "
             "(writes data/inputs/narration_novelty.json, auto-consumed by predict)")
    novelty_parser.add_argument("--quarter", type=str, default=None,
                                help="Target quarter (default: VAL_QUARTER from settings)")

    # 6b. Peer signal (cross-bank topic salience)
    peers_parser = subparsers.add_parser(
        "peers",
        help="Tag peer-bank transcripts (earnings_transcript/peers/) and compute topic "
             "salience (writes data/inputs/peer_signal.json, auto-consumed by predict)")
    peers_parser.add_argument("--quarter", type=str, default=None,
                              help="Target quarter (default: VAL_QUARTER from settings)")

    # 6c. Question intent (why did the analyst ask that)
    intent_parser = subparsers.add_parser(
        "intent",
        help="LLM pass: classify each historical question as persona_consistent / "
             "narration_triggered / unexplained (writes data/db/question_intent.json)")
    intent_parser.add_argument("--quarter", type=str, default=None,
                               help="Single target quarter (default: loop over last 8 quarters)")
    intent_parser.add_argument("--analysts", type=str, default=None,
                               help="Comma-separated analyst names to scope down to (reduces "
                                    "request size / API usage), e.g. 'MB Mahesh,Piran Engineer'")

    # 6d2. Metrics extraction (regex-based, no LLM)
    subparsers.add_parser(
        "metrics",
        help="Regex-extract structured metrics (NIM, GNPA, PAT, ROE, ...) from narration "
             "across all quarters (writes data/db/metrics_timeseries.json)")

    # 6d. Persona synthesis (aggregate question_intent.json, no LLM)
    subparsers.add_parser(
        "personas-derived",
        help="Aggregate data/db/question_intent.json into per-analyst style rates "
             "(writes data/inputs/analyst_personas_transcript_derived.json)")

    # 7. Compile
    # 9. UI compiler (Section 9 -- three-tab application layer)
    ui_parser = subparsers.add_parser(
        "ui",
        help="Compile frontend/ir_platform_ui.html: Tab 1 Analyst Profiles, Tab 2 "
             "Advanced Search (client-side hybrid retrieval), Tab 3 Prediction "
             "(surfaces agentic_predictions.json). Run `agentic` first for Tab 3 content.")

    # 8. Agentic core (Section 3 of the architecture doc)
    agentic_parser = subparsers.add_parser(
        "agentic",
        help="Run the LangGraph-based agentic prediction core (Planning Agent + "
             "Researcher/Analyst/Verifier crew + Analyst-Specific layer + Question "
             "Framer + Verifier grounding gate). Writes data/outputs/agentic_predictions.json "
             "and data/outputs/agentic_brief.md. Does not touch predictions.json.")
    agentic_parser.add_argument("--quarter", type=str, default=None,
                                help="Target quarter (default: VAL_QUARTER from settings)")

    compile_parser = subparsers.add_parser("compile", help="Compile data and graph resources")
    compile_group = compile_parser.add_mutually_exclusive_group(required=True)
    compile_group.add_argument("--dataset", action="store_true", help="Parse PDF transcripts and compile dataset.json")
    compile_group.add_argument("--graph", action="store_true", help="Generate Neo4j-style document graph.json from dataset.json")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "predict":
        from src.engine import main as engine_main
        engine_main()
        
    elif args.command == "evaluate":
        if args.semantic:
            from src.evaluation import run_semantic_evaluator
            run_semantic_evaluator()
        elif args.cross_val:
            from src.evaluation import run_cross_validation
            run_cross_validation()
            
    elif args.command == "app":
        from src.api.fastapi_app import run_app
        run_app(port=args.port)

    elif args.command == "server":
        from src.api.server import run_server
        run_server(port=args.port)
        
    elif args.command == "prep":
        from src.data.loader import generate_prep_sheet
        generate_prep_sheet()
        
    elif args.command == "dashboard":
        from src.dashboard_gen import main as dashboard_main
        dashboard_main()
        
    elif args.command == "novelty":
        from src.signals.narration_novelty import compute_novelty
        compute_novelty(args.quarter)

    elif args.command == "peers":
        from src.signals.peer_signal import compute_peer_signal
        compute_peer_signal(args.quarter)

    elif args.command == "intent":
        from src.signals.question_intent import compute_question_intent, compute_question_intent_recent
        analysts = set(a.strip() for a in args.analysts.split(",")) if args.analysts else None
        if args.quarter:
            compute_question_intent(args.quarter, analysts=analysts)
        else:
            compute_question_intent_recent(analysts=analysts)

    elif args.command == "personas-derived":
        from src.signals.persona_synthesis import synthesize_personas
        synthesize_personas()

    elif args.command == "metrics":
        from src.signals.metrics_extractor import build_metrics_timeseries
        build_metrics_timeseries()

    elif args.command == "agentic":
        from src.agentic.run_agentic import main as agentic_main
        from src.config.settings import VAL_QUARTER
        agentic_main(args.quarter or VAL_QUARTER)

    elif args.command == "ui":
        from src.ui_compiler import main as ui_main
        ui_main()

    elif args.command == "compile":
        if args.dataset:
            from src.data.dataset_compiler import main as compiler_main
            compiler_main()
        elif args.graph:
            from src.graphs.compiler import main as graph_main
            graph_main()

if __name__ == "__main__":
    main()
