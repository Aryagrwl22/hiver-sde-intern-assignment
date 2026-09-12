import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.retrieval import build_retrieval_index
from hiver_agent.evaluate_agent import evaluate_agent_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the Advanced AI Layer (Phase 4)")
    parser.add_argument("--build-index", action="store_true", help="Build the semantic retrieval index")
    parser.add_argument("--evaluate", action="store_true", help="Run agent evaluation on golden candidates")
    args = parser.parse_args()
    
    if args.build_index:
        build_retrieval_index()
        
    if args.evaluate:
        evaluate_agent_pipeline()
        
    if not args.build_index and not args.evaluate:
        print("Please specify --build-index or --evaluate")

if __name__ == "__main__":
    main()
