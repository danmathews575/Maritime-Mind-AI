import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.orchestration.graph import run_agent_workflow
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.agent_test")

def run_test(query: str):
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    try:
        # Check if Ollama is responsive first by running the workflow (graph initializes it)
        # However, to gracefully handle Ollama being offline without crashing, we catch exceptions.
        final_state = run_agent_workflow(query)
        
        print("\n--- AGENT ROUTING & VERIFICATION ---")
        # Format the intent nicely if it's an enum
        intent = final_state.get('intent')
        intent_name = intent.name if intent else 'UNKNOWN'
        
        print(f"Intent: {intent_name}")
        print(f"Retrieval Strategy: {final_state.get('retrieval_strategy', 'UNKNOWN')}")
        print(f"Retrieval Confidence: {final_state.get('retrieval_confidence', 0.0):.2f}")
        print(f"Verification Passed: {final_state.get('verification_passed', False)}")
        print(f"Quality Passed: {final_state.get('quality_passed', False)}")
        print(f"Retry Count: {final_state.get('retry_count', 0)}")
        
        error = final_state.get('error')
        if error:
            print(f"\nGraph reported error: {error}")
        
        print("\n--- RETRIEVAL RESULTS ---")
        text_results = final_state.get('text_results', [])
        image_results = final_state.get('image_results', [])
        print(f"Text Chunks Retrieved: {len(text_results)}")
        print(f"Images Retrieved: {len(image_results)}")
        
        if image_results:
            print("\nImages:")
            for idx, img in enumerate(image_results):
                print(f"  [{idx+1}] {img.image_path} (Page {img.page_number}) - {img.caption}")
                
        print("\n--- FINAL SYNTHESIS ---")
        print(final_state.get('response_text', 'No response generated.'))
        
        print("\n--- CITATIONS ---")
        citations = final_state.get('citations', [])
        if citations:
            # Deduplicate citations for display
            unique_citations = set()
            for c in citations:
                unique_citations.add(f"{c.get('manual_name', 'Unknown')} (Page {c.get('page_number', '?')})")
            
            for idx, c in enumerate(unique_citations):
                print(f"  [{idx+1}] {c}")
        else:
            print("  No citations.")
            
    except Exception as e:
        print(f"\nERROR running workflow: {e}")
        logger.exception("Workflow failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LangGraph Multi-Agent Workflow")
    parser.add_argument("--query", type=str, help="Run a specific query")
    args = parser.parse_args()

    # The test script handles warnings gracefully (like if vector DB is empty)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    if args.query:
        run_test(args.query)
    else:
        test_queries = [
            "Why is cooling water pressure low?",
            "Explain ballast tank operation.",
            "What is the engine room fire response procedure?",
            "Show me the diagram for the main engine cooling system."
        ]
        
        print(f"Running {len(test_queries)} predefined test queries through the agent workflow.")
        print("Note: If the vector database is empty or Ollama is not running, these tests may fail gracefully.")
        for q in test_queries:
            run_test(q)
            print("\n")
