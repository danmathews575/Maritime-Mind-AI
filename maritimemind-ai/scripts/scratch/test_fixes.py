from app.orchestration.graph import run_agent_workflow
from app.configs.config import settings

# Test a standard retrieval query
print("--- Standard Query ---")
state1 = run_agent_workflow("What is the main engine oil pressure alarm limit?")
print("Intent:", state1.get("intent"))
print("Next Agent:", state1.get("next_agent", "N/A"))
print("Response:", state1.get("response_text"))

# Test a troubleshooting query
print("\n--- Troubleshooting Query ---")
state2 = run_agent_workflow("I have a low pressure alarm for the main engine lube oil")
print("Intent:", state2.get("intent"))
print("Next Agent:", state2.get("next_agent", "N/A"))
print("Response:", state2.get("response_text"))

# Test interactive step
print("\n--- Interactive Query ---")
state3 = run_agent_workflow("Yes", history=[{"role": "user", "content": "I have a low pressure alarm for the main engine lube oil"}, {"role": "assistant", "content": state2.get("response_text", "")}])
print("Intent:", state3.get("intent"))
print("Next Agent:", state3.get("next_agent", "N/A"))
print("Response:", state3.get("response_text"))
