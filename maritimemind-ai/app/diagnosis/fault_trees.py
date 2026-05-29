from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class FaultNode(BaseModel):
    """A single step in a fault tree diagnosis."""
    node_id: str
    instruction: str = Field(..., description="The instruction or question to present to the user.")
    action_type: str = Field("question", description="Type of action: question, measure, inspect, resolution, terminal_unresolved")
    possible_answers: List[str] = Field(default_factory=list, description="Expected answers (e.g., ['Yes', 'No'])")
    next_nodes: Dict[str, str] = Field(default_factory=dict, description="Map of answer to next node_id")
    search_queries: List[str] = Field(default_factory=list, description="Background queries to run against retrieval engine for context")
    component_context: str = Field("", description="The specific component being analyzed in this step (e.g., 'main lube oil pump')")

class FaultTree(BaseModel):
    """A full fault tree for a specific problem domain."""
    tree_id: str
    name: str
    subsystem: str
    applicable_alarms: List[str] = Field(default_factory=list)
    applicable_symptoms: List[str] = Field(default_factory=list)
    nodes: Dict[str, FaultNode]
    root_node_id: str

    def get_node(self, node_id: str) -> Optional[FaultNode]:
        return self.nodes.get(node_id)
