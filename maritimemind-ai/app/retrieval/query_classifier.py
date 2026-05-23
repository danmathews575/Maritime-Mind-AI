import re
from typing import Dict, List
from app.models.schemas import QueryIntent

class QueryClassifier:
    """
    Rule-based intent classifier for maritime queries.
    Uses regex keyword matching to map user queries to QueryIntent enums.
    """

    INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.DIAGRAM_REQUEST: [
            r"diagram", r"schematic", r"drawing", r"figure", r"visual",
            r"show\s+me", r"picture", r"illustration", r"layout"
        ],
        QueryIntent.TROUBLESHOOTING: [
            r"troubleshoot", r"fault", r"error", r"alarm", r"failure",
            r"problem", r"issue", r"not\s+working", r"malfunction"
        ],
        QueryIntent.PROCEDURE: [
            r"procedure", r"how\s+to", r"steps?\s+to", r"process\s+for",
            r"maintenance", r"inspection", r"checklist", r"replace", r"repair"
        ],
        QueryIntent.EMERGENCY: [
            r"emergency", r"fire", r"flooding", r"man\s+overboard",
            r"abandon\s+ship", r"collision", r"grounding", r"spill"
        ],
        QueryIntent.SOP_LOOKUP: [
            r"sop", r"standard\s+operating", r"protocol", r"regulation",
            r"compliance", r"solas", r"marpol", r"ism"
        ],
        QueryIntent.EXPLANATION: []  # Default fallback
    }

    def __init__(self):
        # Compile regexes for performance
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    def classify(self, query: str) -> QueryIntent:
        """
        Classifies a query into a single QueryIntent based on keyword matching.
        Returns the first matching intent in priority order.
        Priority: EMERGENCY > DIAGRAM_REQUEST > TROUBLESHOOTING > PROCEDURE > SOP_LOOKUP > EXPLANATION
        """
        priority_order = [
            QueryIntent.EMERGENCY,
            QueryIntent.DIAGRAM_REQUEST,
            QueryIntent.TROUBLESHOOTING,
            QueryIntent.PROCEDURE,
            QueryIntent.SOP_LOOKUP,
        ]

        for intent in priority_order:
            for pattern in self.compiled_patterns[intent]:
                if pattern.search(query):
                    return intent
        
        return QueryIntent.EXPLANATION
