import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from app.models.schemas import QueryIntent


@dataclass
class ClassificationResult:
    """
    Rich classification output with intent, metadata routing hints,
    and priority chunk flags for downstream retrieval filtering.
    """
    intent: QueryIntent
    department_hint: Optional[str] = None
    subsystem_hint: Optional[str] = None
    priority_chunk_flags: Dict[str, bool] = field(default_factory=dict)


class QueryClassifier:
    """
    Rule-based intent classifier for maritime queries with metadata routing.
    Extracts intent, department/subsystem hints, and priority chunk flags
    to enable metadata-filtered retrieval.
    """

    INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
        QueryIntent.DIAGRAM_REQUEST: [
            r"diagram", r"schematic", r"drawing", r"figure", r"visual",
            r"show\s+me", r"picture", r"illustration", r"layout",
            r"wiring", r"piping\s+diagram", r"circuit\s+diagram"
        ],
        QueryIntent.TROUBLESHOOTING: [
            r"troubleshoot", r"fault", r"error", r"alarm", r"failure",
            r"problem", r"issue", r"not\s+working", r"malfunction",
            r"diagnos", r"defect", r"abnormal",
            r"knock", r"vibrat", r"overheat", r"smoke", r"leak", r"noise", 
            r"high\s+temperature", r"low\s+pressure", r"stuck", r"seized", r"hunt"
        ],
        QueryIntent.PROCEDURE: [
            r"procedure", r"how\s+to", r"steps?\s+to", r"process\s+for",
            r"maintenance", r"inspection", r"checklist", r"replace", r"repair",
            r"overhaul", r"disassembl", r"reassembl", r"adjust",
            r"calibrat", r"bleed", r"purge", r"top\s+up"
        ],
        QueryIntent.EMERGENCY: [
            r"emergency", r"fire\s+in", r"fire\s+on\s+board", r"fire\s+in\s+engine\s+room",
            r"fire\s+alarm", r"fire\s+broke", r"fire\s+outbreak",
            r"flooding", r"man\s+overboard",
            r"abandon\s+ship", r"collision", r"grounding", r"spill",
            r"evacuat", r"muster", r"distress", r"mayday", r"sos",
            r"blackout", r"power\s+failure", r"steering\s+failure",
            r"propulsion\s+loss", r"immedi(?:ate)?\s+action",
        ],
        QueryIntent.SOP_LOOKUP: [
            r"sop", r"standard\s+operating", r"protocol", r"regulation",
            r"compliance", r"solas", r"marpol", r"ism", r"stcw",
            r"code\s+of\s+practice", r"statutory", r"class\s+requirement"
        ],
        QueryIntent.EXPLANATION: []  # Default fallback
    }

    # Department routing patterns
    DEPARTMENT_PATTERNS: Dict[str, List[str]] = {
        "engineering": [
            r"engine", r"motor", r"cylinder", r"piston", r"crankshaft",
            r"turbocharger", r"turbocharg", r"generator", r"pump",
            r"compressor", r"separator", r"purifier", r"boiler",
            r"heat\s+exchanger", r"condenser", r"lube\s+oil",
            r"fuel\s+oil", r"cooling\s+water", r"exhaust",
            r"hydraulic", r"pneumatic", r"bearing", r"shaft",
            r"valve", r"pipe", r"piping", r"electrical", r"wiring",
        ],
        "deck": [
            r"ballast", r"mooring", r"cargo", r"anchor", r"winch",
            r"capstan", r"crane", r"hatch", r"derrick", r"rope",
            r"hawser", r"bollard", r"fairlead", r"deck\s+machinery",
            r"hold", r"tank\s+top", r"loading", r"unloading",
            r"stability", r"draft", r"trim", r"free\s+surface",
        ],
        "navigation": [
            r"radar", r"ecdis", r"gps", r"bridge", r"compass",
            r"ais", r"navtex", r"chart", r"passage\s+plan",
            r"steering", r"autopilot", r"gyro", r"magnetic",
            r"furuno", r"radio", r"vhf", r"gmdss",
        ],
        "safety": [
            r"fire\s+fight", r"lifeboat", r"life\s+raft", r"lifejacket",
            r"life\s+jacket", r"evacuat", r"muster", r"drill",
            r"safety\s+equip", r"extinguish", r"breathing\s+apparatus",
            r"ppe", r"first\s+aid", r"medical", r"man\s+overboard",
            r"abandon", r"immersion\s+suit",
        ],
    }

    # Subsystem hint patterns (more specific than department)
    SUBSYSTEM_PATTERNS: Dict[str, List[str]] = {
        "lube_oil": [r"lube\s+oil", r"lubricat", r"\blo\b"],
        "cooling_water": [r"cooling\s+water", r"\bcw\b", r"fresh\s+water\s+cool", r"sea\s+water\s+cool"],
        "fuel_oil": [r"fuel\s+oil", r"\bfo\b", r"heavy\s+fuel", r"\bhfo\b", r"diesel"],
        "exhaust": [r"exhaust\s+gas", r"exhaust\s+temp", r"turbo"],
        "main_engine": [r"main\s+engine", r"\bme\b", r"man\s+b", r"main\s+propulsion"],
        "auxiliary_engine": [r"auxiliary\s+engine", r"aux\s+engine", r"\bae\b", r"genset"],
        "steering_gear": [r"steering\s+gear", r"rudder"],
        "ballast_system": [r"ballast", r"ballast\s+water", r"ballast\s+tank"],
        "fire_fighting": [r"fire\s+fight", r"co2\s+system", r"foam\s+system", r"fire\s+pump"],
    }

    def __init__(self):
        # Compile regexes for performance
        self.compiled_intent_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        self.compiled_dept_patterns = {
            dept: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for dept, patterns in self.DEPARTMENT_PATTERNS.items()
        }
        self.compiled_subsystem_patterns = {
            subsys: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for subsys, patterns in self.SUBSYSTEM_PATTERNS.items()
        }

    def classify(self, query: str) -> ClassificationResult:
        """
        Classifies a query into intent with metadata routing hints.

        Returns a ClassificationResult with:
        - intent: The primary query intent
        - department_hint: Detected department for metadata filtering
        - subsystem_hint: Detected subsystem for finer filtering
        - priority_chunk_flags: Metadata flags to boost matching chunks

        Priority order: EMERGENCY > DIAGRAM_REQUEST > TROUBLESHOOTING > PROCEDURE > SOP_LOOKUP > EXPLANATION
        """
        # 1. Classify intent
        intent = self._classify_intent(query)

        # 2. Extract department hint
        department_hint = self._extract_department(query)

        # 3. Extract subsystem hint
        subsystem_hint = self._extract_subsystem(query)

        # 4. Determine priority chunk flags based on intent
        priority_flags = self._get_priority_flags(intent)

        return ClassificationResult(
            intent=intent,
            department_hint=department_hint,
            subsystem_hint=subsystem_hint,
            priority_chunk_flags=priority_flags,
        )

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classifies query intent using regex keyword matching."""
        priority_order = [
            QueryIntent.EMERGENCY,
            QueryIntent.DIAGRAM_REQUEST,
            QueryIntent.TROUBLESHOOTING,
            QueryIntent.PROCEDURE,
            QueryIntent.SOP_LOOKUP,
        ]

        for intent in priority_order:
            for pattern in self.compiled_intent_patterns[intent]:
                if pattern.search(query):
                    return intent
        
        return QueryIntent.EXPLANATION

    def _extract_department(self, query: str) -> Optional[str]:
        """Extract department hint from query using keyword patterns."""
        best_dept = None
        best_count = 0

        for dept, patterns in self.compiled_dept_patterns.items():
            match_count = sum(1 for p in patterns if p.search(query))
            if match_count > best_count:
                best_count = match_count
                best_dept = dept

        # Only return if we have a clear signal (at least 1 match)
        return best_dept if best_count >= 1 else None

    def _extract_subsystem(self, query: str) -> Optional[str]:
        """Extract subsystem hint from query for finer-grained filtering."""
        for subsys, patterns in self.compiled_subsystem_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return subsys
        return None

    @staticmethod
    def _get_priority_flags(intent: QueryIntent) -> Dict[str, bool]:
        """
        Map intent to chunk metadata flags that should be prioritized
        during retrieval scoring.
        """
        flags = {}
        if intent == QueryIntent.PROCEDURE:
            flags["contains_procedure"] = True
        elif intent == QueryIntent.EMERGENCY:
            flags["contains_emergency_workflow"] = True
        elif intent == QueryIntent.TROUBLESHOOTING:
            flags["contains_procedure"] = True  # Troubleshooting often involves procedures
        elif intent == QueryIntent.DIAGRAM_REQUEST:
            flags["contains_diagram_reference"] = True
        return flags
