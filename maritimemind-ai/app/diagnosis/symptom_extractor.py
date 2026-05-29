import re
from typing import List, Dict, Optional, Tuple
from app.utils.logger import setup_logger

logger = setup_logger("maritimemind.diagnosis.symptom_extractor")

class SymptomExtractor:
    """Extracts symptoms and alarm codes from user queries using regex patterns."""
    
    ALARM_CODE_PATTERN = re.compile(r'\b(alarm|code|error)\s*#?\s*([a-z0-9-]+)\b', re.IGNORECASE)
    
    SYMPTOM_PATTERNS = {
        "low_pressure": re.compile(r'low\s+(.*?)pressure|pressure\s+(?:is\s+)?(?:too\s+)?low\b', re.IGNORECASE),
        "high_temp": re.compile(r'high\s+(.*?)temp(?:erature)?|temp(?:erature)?\s+(?:is\s+)?(?:too\s+)?high\b', re.IGNORECASE),
        "leakage": re.compile(r'leak(?:ing|age)?\b', re.IGNORECASE),
        "vibration": re.compile(r'vibrat(?:ing|ion)\b', re.IGNORECASE),
        "noise": re.compile(r'(?:abnormal\s+|strange\s+)?noise\b', re.IGNORECASE),
        "failure_to_start": re.compile(r'won\'t\s+start|fails?\s+to\s+start|not\s+starting\b', re.IGNORECASE)
    }

    SUBSYSTEM_PATTERNS = {
        "lube_oil": re.compile(r'lube\s+oil|lubricating\s+oil|lo\b', re.IGNORECASE),
        "cooling_water": re.compile(r'cooling\s+water|cw|fresh\s+water\s+cooling|sea\s+water\s+cooling', re.IGNORECASE),
        "fuel_oil": re.compile(r'fuel\s+oil|fo|heavy\s+fuel\s+oil|hfo', re.IGNORECASE),
        "exhaust": re.compile(r'exhaust\s+gas|exhaust', re.IGNORECASE),
        "main_engine": re.compile(r'main\s+engine|me\b', re.IGNORECASE),
        "generator": re.compile(r'generator|gen\b', re.IGNORECASE)
    }

    @classmethod
    def extract(cls, query: str) -> Dict[str, List[str]]:
        """
        Extracts structured diagnostic info from a query.
        Returns:
            {
                "alarms": [...],
                "symptoms": [...],
                "subsystems": [...]
            }
        """
        result = {
            "alarms": [],
            "symptoms": [],
            "subsystems": []
        }

        # Extract alarm codes
        for match in cls.ALARM_CODE_PATTERN.finditer(query):
            result["alarms"].append(match.group(2).upper())

        # Extract symptoms
        for symptom, pattern in cls.SYMPTOM_PATTERNS.items():
            if pattern.search(query):
                result["symptoms"].append(symptom)

        # Extract subsystems
        for subsys, pattern in cls.SUBSYSTEM_PATTERNS.items():
            if pattern.search(query):
                result["subsystems"].append(subsys)

        return result
