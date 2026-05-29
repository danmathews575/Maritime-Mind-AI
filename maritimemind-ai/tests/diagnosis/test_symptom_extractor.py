import pytest
from app.diagnosis.symptom_extractor import SymptomExtractor

def test_extract_alarm_codes():
    query = "I have an alarm #4201 on the main engine"
    result = SymptomExtractor.extract(query)
    assert "4201" in result["alarms"]
    
def test_extract_symptoms():
    query = "The lube oil pressure is too low"
    result = SymptomExtractor.extract(query)
    assert "low_pressure" in result["symptoms"]
    assert "lube_oil" in result["subsystems"]
    
def test_extract_multiple():
    query = "High exhaust temp and vibration on generator ALARM 550"
    result = SymptomExtractor.extract(query)
    assert "550" in result["alarms"]
    assert "high_temp" in result["symptoms"]
    assert "vibration" in result["symptoms"]
    assert "generator" in result["subsystems"]
