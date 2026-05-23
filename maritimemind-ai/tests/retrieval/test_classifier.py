import pytest
from app.retrieval.query_classifier import QueryClassifier
from app.models.schemas import QueryIntent

def test_classifier_emergency():
    classifier = QueryClassifier()
    assert classifier.classify("Fire in the engine room") == QueryIntent.EMERGENCY
    assert classifier.classify("Man overboard procedure") == QueryIntent.EMERGENCY

def test_classifier_diagram():
    classifier = QueryClassifier()
    assert classifier.classify("Show me the cooling pump diagram") == QueryIntent.DIAGRAM_REQUEST
    assert classifier.classify("Schematic for fuel system") == QueryIntent.DIAGRAM_REQUEST

def test_classifier_troubleshooting():
    classifier = QueryClassifier()
    assert classifier.classify("Why is the oil pressure low alarm ringing?") == QueryIntent.TROUBLESHOOTING
    assert classifier.classify("Troubleshoot generator failure") == QueryIntent.TROUBLESHOOTING

def test_classifier_procedure():
    classifier = QueryClassifier()
    assert classifier.classify("How to replace the fuel filter") == QueryIntent.PROCEDURE
    assert classifier.classify("Maintenance procedure for water pump") == QueryIntent.PROCEDURE

def test_classifier_fallback():
    classifier = QueryClassifier()
    assert classifier.classify("What is the purpose of the ballast system?") == QueryIntent.EXPLANATION

def test_classifier_priority():
    classifier = QueryClassifier()
    # "diagram" + "troubleshoot" -> DIAGRAM_REQUEST should win over TROUBLESHOOTING based on priority 
    # WAIT: our classifier returns the first match in `priority_order`. 
    # Priority is EMERGENCY > DIAGRAM_REQUEST > TROUBLESHOOTING > PROCEDURE > SOP_LOOKUP > EXPLANATION
    assert classifier.classify("Show diagram to troubleshoot alarm") == QueryIntent.DIAGRAM_REQUEST
