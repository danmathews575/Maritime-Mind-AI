import pytest
from app.diagnosis.knowledge_base import KnowledgeBase

def test_knowledge_base_initialization():
    KnowledgeBase.init_trees()
    trees = KnowledgeBase.get_all_trees()
    assert len(trees) > 0
    
def test_find_best_tree():
    KnowledgeBase.init_trees()
    # Provide symptoms and alarms that match the lube oil tree
    tree = KnowledgeBase.find_best_tree(
        symptoms=["low_pressure"], 
        subsystems=["lube_oil"], 
        alarms=["ALARM_4201"]
    )
    assert tree is not None
    assert tree.tree_id == "me_lo_pressure"
    
def test_find_best_tree_no_match():
    KnowledgeBase.init_trees()
    tree = KnowledgeBase.find_best_tree(
        symptoms=["vibration"], 
        subsystems=["radar"], 
        alarms=["UNKNOWN_123"]
    )
    assert tree is None
