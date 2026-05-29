from typing import Dict, List, Optional
from app.diagnosis.fault_trees import FaultNode, FaultTree

class KnowledgeBase:
    """Pre-built, generic maritime fault trees."""

    _TREES: Dict[str, FaultTree] = {}

    @classmethod
    def init_trees(cls):
        if cls._TREES:
            return
            
        # 1. Main Engine Low Lube Oil Pressure
        me_lo_pressure_nodes = {
            "root": FaultNode(
                node_id="root",
                instruction="Check the local pressure gauge for the main engine lubricating oil. Is it reading low (below 1.5 bar) and matching the alarm?",
                action_type="inspect",
                possible_answers=["Yes", "No", "Unsure"],
                next_nodes={"Yes": "check_pump", "No": "sensor_fault", "Unsure": "check_pump"},
                search_queries=["main engine lube oil low pressure alarm criteria", "local pressure gauge reading lube oil"],
                component_context="main engine lube oil pressure gauge"
            ),
            "sensor_fault": FaultNode(
                node_id="sensor_fault",
                instruction="The local gauge is normal, suggesting a faulty pressure transmitter or wiring issue. Inspect the pressure sensor wiring and connections.",
                action_type="resolution",
                next_nodes={},
                search_queries=["lube oil pressure transmitter replacement", "pressure sensor calibration"],
                component_context="lube oil pressure transmitter"
            ),
            "check_pump": FaultNode(
                node_id="check_pump",
                instruction="Check the lube oil pump discharge pressure. Is it normal (above 4.0 bar)?",
                action_type="measure",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "check_filter", "No": "check_level"},
                search_queries=["lube oil pump discharge pressure normal range", "main engine lube oil pump"],
                component_context="lube oil pump"
            ),
            "check_filter": FaultNode(
                node_id="check_filter",
                instruction="Pump discharge pressure is normal. Check the differential pressure across the lube oil auto-filter. Is it high?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "clean_filter", "No": "check_system_leaks"},
                search_queries=["lube oil auto-filter differential pressure limit", "clean lube oil filter"],
                component_context="lube oil auto-filter"
            ),
            "clean_filter": FaultNode(
                node_id="clean_filter",
                instruction="The filter is clogged. Switch to the bypass/manual filter and initiate the auto-filter cleaning cycle or manually clean the filter elements.",
                action_type="resolution",
                next_nodes={},
                search_queries=["switch lube oil filter to bypass", "clean lube oil filter elements procedure"],
                component_context="lube oil auto-filter elements"
            ),
            "check_system_leaks": FaultNode(
                node_id="check_system_leaks",
                instruction="Filter DP is normal. Inspect the main engine crankcase and external piping for major leaks. Are there any visible leaks?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "repair_leak", "No": "bearing_inspection"},
                search_queries=["main engine lube oil piping leak", "crankcase inspection"],
                component_context="main engine lube oil piping"
            ),
            "repair_leak": FaultNode(
                node_id="repair_leak",
                instruction="Isolate the affected section if possible and repair the leak. Check oil level before restarting.",
                action_type="resolution",
                next_nodes={},
                search_queries=[],
                component_context="piping repair"
            ),
            "bearing_inspection": FaultNode(
                node_id="bearing_inspection",
                instruction="No external leaks found. The issue may be internal (e.g., worn bearings causing excessive clearance). This requires stopping the engine for a crankcase inspection.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=["main bearing clearance measurement", "crankcase inspection procedure"],
                component_context="main engine bearings"
            ),
            "check_level": FaultNode(
                node_id="check_level",
                instruction="Pump discharge pressure is low. Check the lube oil sump tank level. Is the level low?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "top_up_oil", "No": "pump_failure"},
                search_queries=["main engine lube oil sump level indicator"],
                component_context="lube oil sump tank"
            ),
            "top_up_oil": FaultNode(
                node_id="top_up_oil",
                instruction="Top up the sump tank to the normal operating level and check for any major system leaks that caused the loss.",
                action_type="resolution",
                next_nodes={},
                search_queries=["lube oil top up procedure", "sump tank capacity"],
                component_context="lube oil sump tank"
            ),
            "pump_failure": FaultNode(
                node_id="pump_failure",
                instruction="Sump level is normal but pump pressure is low. Switch to the standby lube oil pump. Did the pressure recover?",
                action_type="action",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "overhaul_pump", "No": "suction_issue"},
                search_queries=["start standby lube oil pump", "lube oil pump changeover"],
                component_context="standby lube oil pump"
            ),
            "overhaul_pump": FaultNode(
                node_id="overhaul_pump",
                instruction="The primary pump has failed or worn out. Schedule an overhaul for the primary pump.",
                action_type="resolution",
                next_nodes={},
                search_queries=["lube oil pump overhaul procedure", "screw pump maintenance"],
                component_context="primary lube oil pump"
            ),
            "suction_issue": FaultNode(
                node_id="suction_issue",
                instruction="Both pumps fail to build pressure. Check the pump suction strainer and suction valves. Are they clear and open?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "terminal_unknown", "No": "clear_suction"},
                search_queries=["clean lube oil pump suction strainer"],
                component_context="lube oil pump suction strainer"
            ),
            "clear_suction": FaultNode(
                node_id="clear_suction",
                instruction="Clear the suction strainer or fully open the suction valves, then restart the pump.",
                action_type="resolution",
                next_nodes={},
                search_queries=[],
                component_context="suction strainer"
            ),
            "terminal_unknown": FaultNode(
                node_id="terminal_unknown",
                instruction="Suction is clear but pressure is still low. This is a critical system failure requiring engineering team intervention.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=[],
                component_context="lube oil system"
            )
        }
        
        cls._TREES["me_lo_pressure"] = FaultTree(
            tree_id="me_lo_pressure",
            name="Main Engine Low Lube Oil Pressure Diagnosis",
            subsystem="lube_oil",
            applicable_alarms=["LO_LOW_PRESS", "ALARM_4201"],
            applicable_symptoms=["low_pressure"],
            nodes=me_lo_pressure_nodes,
            root_node_id="root"
        )
        
        # We can add more trees here (e.g. High Exhaust Temp, Cooling Water) as needed.

    @classmethod
    def get_tree(cls, tree_id: str) -> Optional[FaultTree]:
        if not cls._TREES:
            cls.init_trees()
        return cls._TREES.get(tree_id)

    @classmethod
    def get_all_trees(cls) -> List[FaultTree]:
        if not cls._TREES:
            cls.init_trees()
        return list(cls._TREES.values())
        
    @classmethod
    def find_best_tree(cls, symptoms: List[str], subsystems: List[str], alarms: List[str]) -> Optional[FaultTree]:
        if not cls._TREES:
            cls.init_trees()
            
        best_match = None
        best_score = 0
        
        for tree in cls._TREES.values():
            score = 0
            if tree.subsystem in subsystems:
                score += 3
            for alarm in alarms:
                if alarm in tree.applicable_alarms:
                    score += 5
            for symptom in symptoms:
                if symptom in tree.applicable_symptoms:
                    score += 2
                    
            if score > best_score:
                best_score = score
                best_match = tree
                
        # Only return a tree if we have a decent match (score >= 2)
        if best_score >= 2:
            return best_match
        return None
