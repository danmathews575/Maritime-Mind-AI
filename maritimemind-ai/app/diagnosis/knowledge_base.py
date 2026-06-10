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
        
        # ── 2. High Exhaust Temperature ──────────────────────────────────
        high_exh_temp_nodes = {
            "root": FaultNode(
                node_id="root",
                instruction="Check the exhaust gas temperature readings on the engine monitoring system. Which cylinder(s) show high exhaust temperature deviation (>50°C above normal)?",
                action_type="inspect",
                possible_answers=["Single cylinder", "Multiple cylinders", "All cylinders"],
                next_nodes={"Single cylinder": "check_injector", "Multiple cylinders": "check_turbocharger", "All cylinders": "check_air_cooler"},
                search_queries=["exhaust gas temperature limits main engine", "high exhaust temperature alarm"],
                component_context="exhaust gas temperature monitoring"
            ),
            "check_injector": FaultNode(
                node_id="check_injector",
                instruction="A single cylinder deviation suggests a fuel injector issue. Check the fuel injector for that cylinder. Is the injector spray pattern normal when tested?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "check_exhaust_valve", "No": "replace_injector"},
                search_queries=["fuel injector test procedure", "injector spray pattern inspection"],
                component_context="fuel injector"
            ),
            "replace_injector": FaultNode(
                node_id="replace_injector",
                instruction="Replace the faulty fuel injector with a reconditioned spare. After replacement, run the engine and monitor the exhaust temperature for that cylinder.",
                action_type="resolution",
                next_nodes={},
                search_queries=["fuel injector replacement procedure", "injector nozzle change"],
                component_context="fuel injector replacement"
            ),
            "check_exhaust_valve": FaultNode(
                node_id="check_exhaust_valve",
                instruction="Injector is normal. Check the exhaust valve for that cylinder. Perform a compression test or check valve seating. Is compression normal?",
                action_type="measure",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "check_scavenge", "No": "overhaul_exhaust_valve"},
                search_queries=["exhaust valve inspection", "compression test procedure main engine"],
                component_context="exhaust valve"
            ),
            "overhaul_exhaust_valve": FaultNode(
                node_id="overhaul_exhaust_valve",
                instruction="The exhaust valve requires overhaul. Remove and recondition the valve, check the valve seat and spindle for carbon buildup or erosion.",
                action_type="resolution",
                next_nodes={},
                search_queries=["exhaust valve overhaul procedure", "valve seat grinding"],
                component_context="exhaust valve overhaul"
            ),
            "check_scavenge": FaultNode(
                node_id="check_scavenge",
                instruction="Compression is normal. Open the scavenge port inspection covers for that cylinder. Are the scavenge ports fouled or is there evidence of a scavenge fire?",
                action_type="inspect",
                possible_answers=["Ports fouled", "Evidence of fire", "Ports clear"],
                next_nodes={"Ports fouled": "clean_scavenge_ports", "Evidence of fire": "scavenge_fire_response", "Ports clear": "terminal_single_cyl"},
                search_queries=["scavenge port inspection", "scavenge fire detection"],
                component_context="scavenge ports"
            ),
            "clean_scavenge_ports": FaultNode(
                node_id="clean_scavenge_ports",
                instruction="Clean the scavenge ports and inspect the piston rings for excessive blowpast. Schedule a piston pull if ring condition is poor.",
                action_type="resolution",
                next_nodes={},
                search_queries=["scavenge port cleaning", "piston ring inspection"],
                component_context="scavenge ports and piston rings"
            ),
            "scavenge_fire_response": FaultNode(
                node_id="scavenge_fire_response",
                instruction="⚠️ SCAVENGE FIRE DETECTED. Reduce engine speed immediately. Increase cylinder lubrication. Do NOT stop the engine suddenly. Monitor closely and prepare fire-fighting equipment.",
                action_type="resolution",
                next_nodes={},
                search_queries=["scavenge fire emergency procedure", "scavenge fire response"],
                component_context="scavenge fire emergency"
            ),
            "terminal_single_cyl": FaultNode(
                node_id="terminal_single_cyl",
                instruction="All checks are normal but temperature remains high. This may indicate fuel timing issues for this cylinder. Requires Chief Engineer assessment and possible timing adjustment.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=["fuel timing adjustment", "VIT adjustment"],
                component_context="fuel timing"
            ),
            "check_turbocharger": FaultNode(
                node_id="check_turbocharger",
                instruction="Multiple cylinders affected suggests a turbocharger issue. Check the turbocharger RPM and boost pressure. Is the turbocharger running at normal speed?",
                action_type="measure",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "check_air_cooler", "No": "tc_fouling"},
                search_queries=["turbocharger RPM normal range", "turbocharger performance check"],
                component_context="turbocharger"
            ),
            "tc_fouling": FaultNode(
                node_id="tc_fouling",
                instruction="Turbocharger performance is degraded. Perform a water washing of the turbine and compressor side. If performance does not recover, the turbocharger requires overhaul.",
                action_type="resolution",
                next_nodes={},
                search_queries=["turbocharger water washing procedure", "turbine side cleaning"],
                component_context="turbocharger cleaning"
            ),
            "check_air_cooler": FaultNode(
                node_id="check_air_cooler",
                instruction="Check the scavenge air temperature after the air cooler. Is it higher than normal (above 45°C)?",
                action_type="measure",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "clean_air_cooler", "No": "terminal_multi_cyl"},
                search_queries=["charge air cooler temperature limits", "scavenge air temperature"],
                component_context="charge air cooler"
            ),
            "clean_air_cooler": FaultNode(
                node_id="clean_air_cooler",
                instruction="The charge air cooler is fouled. Clean both the air side and the water side of the cooler. Check cooling water flow through the cooler.",
                action_type="resolution",
                next_nodes={},
                search_queries=["charge air cooler cleaning procedure", "air cooler maintenance"],
                component_context="charge air cooler"
            ),
            "terminal_multi_cyl": FaultNode(
                node_id="terminal_multi_cyl",
                instruction="Air cooler is clean and turbocharger is normal. This may indicate a fuel quality issue or engine load imbalance. Check fuel analysis reports and power balance.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=["fuel quality analysis", "engine power balance test"],
                component_context="fuel quality and load balance"
            ),
        }
        
        cls._TREES["high_exhaust_temp"] = FaultTree(
            tree_id="high_exhaust_temp",
            name="Main Engine High Exhaust Gas Temperature Diagnosis",
            subsystem="exhaust",
            applicable_alarms=["EXH_HIGH_TEMP", "ALARM_4301", "EXH_TEMP_DEV"],
            applicable_symptoms=["high_temp"],
            nodes=high_exh_temp_nodes,
            root_node_id="root"
        )
        
        # ── 3. Cooling Water System Failure ──────────────────────────────
        cw_failure_nodes = {
            "root": FaultNode(
                node_id="root",
                instruction="Check the jacket cooling water outlet temperature. Is it rising above normal operating range (typically above 85°C)?",
                action_type="inspect",
                possible_answers=["Yes", "No, but alarm triggered"],
                next_nodes={"Yes": "check_expansion_tank", "No, but alarm triggered": "check_cw_sensor"},
                search_queries=["jacket cooling water temperature limits", "cooling water alarm criteria"],
                component_context="jacket cooling water system"
            ),
            "check_cw_sensor": FaultNode(
                node_id="check_cw_sensor",
                instruction="Temperature appears normal on local gauges but the alarm triggered. This suggests a faulty temperature sensor or transmitter. Inspect the sensor wiring and calibration.",
                action_type="resolution",
                next_nodes={},
                search_queries=["cooling water temperature sensor calibration", "temperature transmitter replacement"],
                component_context="cooling water temperature sensor"
            ),
            "check_expansion_tank": FaultNode(
                node_id="check_expansion_tank",
                instruction="Check the cooling water expansion tank level. Is the level low or is there evidence of water loss?",
                action_type="inspect",
                possible_answers=["Yes, level is low", "No, level is normal"],
                next_nodes={"Yes, level is low": "find_leak", "No, level is normal": "check_thermostat"},
                search_queries=["expansion tank level indicator", "cooling water system capacity"],
                component_context="cooling water expansion tank"
            ),
            "find_leak": FaultNode(
                node_id="find_leak",
                instruction="⚠️ Cooling water loss detected. Check for leaks at: cylinder head gaskets, heat exchanger, piping connections, and pump seals. Is a leak found?",
                action_type="inspect",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "repair_cw_leak", "No": "internal_leak"},
                search_queries=["cooling water leak detection", "cylinder head gasket inspection"],
                component_context="cooling water piping and seals"
            ),
            "repair_cw_leak": FaultNode(
                node_id="repair_cw_leak",
                instruction="Isolate the leaking section if possible. Repair or replace the leaking component. Top up the expansion tank with treated fresh water and vent the system.",
                action_type="resolution",
                next_nodes={},
                search_queries=["cooling water system repair", "expansion tank top up procedure"],
                component_context="cooling water piping repair"
            ),
            "internal_leak": FaultNode(
                node_id="internal_leak",
                instruction="No external leak found. The water may be leaking internally into the engine (through a cracked liner or head gasket). Check lube oil for water contamination. This requires engineering team investigation.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=["lube oil water contamination test", "cylinder liner crack detection"],
                component_context="internal cooling water leak"
            ),
            "check_thermostat": FaultNode(
                node_id="check_thermostat",
                instruction="Water level is normal. Check the thermostatic control valve. Is the three-way valve operating correctly and diverting flow through the cooler?",
                action_type="inspect",
                possible_answers=["Yes", "No, stuck or faulty"],
                next_nodes={"Yes": "check_cw_pump", "No, stuck or faulty": "replace_thermostat"},
                search_queries=["thermostatic valve inspection", "three-way valve operation"],
                component_context="thermostatic control valve"
            ),
            "replace_thermostat": FaultNode(
                node_id="replace_thermostat",
                instruction="The thermostatic valve is stuck or malfunctioning. Replace the wax element or the entire valve assembly. Until replaced, manually control the bypass valve.",
                action_type="resolution",
                next_nodes={},
                search_queries=["thermostatic valve replacement", "cooling water bypass valve"],
                component_context="thermostatic valve replacement"
            ),
            "check_cw_pump": FaultNode(
                node_id="check_cw_pump",
                instruction="Check the jacket cooling water pump discharge pressure and flow. Is the pump delivering normal pressure?",
                action_type="measure",
                possible_answers=["Yes", "No"],
                next_nodes={"Yes": "check_heat_exchanger", "No": "switch_cw_pump"},
                search_queries=["cooling water pump discharge pressure", "centrifugal pump performance"],
                component_context="jacket cooling water pump"
            ),
            "switch_cw_pump": FaultNode(
                node_id="switch_cw_pump",
                instruction="Switch to the standby cooling water pump. If pressure recovers, the primary pump requires overhaul (impeller wear, mechanical seal, or bearing failure).",
                action_type="resolution",
                next_nodes={},
                search_queries=["cooling water pump changeover", "centrifugal pump overhaul"],
                component_context="standby cooling water pump"
            ),
            "check_heat_exchanger": FaultNode(
                node_id="check_heat_exchanger",
                instruction="Pump is normal. Check the central cooler / plate heat exchanger. Is there a high temperature differential across the cooler (inlet vs outlet)?",
                action_type="measure",
                possible_answers=["Yes, large differential", "No, small differential"],
                next_nodes={"Yes, large differential": "clean_heat_exchanger", "No, small differential": "terminal_cw"},
                search_queries=["heat exchanger performance check", "plate cooler inspection"],
                component_context="central plate cooler"
            ),
            "clean_heat_exchanger": FaultNode(
                node_id="clean_heat_exchanger",
                instruction="The heat exchanger is fouled. Open and clean the plate stack (fresh water side and sea water side). Inspect zinc anodes and replace if consumed.",
                action_type="resolution",
                next_nodes={},
                search_queries=["plate heat exchanger cleaning procedure", "zinc anode replacement"],
                component_context="plate heat exchanger cleaning"
            ),
            "terminal_cw": FaultNode(
                node_id="terminal_cw",
                instruction="All cooling system components check normal but temperature remains high. This may indicate sea water system issues (high sea water inlet temperature, fouled sea chest) or excessive engine load. Requires further investigation.",
                action_type="terminal_unresolved",
                next_nodes={},
                search_queries=["sea water cooling system", "sea chest cleaning"],
                component_context="sea water cooling circuit"
            ),
        }
        
        cls._TREES["cw_failure"] = FaultTree(
            tree_id="cw_failure",
            name="Jacket Cooling Water System Failure Diagnosis",
            subsystem="cooling_water",
            applicable_alarms=["CW_HIGH_TEMP", "ALARM_4401", "CW_LOW_PRESS", "CW_LEVEL_LOW"],
            applicable_symptoms=["high_temp", "leakage"],
            nodes=cw_failure_nodes,
            root_node_id="root"
        )

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
