# MaritimeMind AI - Demo Script

This document contains a set of curated demo queries designed to showcase the multimodal retrieval capabilities of the MaritimeMind AI platform.

## Multimodal Queries (Retrieves Text + Images)

These queries trigger the `DIAGRAM_REQUEST`, `PROCEDURE`, or `TROUBLESHOOTING` intents, instructing the agent to pull both text instructions and associated engineering diagrams or schematics.

1. **Diagram Intent (Engineering & Cooling Water)**
   > *"Show me the cooling water piping diagram."*
   > **Expected System Behavior:** Triggers `DIAGRAM_REQUEST`. Prioritizes retrieving the visual schematic for the cooling water system and supplements it with explanatory text.

2. **Procedure Intent (Engineering & Main Engine)**
   > *"What is the procedure to overhaul the main engine cylinder?"*
   > **Expected System Behavior:** Triggers `PROCEDURE`. Retrieves the step-by-step text procedure and also pulls any referenced component diagrams or exploded views.

3. **Troubleshooting Intent (Engineering & Lube Oil)**
   > *"Troubleshoot low lube oil pressure alarm."*
   > **Expected System Behavior:** Triggers `TROUBLESHOOTING`. Initiates the diagnostic workflow, providing potential causes, text-based checks, and relevant system schematics.

4. **Procedure Intent (Deck & Ballast System)**
   > *"How do I replace the ballast pump?"*
   > **Expected System Behavior:** Triggers `PROCEDURE`. Retrieves text chunks with safety warnings and step-by-step procedures, along with associated pump diagrams.

5. **Diagram Intent (Navigation & Steering Gear)**
   > *"Can you provide the schematic for the steering gear?"*
   > **Expected System Behavior:** Triggers `DIAGRAM_REQUEST`. The system executes a CLIP-based cross-modal search and a metadata expansion search to find the steering gear layout diagram.

## Text-Only Queries (Retrieves Text)

These queries trigger `EXPLANATION` or `SOP_LOOKUP` intents, returning only grounded text.

6. **SOP Lookup Intent**
   > *"What are the MARPOL compliance regulations for oil discharge?"*
   > **Expected System Behavior:** Triggers `SOP_LOOKUP`. Uses pure vector + BM25 text search to retrieve standard operating procedures and regulations.

7. **Explanation Intent**
   > *"Explain the purpose of the ship's economizer."*
   > **Expected System Behavior:** Triggers `EXPLANATION`. Retrieves descriptive text explaining the component's function.
