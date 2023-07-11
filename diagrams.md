# Support Inbox Copilot Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## n8n workflow diagram

```mermaid
flowchart TD
    N1["Step 1\nMapped common support cases such as order status, refunds, and shipping delays to "]
    N2["Step 2\nUsed n8n to orchestrate Gmail intake, message parsing, AI classification, reply dr"]
    N1 --> N2
    N3["Step 3\nDesigned Gemini Flash prompts to separate intent detection from response drafting "]
    N2 --> N3
    N4["Step 4\nAdded urgency logic and Slack-style alerting for sensitive cases where fast human "]
    N3 --> N4
    N5["Step 5\nKept a human approval step in the middle of the flow so the team could edit, appro"]
    N4 --> N5
```

## Intent classification flow

```mermaid
flowchart LR
    N1["Inputs\nPrompt variants, evaluation examples, and scoring notes"]
    N2["Decision Layer\nIntent classification flow"]
    N1 --> N2
    N3["User Surface\nOperational artifact surface such as sheets, prompt libraries, or playbooks"]
    N2 --> N3
    N4["Business Outcome\nSLA adherence"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
