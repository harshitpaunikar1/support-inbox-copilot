# Support Inbox Copilot

This repository documents a human-in-the-loop email support workflow that reads incoming Gmail messages, classifies intent, drafts responses, and flags urgent cases before anything is sent.

## Domain
E-commerce / Support Ops

## Overview
Reduced repetitive inbox work without giving up review control.

## Methodology
1. Mapped common support cases such as order status, refunds, and shipping delays to decide which emails were safe to assist and which needed escalation.
2. Used n8n to orchestrate Gmail intake, message parsing, AI classification, reply drafting, and downstream review steps without a heavy custom backend.
3. Designed Gemini Flash prompts to separate intent detection from response drafting so the workflow stayed safer on messy customer messages.
4. Added urgency logic and Slack-style alerting for sensitive cases where fast human review mattered more than automation speed.
5. Kept a human approval step in the middle of the flow so the team could edit, approve, or stop replies before sending.
6. Logged outputs and decisions in Google Sheets to preserve an audit trail for missed priorities, bad drafts, and workflow tuning.

## Skills
- n8n
- Gmail Automation
- Gemini Flash
- Human-in-the-Loop Workflow Design
- Slack Alerting
- Google Sheets Logging
- Prompt Design
- Support Operations Automation

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
