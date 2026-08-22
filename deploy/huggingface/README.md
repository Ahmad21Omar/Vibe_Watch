---
title: Vibewatch
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Describe a mood, get a grounded movie & TV recommendation (RAG)
---

# 🎬 Vibewatch

Describe a **mood or theme** — not a title — and get a recommendation that is *grounded*:
the answer is written only from titles actually retrieved from a vector database, and
those titles are shown next to it so you can check.

Constraints are read out of your sentence: *"funny movies from before 2000"* becomes a
semantic search for **funny** plus hard filters for **movies** and **≤ 1999**. Follow-ups
work too — ask for *"something funnier"* and the earlier request still applies.

**Source, architecture and evaluation numbers:**
👉 https://github.com/Ahmad21Omar/Vibe_Watch

Built with Python, FastAPI, Qdrant, Gemini and LangGraph. Retrieval quality is measured
against a hand-labelled gold set (recall@5 0.83), and answer faithfulness by an
LLM-as-judge that is itself validated against a planted hallucination.
