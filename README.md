# AI-Powered Thumbnail & Title Recommendation System for YouTube Videos

An end-to-end system that learns patterns from high-performing mid-to-long YouTube videos (thumbnail visuals + on-thumbnail text + titles) and generates **recommendations** and **draft thumbnails** for new uploads. The goal is to help creators produce thumbnails/titles that are clear, compelling, and consistent with patterns observed in viral or high-engagement content.

> Note: This project is intended for *ethical* optimization (clarity, relevance, accessibility). It does **not** aim to mislead viewers.

---

## ✨ What This Project Does

Given a YouTube video (or metadata + frames), the system can:
- **Analyze viral thumbnail patterns** (composition, colors, faces/emotions, object presence, contrast, text density).
- **Extract & model text within thumbnails** using PaddleOCR (font size proxies, placement, phrase length).
- **Recommend titles** using LLM-driven suggestions, conditioned on topic/keywords/style.
- **Generate draft thumbnails** by selecting best frames, applying learned layout heuristics, and optionally overlaying concise text.
- **Score and rank** multiple thumbnail/title candidates using a learned “engagement-likeness” predictor.

---

## 🧠 Core Idea

We treat a successful YouTube package as a joint optimization problem:

**Thumbnail (visual + text) + Title (language) + Topic + Audience signals → Engagement outcome**

This repo builds:
1) a **pattern mining pipeline** to learn what high-performing videos tend to look like, and  
2) a **generation + ranking pipeline** to propose and select candidates for a new video.

---

## 🔥 Features

### Thumbnail Understanding
- Frame sampling and keyframe selection
- Thumbnail-style feature extraction (contrast, saturation, color harmony, sharpness)
- Face detection & emotion cues (optional)
- Object detection / scene cues (optional)
- OCR for on-thumbnail text (words, length, placement, density)

### Title Understanding
- Topic extraction (keywords/entities)
- Style clustering (e.g., “how-to”, “explainer”, “storytime”, “reaction”)
- LLM-based title suggestions with constraints (length, tone, clarity)

### Generation & Ranking
- Candidate generation (multiple thumbnails + text overlays)
- Candidate scoring (predictive model / heuristic hybrid)
- Human-in-the-loop selection (recommended final)

---

## 🏗️ System Architecture (High Level)

**Data Collection**
- Pull video metadata (title, views, likes, duration, upload date, channel stats)
- Download thumbnails + sample frames
- Label as “high-performing” using thresholds or normalized metrics (e.g., views per day)

**Pattern Mining**
- Vision features: composition, color, faces/objects, text density
- Text features: OCR phrases, length, sentiment, capitalization patterns
- Title features: length, readability, entities, sentiment, structure

**Modeling**
- Similarity search (find closest viral references)
- Predictive ranking model for candidates (binary/continuous proxy of engagement)
- LLM-based title generator + rule-based filters

**Output**
- Top-N recommended titles
- Top-N generated thumbnails
- “Why this works” explanation (interpretable features)

---

## 📦 Repo Structure (suggested)

