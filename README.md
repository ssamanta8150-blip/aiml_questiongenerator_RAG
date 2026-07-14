# AI-Powered Current Affairs Question Generator

An intelligent Flask-based web application that automatically generates high-quality competitive examination questions from PDFs, URLs, and current affairs sources using Large Language Models (LLMs).

The system combines **Google Gemini**, **Groq**, **semantic similarity detection**, **automatic web scraping**, and a persistent question bank to produce difficult, UPSC-style multiple-choice questions while preventing duplicates.

---

# Features

* 📄 Upload PDFs or scrape articles directly from URLs
* 🤖 AI-generated MCQs using Google Gemini
* 🧠 Semantic duplicate detection using embeddings
* 📚 Persistent Question Bank with automatic updates
* 📅 Automatic scraping of daily current affairs
* 🔍 Supports Daily, Weekly, and Monthly databases
* ⚡ Parallel question generation using multiple Gemini API keys
* 📑 Download the accumulated Question Bank as a PDF
* 🌐 Flask-based web interface
* 💾 SQLite database backend
* 🔄 Background processing for question banking

---

# Project Architecture

```text
                   User
                     │
                     ▼
             Flask Web Interface
                     │
     ┌───────────────┼───────────────┐
     │               │               │
PDF Upload      URL Scraper     Auto Scraper
     │               │               │
     └───────────────┴───────────────┘
                     │
             Daily / Weekly / Monthly
                 SQLite Database
                     │
                     ▼
           Parallel Gemini Engines
         (Multiple API Keys / Threads)
                     │
                     ▼
            AI Generated Questions
                     │
         Semantic Duplicate Detection
        (Gemini Embeddings + Cosine Similarity)
                     │
                     ▼
            Persistent Question Bank
                     │
                     ▼
         Downloadable Question Bank PDF
```

---

# Major Components

## 1. Content Ingestion

The application accepts content from:

* PDF files
* Website URLs
* Automatically scraped current affairs portals

Extracted text is categorized into:

* Daily
* Weekly
* Monthly

databases.

---

## 2. Automated Current Affairs Scraper

The application periodically scrapes current affairs content from supported sources such as:

* PIB (Press Information Bureau)
* Vajiram & Ravi

and stores them automatically inside the Daily database.

---

## 3. AI Question Generation

Questions are generated using multiple Google Gemini API keys running in parallel.

Supported formats include:

* Statement-based questions
* Assertion–Reason questions
* Match the Following
* Conceptual questions
* Multiple statement ("How many are correct") questions

The generated questions follow UPSC-style patterns.

---

## 4. Duplicate Detection

Instead of checking only exact text matches, every question is converted into an embedding.

Duplicate detection combines:

* String normalization
* Semantic embeddings
* Cosine similarity

Only genuinely unique questions are stored inside the Question Bank.

---

## 5. Persistent Question Bank

Every accepted question stores:

* Question
* Options
* Correct answer
* Explanation
* Embedding vector

The Question Bank is maintained as a JSON database and can be exported as a professionally formatted PDF.

---

# Technologies Used

## Backend

* Flask
* SQLAlchemy
* WTForms

## Artificial Intelligence

* Google Gemini
* Gemini Embeddings
* Groq

## Machine Learning

* NumPy
* Scikit-learn

## Web Scraping

* Requests
* BeautifulSoup
* DuckDuckGo Search

## PDF Processing

* PyPDF
* FPDF

## Database

* SQLite

---

# Project Structure

```text
project/
│
├── app.py
├── templates/
├── static/
├── scraper_config.json
├── Question Bank/
│      └── bank_data.json
│
├── data.sqlite
├── requirements.txt
└── README.md
```

---

# Workflow

1. Upload PDFs or enter website URLs.
2. Content is extracted and stored in the selected database.
3. User selects data sources.
4. Multiple Gemini engines generate examination questions in parallel.
5. Questions are checked against the semantic Question Bank.
6. Unique questions are permanently stored.
7. Users can export the accumulated Question Bank as a PDF.

---

# Key Features

* Multi-threaded question generation
* Automatic duplicate removal
* Persistent knowledge base
* Live current affairs scraping
* UPSC-oriented question styles
* Semantic similarity filtering
* Parallel AI inference
* PDF export functionality

---

# Future Improvements

* User authentication
* Difficulty estimation using machine learning
* Automatic syllabus tagging
* Support for additional competitive examinations
* REST API for external integrations
* Docker deployment
* Cloud database support
* Vector database integration (FAISS/ChromaDB)
* RAG-enabled answer explanations

---

# Disclaimer

This project is intended for educational and research purposes. AI-generated questions should be reviewed before use in formal examinations.

---


