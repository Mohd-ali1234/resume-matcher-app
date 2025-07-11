# 📄 AI Resume Matcher (Streamlit)

This is a simple Streamlit app that compares a resume PDF to a job description and gives a semantic match score using BERT embeddings.

## 🚀 Features
- Upload a resume (PDF)
- Paste job description
- Semantic similarity using Sentence-BERT
- Score shown with color-coded feedback

## 🛠 Tech Stack
- Streamlit
- PyMuPDF
- NLTK
- Sentence Transformers (BERT)

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run resume_matcher_app.py
