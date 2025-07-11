# resume_matcher_app.py

import fitz  # PyMuPDF
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sentence_transformers import SentenceTransformer, util

# NLTK setup
nltk.download('stopwords')
tokenizer = TreebankWordTokenizer()

# Load Sentence-BERT model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    words = tokenizer.tokenize(text)
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Streamlit UI
st.title("📄 Resume Matcher ATS (Semantic AI)")
st.markdown("Upload your **resume (PDF)** and paste the **job description** to get a smart match score using BERT.")

# File uploader
resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Job description input
job_description = st.text_area("Paste Job Description", height=200)

# Submit button
if st.button("Check Match Score"):
    if resume_pdf and job_description.strip() != "":
        try:
            # Step 1: Extract and clean text
            resume_raw_text = extract_text_from_pdf(resume_pdf)
            resume_clean = preprocess_text(resume_raw_text)
            job_clean = preprocess_text(job_description)

            # Step 2: Encode using BERT
            embeddings = model.encode([resume_clean, job_clean], convert_to_tensor=True)

            # Step 3: Compute similarity
            similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            match_percent = similarity_score * 100

            # Step 4: Show result
            st.success(f"✅ Semantic Match Score: **{match_percent:.2f}%**")

            if match_percent > 75:
                st.markdown("🟢 Excellent match! Your resume aligns very well with the job.")
            elif match_percent > 50:
                st.markdown("🟡 Moderate match. Some improvements could help.")
            else:
                st.markdown("🔴 Weak match. Consider tailoring your resume to the job description.")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
    else:
        st.warning("Please upload a resume and enter the job description.")
