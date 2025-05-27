import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document

# Set page config
st.set_page_config(page_title="RAG App with Streamlit", layout="wide")

st.title("RAG Chat with Your Documents")

# Upload files
uploaded_files = st.file_uploader("Upload your documents (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Function to extract text
def extract_text(file):
    file_type = file.type
    text = ""
    if file_type == "application/pdf":
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_type == "text/plain":
        text = str(file.read(), "utf-8")
    return text

# Store all combined text
all_text = ""

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    for file in uploaded_files:
        st.markdown(f" `{file.name}`")
        text = extract_text(file)
        all_text += text + "\n"

    st.markdown("### Extracted Document Preview:")
    st.text_area("Combined Text", all_text[:3000], height=300)  # just show first 3000 chars

from rag_chain import create_vector_store, get_qa_chain

# Process and build vector store
if all_text:
    st.markdown("---")
    st.subheader("Ask Questions from Documents")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Processing..."):
            vs = create_vector_store(all_text)
            qa_chain = get_qa_chain(vs)
            response = qa_chain.run(query)
        st.success("Answer:")
        st.write(response)
