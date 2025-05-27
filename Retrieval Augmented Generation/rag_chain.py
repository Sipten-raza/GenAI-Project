from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

from transformers import pipeline
import torch

def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore

def get_qa_chain(vectorstore):
    # Load a local transformer model for text generation
    local_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",   # Or you can try "tiiuae/falcon-rw-1b"
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=local_pipeline)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain
