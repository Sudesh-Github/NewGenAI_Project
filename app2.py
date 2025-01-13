import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import faiss
import streamlit as st

# Configure pytesseract and poppler
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\DELL\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin'

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF using pdf2image."""
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"page_{i + 1}.png"
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    st.write(f"Extracted {len(images)} pages from the PDF.")
    return image_paths

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def analyze_diagram(image_path):
    """Analyze diagram in an image to extract dimensions."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use edge detection and contours for dimension extraction
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Placeholder: Replace this with logic to extract specific dimensions
    dimensions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        dimensions.append((x, y, w, h))
    return dimensions

def create_embeddings(text_data, model):
    """Generate embeddings for the extracted text."""
    embeddings = model.encode(text_data, convert_to_tensor=True)
    return embeddings

def build_vector_database(texts, model):
    """Build a FAISS-based vector database for text embeddings."""
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_context(query, index, embeddings, texts, model, top_k=3):
    """Retrieve the most relevant context for a query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_contexts = [texts[idx] for idx in indices[0]]
    return " ".join(retrieved_contexts)

def answer_question_rag(question, context, qa_pipeline):
    """Answer a question using the RAG approach."""
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

def main():
    st.title("PDF Diagram and RAG Chatbot Assistant")
    
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        st.session_state.uploaded_pdf = True 
        pdf_path = "uploaded_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("PDF uploaded successfully!")

    if st.session_state.get('uploaded_pdf', False):
        st.subheader("Extracted Pages:")
        images = extract_images_from_pdf(pdf_path)

        st.subheader("Chatbot")
        selected_page = st.selectbox("Select Page for Query", list(range(1, len(images) + 1)))
        user_question = st.text_input("Ask a question:")

        if user_question:
            # Extract text from the selected page
            selected_page_text = extract_text_from_image(f"page_{selected_page}.png") 

            # Create embeddings for the selected page text
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode([selected_page_text]) 
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            # Retrieve context (in this case, the selected page text)
            context = retrieve_context(user_question, index, embeddings, [selected_page_text], model)

            # Answer the question using RAG
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            answer = answer_question_rag(user_question, context, qa_pipeline)
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()