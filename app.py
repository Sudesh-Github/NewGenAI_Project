import os
import PyPDF2
import faiss
import torch
import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Load a powerful LLM (e.g., Mistral-7B, GPT-4 API, T5-based model)
llm_model_name = "google/flan-t5-small"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
#llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name, torch_dtype=torch.float16)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a research paper (PDF)."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to chunk text into sections
def chunk_text(text, chunk_size=300):
    """Splits text into sections based on paper structure."""
    sections = {}
    split_text = text.split("\n")
    #chunk_size = min(chunk_size, len(split_text))
    current_section = "Other"
    sections[current_section] = []

    for line in split_text:
        line = line.strip()
        if line.lower().startswith("abstract"):
            current_section = "Abstract"
            sections[current_section] = []
        elif line.lower().startswith("introduction"):
            current_section = "Introduction"
            sections[current_section] = []
        elif line.lower().startswith("conclusion"):
            current_section = "Conclusion"
            sections[current_section] = []
        
        sections[current_section].append(line)

    # Convert sections to chunks
    for section in sections:
        sections[section] = " ".join(sections[section])

    return sections

# Function to create FAISS vector database
def build_vector_database(sections):
    """Builds FAISS vector index for research paper sections."""
    chunk_texts = list(sections.values())
    embeddings = embedding_model.encode(chunk_texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, list(sections.keys()), chunk_texts

# Function to retrieve relevant context
def retrieve_context(query, index, embeddings, section_titles, section_texts, top_k=1):
    """Retrieves most relevant sections for a query."""
    query_embedding = embedding_model.encode([query])
    embeddings = torch.tensor(embeddings)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_contexts = [f"**{section_titles[idx]}**: {section_texts[idx]}" for idx in indices[0]]
    return "\n".join(retrieved_contexts)



# Function to generate a concise answer
def generate_answer_rag(question, context, max_length=512):
    """Truncate input text to prevent exceeding model token limit."""
    input_text = f"Question: {question}\nContext: {context[:max_length]}"
    input_ids = llm_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = llm_model.generate(input_ids, max_length=150)
    return llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)




# Streamlit UI
def main():
    st.title("AI Research Paper RAG Chatbot")

    uploaded_pdf = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        pdf_path = "uploaded_paper.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success("PDF uploaded successfully!")

        # Extract and preprocess text
        text = extract_text_from_pdf(pdf_path)
        text_sections = chunk_text(text)

        # Build FAISS vector database
        index, embeddings, section_titles, section_texts = build_vector_database(text_sections)
        st.write(f"Paper processed into {len(text_sections)} sections for efficient retrieval.")

        # User query input
        user_question = st.text_input("Ask a question about the paper:")
        if user_question:
            context = retrieve_context(user_question, index, embeddings, section_titles, section_texts)
            answer = generate_answer_rag(user_question, context)
            
            st.write(f"**Retrieved Context:**\n{context}")
            st.write(f"**Generated Answer:**\n{answer}")

if __name__ == "__main__":
    main()
