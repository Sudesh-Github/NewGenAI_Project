
#--------------------------------------------------------

from dotenv import load_dotenv  # type: ignore
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import docx  
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_chat import message  

st.set_page_config(page_title="BANK_RISK_CONTROLLER_SYSTEMS")


def run_chatbot():

    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    st.header("CHATBOT for PDF Reader")

    # File uploader and process button below the header
    st.image("D:/myproject/Final_Project/Final_Bank_Project/ChatBot/chatbot1.png", caption="Ask Your PDF")
    uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
    process = st.button("Process")

    # Optional: Add logic to handle the uploaded files and process button click
    if process:
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} file(s)...")
            # Add your file processing logic here
        else:
            st.warning("Please upload at least one file before processing.")

    if process:
        files_text = get_files_text(uploaded_files)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        # create vector stores
        vectorstore = get_vectorstore(text_chunks)
        # create conversation chain
        st.session_state.conversation = get_rag_conversation_chain(vectorstore)

        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)


def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page_num in range((len(pdf_reader.pages))):  # Extract text from first 6 pages
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text


def get_csv_text(file):
    return "a"


def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base


def get_rag_conversation_chain(vectorstore):
    handler = StdOutCallbackHandler()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 64})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Remove the qa_chain argument from from_llm
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        callbacks=[handler]
    )
    return conversation_chain

    


def handel_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == "__main__":
    run_chatbot()