# LangChain Chatbot using Groq API + Streamlit with Memory + RAG

import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# Set up Streamlit UI
st.set_page_config(page_title="GenAI & ML Chatbot", layout="centered")
st.title("🤖 GenAI & ML Chatbot with RAG (LangChain + Groq)")

# Upload and process file
uploaded_file = st.file_uploader("Upload a .txt file for RAG (optional)", type=["pdf"])
docs = []
if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(uploaded_file.getvalue().decode("latin1"))
            tmp_path = tmp_file.name
        loader = TextLoader(tmp_path, encoding="latin1")
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(raw_docs)
    except Exception as e:
        st.error(f"Error loading document: {e}")

# Create vectorstore if docs are present
retriever = None
if docs:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

serper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
# Create Conversational RAG chain if retriever is available
if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=False
    )
else:
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
You are an expert AI assistant in Generative AI, ML, and Deep Learning.
You remember past conversations to give better answers.

{chat_history}

User: {question}
AI:
"""
    )
    qa_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Ask me anything about GenAI, ML, or DL:", key="user_input")
if user_input:
    response = None
    if retriever:
        response = qa_chain({"question": user_input})["answer"]
    else:
        try:
            response = serper.run(user_input)
        except Exception as e:
            st.error(f"Web search failed: {e}")
            response = qa_chain({"question": user_input})["text"]
    
    st.session_state.chat_history.append((user_input, response))

# Display history
if st.session_state.chat_history:
    for i, (user_q, bot_a) in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {bot_a}")
        st.markdown("---")
