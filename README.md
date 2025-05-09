GenAI Math Tutor Chatbot with RAG (LangChain + Groq)
This project is an AI-powered Math Tutor Chatbot built using LangChain, the Groq API, and Streamlit. It is specifically designed to answer math-related questions only, including topics from high school to advanced math. It uses Retrieval-Augmented Generation (RAG) to enhance its accuracy by referencing uploaded documents and maintains conversational memory for coherent interactions.

🎯 Features
🧠 Math-Focused Responses: Specially designed to answer only math-related questions.

📄 Document Upload for RAG: Supports uploading .pdf or .txt files to enhance response accuracy through vector-based document retrieval.

🔁 Conversational Memory: Remembers previous math questions and answers during a session.

🌐 Web Search Fallback: Uses Google Serper API to fetch math-related answers when no document is provided.

🚀 Groq API Integration: Utilizes Groq's ultra-fast LLM for generating responses.

🖥️ Streamlit UI: A clean and interactive frontend for easy usage.

📦 Requirements
Python 3.7+

streamlit

langchain

langchain_groq

langchain_community

python-dotenv

faiss-cpu

sentence-transformers
