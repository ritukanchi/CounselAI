import streamlit as st
import os
from dotenv import load_dotenv
from src.generate_response import get_response
from qdrant_client import QdrantClient
import asyncio

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_cloud_url = os.getenv('QDRANT_CLOUD_URL')
qdrant_cloud_api_key = os.getenv('QDRANT_CLOUD_API_KEY')

def chat_ui():
    st.set_page_config(page_title="Legal chatbot", layout='wide')

    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
           display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("RAG Legal Chatbot")
    st.markdown("*0.01v beta*")
    st.markdown("**Llama3.2 by Groq**")
    st.warning("This Chatbot can make mistakes. Verify responses before any action.")

    # Sidebar with examples
    st.sidebar.title("Example queries to try")
    queries = [
        "How can a woman travel in India alone?",
        "A 22-year-old girl was raped by men in Kolkata RG Kar Medical College.",
        "What are the laws of foreign direct investment in Stock market?",
        "A random guy was staring at me in the metro and attempted to take my photo.",
        "Which cities in India has most rape cases?",
        "Hey! Someone's dog bite me."
    ]
    for query in queries:
        st.sidebar.write(f"- {query}")

    # Initialize chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process new user query
    def process_query(query):
        st.session_state.chat_history = []  # Reset for fresh conversation
        response = get_response(query, st.session_state.chat_history)
        st.session_state.chat_history.append({"human": query, "ai": response['output']})
        return response['output']

    try:
        if prompt := st.chat_input("Ask a query: "):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = process_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    except Exception as e:
        st.error("Something went wrong! Chatbot can't work at this time.")
        st.write(str(e))

if __name__ == '__main__':
    chat_ui()
