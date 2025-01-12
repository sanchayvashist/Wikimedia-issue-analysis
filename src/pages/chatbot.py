import os
import sys

import streamlit as st

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.utils import generate_embeddings, extract_plain_text, get_gemini_response

load_dotenv("src/.env")


# Initialize the Qdrant client
qdrant_client = QdrantClient(url = os.getenv("QDRANT_URL"), api_key= os.getenv("QDRANT_API_KEY"))  # Update with your Qdrant configuration

if __name__ == "__main__":
    load_dotenv()
    st.title("Doctor Droid Bot")
    st.session_state.messages = []
    
    # Get Job discription for the role
    # question = st.text_input("Enter your question about previous production incidents:")
    
    if question := st.chat_input("Enter your question about previous Wikimedia incidents:"):
        hits = qdrant_client.search(
            collection_name="doctor-droid",
            query_vector=generate_embeddings(question)[0],
            limit=1
        )
        
        relevant_file_path = os.path.join("src", hits[0].payload['path'], hits[0].payload['file_name'])
        incident_report = extract_plain_text(relevant_file_path)
        
        replacements = {
        "incident_reports" : incident_report,
        "question": question
        }

        with open("src/chatbot_prompt.txt", "r") as file:
            prompt = file.read()
        
        for placeholder, value in replacements.items():
            prompt = prompt.replace(f'{{{placeholder}}}', value)

        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(question)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response, usage_dict = get_gemini_response(st.session_state.messages)
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "parts": [response]})
                
        