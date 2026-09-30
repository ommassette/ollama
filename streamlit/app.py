import logging
from typing import List, Dict, Optional
import streamlit as st
from ollama import chat, generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = "saimon:latest"

st.set_page_config(page_title="Saimon Chatbot")
st.title("Saimon AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


def summarize_text(text: str) -> Optional[str]:
    try:
        logging.info("Summarizing input text...")
        prompt = f"Summarize the following text clearly and concisely:\n\n{text}"
        
        response = generate(
            model=MODEL,
            prompt=prompt,
            options={'temperature': 0.0}
        )
        
        summary = response.get('response', '')
        logging.info("Summary generated successfully.")
        return summary
    except Exception as e:
        logging.error("Failed to generate summary: %s", e)
        return None


def summarize_conversation(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages:
        logging.warning("No conversation history to summarize.")
        return None

    try:
        logging.info("Summarizing full conversation context...")
        summary_messages = messages + [
            {'role': 'user', 'content': 'Summarize our conversation so far into bullet points.'}
        ]

        response = chat(
            model=MODEL,
            messages=summary_messages,
            options={'temperature': 0.0},
            stream=False
        )

        summary = response.get('message', {}).get('content', '')
        logging.info("Conversation summary generated successfully.")
        return summary
    except Exception as e:
        logging.error("Failed to summarize conversation context: %s", e)
        return None


with st.sidebar:
    st.header("Actions")
    if st.button("Summarize Chat"):
        with st.spinner("Summarizing conversation..."):
            chat_summary = summarize_conversation(st.session_state.messages)
            if chat_summary:
                st.info(chat_summary)
            else:
                st.warning("No chat history to summarize or request failed.")
                
    st.divider()
    st.subheader("Summarize Specific Text")
    target_text = st.text_area("Enter text to summarize:")
    if st.button("Summarize Text"):
        if target_text.strip():
            with st.spinner("Summarizing text..."):
                text_summary = summarize_text(target_text)
                if text_summary:
                    st.success(text_summary)
                else:
                    st.error("Failed to summarize text.")
        else:
            st.warning("Please enter text first.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Saimon something...")

if user_input:
    logging.info("User Prompt: %s", user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = chat(
                model=MODEL,
                messages=st.session_state.messages,
                options={'temperature': 0.0},
                stream=True,
            )

            for chunk in stream:
                content = chunk.get('message', {}).get('content', '')
                full_response += content
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            logging.info("Saimon's Response:\n%s", full_response)
            
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            logging.error("Failed to execute chat stream: %s", e)
            st.error("Failed to generate response.")