import streamlit as st
from app import API_Chatbot
import time
from typing import List, Dict
import loggin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config and styling
st.set_page_config(
    page_title="CrustData API Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChat {
        padding-bottom: 100px;
    }
    .stSpinner {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_chatbot() -> API_Chatbot:
    """Initialize the API chatbot with error handling."""
    try:
        return API_Chatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.error("Please check your environment variables and credentials.")
        st.stop()

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()

def display_chat_header():
    """Display the chat interface header."""
    st.title("🤖 CrustData API Chatbot")
    st.markdown("""
        Welcome! I can help you with:
        - Understanding API endpoints
        - Finding specific API documentation
        - Explaining API parameters and usage
    """)

def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_user_input(prompt: str):
    """Process user input and generate response."""
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                # Add slight delay for better UX
                time.sleep(0.5)
                reply = st.session_state.chatbot.call_llm(prompt)
                
                # Display response with typing effect
                message_placeholder = st.empty()
                full_response = ""
                for chunk in reply.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        st.error("I encountered an error processing your request. Please try again.")

def add_sidebar():
    """Add sidebar with additional information."""
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot helps you navigate API documentation using:
        - Vector similarity search (Qdrant)
        - Natural language processing (Gemini)
        - Semantic search
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def main():
    """Main application function."""
    try:
        initialize_session_state()
        display_chat_header()
        add_sidebar()
        
        # Display chat history
        display_chat_messages()
        
        # Handle user input
        if prompt := st.chat_input("Ask about the API..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the input
            process_user_input(prompt)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()
