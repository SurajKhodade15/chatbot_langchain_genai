import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import traceback

# Load environment variables
load_dotenv()

# Configure LangChain tracking if enabled
if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Updated to use the correct v2 tracing variable
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    print("LangSmith tracing enabled with project:", os.getenv("LANGCHAIN_PROJECT"))

# Set page configuration
st.set_page_config(
    page_title="AI Chatbot with LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .user .avatar {
        background-color: #6c757d;
        color: white;
    }
    .assistant .avatar {
        background-color: #0d6efd;
        color: white;
    }
    .chat-message .content {
        flex: 1;
    }
    .stTextInput {
        margin-bottom: 0.5rem;
    }
    .stButton {
        margin-top: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_api_key_valid" not in st.session_state:
    st.session_state.is_api_key_valid = False

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
If you don't know the answer to a question, don't share false information."""

# Available models mapping with descriptions
AVAILABLE_MODELS = {
    "gemma2-9b-it": "Gemma 2 9B IT - Most powerful model, best quality responses",
    "llama3-8b-8192": "Llama 3 8B - Efficient model, good for simpler tasks",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B - Excellent for detailed reasoning and analysis"
}

def display_chat_message(role: str, content: str):
    """Display a chat message with formatting based on the role."""
    with st.container():
        col1, col2 = st.columns([1, 11])
        
        avatar_icon = "👤" if role == "user" else "🤖"
        background_class = "user" if role == "user" else "assistant"
        
        with col1:
            st.markdown(f"""
            <div class="avatar {background_class}">
                {avatar_icon}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="chat-message {background_class}">
                <div class="content">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)

def generate_response(question: str, api_key: str, model_name: str, 
                     temperature: float, max_tokens: int, system_prompt: str) -> str:
    """
    Generate a response from the LLM based on the given parameters.
    
    Args:
        question: The user's question
        api_key: Groq API key
        model_name: Name of the LLM model to use
        temperature: Temperature setting for response creativity
        max_tokens: Maximum tokens in the response
        system_prompt: System prompt to guide the assistant's behavior
        
    Returns:
        Generated response text
    """
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            ('user', '{question}')
        ])
        
        # Create model with error handling
        model = ChatGroq(
            model=model_name, 
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            tags=["chatbot", model_name],  # Add tags for better organization in LangSmith
        )
        
        # Create chain and generate response
        parser = StrOutputParser()
        
        # Import LangChain tracing components
        from langchain.callbacks.tracers import LangChainTracer
        from langchain.callbacks.manager import CallbackManager
        
        # Set up tracing if LangSmith API key is present
        if os.getenv("LANGCHAIN_API_KEY"):
            tracer = LangChainTracer(
                project_name=os.getenv("LANGCHAIN_PROJECT", "Groq Chatbot GenAI")
            )
            callback_manager = CallbackManager([tracer])
            
            # Create chain with tracing enabled
            chain = (
                prompt 
                | model.with_config(callbacks=[tracer], tags=[f"model:{model_name}", "production"]) 
                | parser
            )
        else:
            # Create standard chain without tracing
            chain = prompt | model | parser
        
        with st.spinner("Thinking..."):
            response = chain.invoke({'question': question})
        
        return response
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return "❌ Error: Invalid API key. Please check your Groq API key and try again."
        elif "429" in error_msg:
            return "❌ Error: Rate limit exceeded. Please wait a moment and try again."
        else:
            st.error(f"An error occurred: {error_msg}")
            return f"❌ Error: Something went wrong while generating the response. Technical details: {error_msg}"

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # API Key input with validation
    api_key = st.text_input("Enter your Groq API Key:", 
                           value=os.getenv("GROQ_API_KEY", ""),
                           type="password", 
                           help="Get your API key from https://console.groq.com/")
    
    # Only enable model selection if API key is provided
    if api_key:
        st.session_state.is_api_key_valid = True
    else:
        st.session_state.is_api_key_valid = False
    
    # Model selection
    model_options = list(AVAILABLE_MODELS.items())
    selected_model_index = st.selectbox(
        "Select LLM model:",
        options=range(len(model_options)),
        format_func=lambda i: f"{model_options[i][0]} - {model_options[i][1]}"
    )
    selected_model = model_options[selected_model_index][0]
    
    # Advanced options in an expander
    with st.expander("Advanced Settings"):
        # Temperature slider
        temperature = st.slider(
            "Temperature (creativity):", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            help="Higher values make output more creative, lower values make it more deterministic"
        )
        
        # Max tokens slider
        max_tokens = st.slider(
            "Maximum response length:", 
            min_value=50, 
            max_value=4096, 
            value=800,
            help="Maximum number of tokens in the response"
        )
        
        # Custom system prompt
        system_prompt = st.text_area(
            "System prompt:",
            value=DEFAULT_SYSTEM_PROMPT,
            height=150,
            help="Instructions that define how the AI assistant behaves"
        )
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Add credits
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This chatbot uses LangChain with Groq AI.
    
    Built with:
    - Streamlit
    - LangChain
    - Groq API
    """)

# Main chat interface
st.title("🤖 AI Chatbot with LangChain")
st.markdown("Ask anything and get instant responses powered by AI!")

# Display chat history
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# Input area
user_input = st.chat_input("Type your message here..." if st.session_state.is_api_key_valid else "Enter API key in sidebar first...")

# Process user input
if user_input and st.session_state.is_api_key_valid:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)
    
    # Generate and display assistant response
    try:
        with st.spinner("Thinking..."):
            response = generate_response(
                user_input, 
                api_key, 
                selected_model, 
                temperature, 
                max_tokens,
                system_prompt
            )
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_chat_message("assistant", response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"❌ Error: {str(e)}"
        })
        display_chat_message("assistant", f"❌ Error: {str(e)}")

# Show API key warning if not provided
elif user_input and not st.session_state.is_api_key_valid:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
    
# First-time instructions
elif not st.session_state.messages:
    st.info("👋 Welcome! Enter your Groq API key in the sidebar and start chatting with the AI assistant.")