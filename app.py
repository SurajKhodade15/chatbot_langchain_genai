import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmit tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING"] = 'true'


## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Please respond to the user query.'),
        ('user', 'Question: {question} ')
    ]
)

def generate_response(question, api_key, llm, temperature, token_limit):
    ChatGroq.groq_api_key = api_key  # Use the API key provided by the user
    ChatGroq.groq_llm = llm
    ChatGroq.groq_temperature = temperature
    ChatGroq.groq_token_limit = token_limit
    model = ChatGroq(model = llm)
    parser = StrOutputParser()
    chain = prompt | model | parser
    response = chain.invoke({'question': question})
    return response

## #Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")



## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq AI API Key:",type="password")

## Select the Groq AI model
engine=st.sidebar.selectbox("Select Groq AI model",["Gemma2-9b-It","compound-beta","deepseek-r1-distill-llama-70b"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter the Groq AI API Key in the sidebar")
else:
    st.write("Please provide the user input")