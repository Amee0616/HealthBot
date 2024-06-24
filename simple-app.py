import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import boto3
from urllib.parse import urlparse
from pinecone import Pinecone
import pinecone
from langchain_openai import ChatOpenAI
import openai
from langchain.chains import LLMChain, RetrievalQA
import time
import re
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uuid
import warnings
from gtts import gTTS
import base64
import speech_recognition as sr
import tempfile

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Custom Chatbot", layout="wide")
st.title("HealthBot: The Insightful Retrieval Companion")

# Function to generate pre-signed URL
def generate_presigned_url(s3_uri):
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600
    )
    return presigned_url

# Function to retrieve documents, generate URLs, and format the response
def retrieve_and_format_response(query, retriever, llm, chat_history):
    docs = retriever.get_relevant_documents(query)
    
    formatted_docs = []
    for doc in docs:
        content_data = doc.page_content
        s3_uri = doc.metadata['id']
        s3_gen_url = generate_presigned_url(s3_uri)
        formatted_doc = f"{content_data}\n\n[More Info]({s3_gen_url})"
        formatted_docs.append(formatted_doc)
    
    combined_content = "\n\n".join(formatted_docs)
    
    # Create a prompt for the LLM to generate an explanation based on the retrieved content
    prompt = f"Instruction: You are a helpful assistant to help users with their patient education queries. \
               Based on the following information, provide a summarized & concise explanation using a couple of sentences. \
               Only respond with the information relevant to the user query {query}, \
               if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
               But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
               In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info Download]({s3_gen_url}) \
               Context: {combined_content} \
               Chat History: {chat_history}"
    
    # Originally there were no message
    message = HumanMessage(content=prompt)

    response = llm([message])
    return response

# Function to save chat history to a file
def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        file.write(history)

# Function to upload the file to S3
def upload_file_to_s3(bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)

# Function to get chat history as text
def get_chat_history_text(messages):
    chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    return chat_history_text

# Function to convert text to speech and return the audio file URL
def text_to_audio(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f"data:audio/mp3;base64,{audio_base64}"

# Function to convert audio to text
def audio_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Sorry, I could not understand the audio."
    except sr.RequestError:
        text = "Sorry, there was an issue with the speech recognition service."
    return text

# Setup - Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Langchain stuff
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Initialize the conversation memory
memory = ConversationBufferMemory()

prompt_template = ChatPromptTemplate.from_template(
        "Instruction: You are a helpful assistant to help users with their patient education queries. \
        Based on the following information, provide a summarized & concise explanation using a couple of sentences. \
        Only respond with the information relevant to the user query {query}, \
        if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
        But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
        In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info]({s3_gen_url}) \
        Context: {combined_content}"
    )

# Initialize necessary objects (s3 client, Pinecone, OpenAI, etc.)
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)
# PINECONE
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "test"
openai.api_key = OPENAI_API_KEY

# Set up LangChain objects
# VOYAGE AI
model_name = "voyage-large-2"  
embedding_function = VoyageAIEmbeddings(
    model=model_name,  
    voyage_api_key=VOYAGE_AI_API_KEY
)
# Initialize the Pinecone client
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name=index_name
)
retriever = vector_store.as_retriever()

# Initialize rag_chain
rag_chain = (
    {"retrieved_context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar for chat history and download button
st.sidebar.title("Chat History")
for i, message in enumerate(st.session_state["messages"]):
    with st.sidebar.expander(f"Message {i+1} - {message['role']}"):
        st.write(message["content"])

# Download chat history as text file
if st.sidebar.button("Download Chat History"):
    chat_history_text = get_chat_history_text(st.session_state["messages"])
    st.sidebar.download_button(
        label="Download",
        data=chat_history_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input as text or audio
user_input_text = ""
user_input_audio = st.file_uploader("Upload your audio message", type=["wav", "mp3"])

if user_input_audio:
    with st.spinner("Processing audio..."):
        user_input_text = audio_to_text(user_input_audio)
        st.markdown(f"**You (transcribed):** {user_input_text}")

if user_input_text:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input_text})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input_text)
    
    # Generate and display bot response
    with st.spinner("Thinking..."):
        # Compile the chat history
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
        
        bot_response = retrieve_and_format_response(user_input_text, retriever, llm, chat_history).content
        
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        audio_url = text_to_audio(bot_response)
        st.audio(audio_url, format="audio/mp3")
