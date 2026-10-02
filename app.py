# import libraries
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from typing import List, Dict, Any

#load environment variables
load_dotenv()

#app configurations
st.set_page_config(page_title="Contract RAG Chatbot",
                   page_icon="💬",  # speech bubble icon
                   layout="centered")

#add title to the app
st.title("Contract RAG Chatbot")

#Add description to the app
st.markdown("** Your Inteligent Contract RAG Chatbot**")
st.divider()

#add collapsible section
with st.expander("❗ About the Contract RAG Chatbot", expanded=False):
    st.markdown(
        """
        🤖 **Contract Rag Chatbot**

        🧠 Model gpt-5 via openAI Response API

        📂 File Search tool using your pre-built Vector Store

        ✨ **Features**
        - 💬 multi turn chat
        - 🖼️ image input
        - 🧹 clear conversation

        🔐 **Secrets**  
        read openai api key and vector store id from .env file
        """
    )

#retrieve the credentials
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
vector_store_id = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]

#Set the openai key in the os
os.environ["OPENAI_API_KEY"] = openai_api_key

#initialize the OpenAI client
client = OpenAI()

#warn if openai key or the vector store id is not set
if not openai_api_key:
    st.warning("OpenAI API key is not set. Please set the OPENAI_API_KEY in the .env file or Streamlit secrets.")

if not vector_store_id:
    st.warning("Vector store ID is not set. Please set the VECTOR_STORE_ID in the .env file or Streamlit secrets.")

#configuration of the system prompt:
system_prompt = """
You are a toxic HR  manager who likes to learn the employment contacts thoroughly
"""

#Store the previous response id
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

#initialise the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_widget_version" not in st.session_state:
    st.session_state.upload_widget_version = 0

# Create a sidebar with user control
with st.sidebar:
    st.header("User Controls")
    st.divider()

    # Clear the conversation history - reset chat history and context
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.session_state.upload_widget_version += 1
        st.rerun()

# Helper functions:
def build_input_parts(text:str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the input parts array for the OpenAI from text and uploaded files.
    args:
        text: text to be sent to the openAI
        files: The images and PDFs to be sent to OpenAI
    returns:
        a list of input parts compatible with the OpenAI responses API
    """
    content = []
    if text and text.strip():
        content.append({"type": "input_text", "text": text.strip()}) 

    for file in files:
        if file["mime_type"] == "application/pdf":
            content.append({
                "type": "input_file",
                "filename": file["filename"],
                "file_data": file["data_url"],
            })
        else:
            content.append({
                "type": "input_image",
                "image_url": file["data_url"],
                "detail": "auto",
            })
    return [{"type": "message", "role": "user", "content": content}] if content else []

# Function to generate a response from the OpenAI responses API
def call_responses_api(parts:list[Dict[str, Any]], previous_response_id:str=None) -> Any:
    """
    Call the OpenAI responses API with the given input parts and previous response ID.
    args:
        parts: The input parts to be sent to the OpenAI
        previous_response_id: The ID of the previous response for context
    returns:
        The response from the OpenAI responses API
    """
    tools = [
        { "type": "file_search", "vector_store_ids": [vector_store_id], "max_num_results": 20}
    ]
    response = client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=system_prompt,
        tools = tools,
        previous_response_id=previous_response_id,
    
    )
    return response

# Function to get the text output
def get_text_output(response:Any) -> str:
    """
  Get the text output from the OpenAI responses API.
    """
    return response.output_text

# Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            for p in m.get("content", []):
                if p['type'] == "message":
                    for content_item in p.get("content", []):
                        if content_item['type'] == "input_text":
                            st.markdown(content_item['text'])
                        elif content_item['type'] == "input_image":
                            image_url = content_item['image_url']
                            if isinstance(image_url, dict):
                                image_url = image_url['url']
                            st.image(image_url, width=100)
                        elif content_item['type'] == "input_file":
                            st.caption(f"PDF: {content_item['filename']}")
                        else:
                            st.error(f"Unknown Content type:{content_item['type']}")
        else:
            st.markdown(m["content"])
# User interface - upload images and PDFs
uploaded = st.file_uploader(
    "Upload images or PDFs",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    key=f"uploaded_files_{st.session_state.upload_widget_version}",
)
# User interface - chat input
prompt = st.chat_input("Enter your message here")

if prompt  is not None:
    # Process uploads into API-compatible data URLs
    files = [
        {
            "filename": file.name,
            "mime_type": file.type or "application/octet-stream",
            "data_url": f"data:{file.type or 'application/octet-stream'};base64,{base64.b64encode(file.read()).decode()}"
        }
        for file in uploaded or []
    ]
    
    # build the input parts of the API call
    parts = build_input_parts(prompt, files)
    st.session_state.upload_widget_version += 1

    #store the messages
    st.session_state.messages.append({"role": "user", "content": parts})

    # Display the user's message
    with st.chat_message("user"):
        for p in parts:
            if p['type'] =="message":
                for content_item in p.get("content",[]):
                    if content_item['type'] == "input_text":
                        st.markdown(content_item['text'])
                    elif content_item['type'] == "input_image":
                        image_url = content_item['image_url']
                        if isinstance(image_url, dict):
                            image_url = image_url['url']
                        st.image(image_url, width=100)
                        # display the image in a smaller size
                    elif content_item['type'] == "input_file":
                        st.caption(f"PDF: {content_item['filename']}")

                    else:
                        st.error(f"Unknown Content type:{content_item['type']}")

    # Generate the AI RESPONSE
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_responses_api(parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                # Display the AI's response
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                # Retrieve the ID if available
                if hasattr(response,"id"):
                    st.session_state.previous_response_id = response.id
            except Exception as e:
                st.error(f"Error generating response: {e}")
