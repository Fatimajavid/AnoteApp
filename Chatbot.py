import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

import json
import datetime
import shutil
import tempfile
import openai
from openai import OpenAI
from tempfile import NamedTemporaryFile


# Set up OpenAI API
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key= API_KEY)

def process_file(uploaded_file):
    # Create a temporary file to save the uploaded file
    temp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file_path = temp_file.name

    # Save the uploaded file to the temporary file
    temp_file.write(uploaded_file.read())

    st.success(f"File processed successfully: {uploaded_file.name} in {temp_file_path}")
    return temp_file_path


def ask_gpt_finetuned_model(path_to_10k, question):
    db, db_dir = create_knowledge_hub(path_to_10k)

    source1 = db.similarity_search(question, k=2)[0].page_content
    source2 = db.similarity_search(question, k=2)[1].page_content

    client = OpenAI()

     # Load existing conversation from the JSON file
    try:
        with open("conversation_data.jsonl", "r") as file:
            conversation_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        conversation_data = {"messages": []}

    messages = conversation_data["messages"]

    # Add the latest user message
    messages.append({"role": "user", "content": f"{source1}{source2} Now, this is our question: {question}"})

    # Include the conversation history in the API call
    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0613:personal:anote:8DO8V2LB",
        messages=[
            {"role": "system", "content": "You are a factual chatbot that answers questions about 10-K documents. You only answer with answers you find in the text, no outside information."},
            *messages  # Unpack the list of messages
        ]
    )

    # messages.append({"role": "user", "content": f"{source1}{source2} Now, this is our question: {question}"})


    # completion = client.chat.completions.create(
    #     model="ft:gpt-3.5-turbo-0613:personal:anote:8DO8V2LB",
    #     messages=[
    #         {"role": "system", "content": "You are a factual chatbot that answers questions about 10-K documents. You only answer with answers you find in the text, no outside information."},
    #         {"role": "user", "content": f"{source1}{source2} Now, this is our question: {question}"}
    #     ]
    # )

    delete_chroma_db(db_dir)

    # Append the new assistant message to the conversation data
    messages.append({"role": "assistant", "content": completion.choices[0].message.content})

    # Update the conversation history in the JSON file
    with open("conversation_data.jsonl", "w") as file:
        json.dump({"messages": messages}, file)
    
    return completion.choices[0].message.content

def create_knowledge_hub(path_to_10k):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    db_directory = "db_" + timestamp

    loader = PyPDFLoader(path_to_10k)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=5,
        separators=["\n\n", "\n", " ", ""],
        length_function=len)
    texts = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=db_directory
    )
    vectordb.persist()

    return vectordb, db_directory

def delete_chroma_db(db_directory):
    try:
        shutil.rmtree(db_directory)
        #print(f"Chroma database '{db_directory}' deleted successfully.")
    except FileNotFoundError:
        print(f"Chroma database '{db_directory}' not found.")
    except Exception as e:
        print(f"Error deleting Chroma database: {str(e)}")

def fill_json(path_to_json, path_to_10k, question, answer):
    db, db_dir = create_knowledge_hub(path_to_10k)

    source1 = db.similarity_search(question, k = 2)[0].page_content
    source2 = db.similarity_search(question, k = 2)[1].page_content

    source1 = source1.replace(r'\x', '')
    source2 = source2.replace(r'\x', '')

    source1 = source1.replace('\n', ' ')
    source2 = source2.replace('\n', ' ')

    source1 = source1.replace('\"', ' ')
    source2 = source2.replace('\"', ' ')

    source1 = source1.replace('\'', ' ')
    source2 = source2.replace('\'', ' ')

    ROLE_SYSTEM = "You are a factual chatbot that answers questions about 10-K documents. You only answer with answers you find in the text, no outside information."
    
    my_data = (
        f'{{"messages": [{{"role": "system", "content": "{ROLE_SYSTEM}"}},'
        f'{{"role": "user", "content": "This is our information from the 10-K: {source1} {source2}. Now, this is our question: {question}"}},'
        f'{{"role": "assistant", "content": "{answer}"}}]}}'
        '\n'
    )

    delete_chroma_db(db_dir)

    try:
        with open(path_to_json, "a") as file:
            file.write(my_data)
    except (FileNotFoundError, json.JSONDecodeError):
        return

def main():
    st.set_page_config(page_title="ANOTE")

    st.header( "ANOTE Financial Chatbot :speech_balloon:")
    st.subheader("Hello! Please upload a pdf of your 10k document(s) so that I can assist you!")

    # File Uploader for PDFs
    uploaded_files = st.file_uploader("Upload 10-K PDFs", type="pdf", accept_multiple_files=True)

    file_paths = []  # List to store file paths

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = process_file(uploaded_file)
            file_paths.append(file_path)

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hello! How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Button to ask the question
    if prompt:
        with st.spinner("Loading response"):
            for file_path in file_paths:
                answer = ask_gpt_finetuned_model(file_path, prompt)
                st.markdown(f"{answer}")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

if __name__ == '__main__':
    main()