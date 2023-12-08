# # Importing pysqlite3 and replacing the sqlite3 module
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import sqlite
# # Import other necessary modules
import streamlit as st

st.set_page_config(
    page_title="ANOTE Financial Chatbot",
    page_icon="images/anote_ai_logo.png",
)
st.header("ANOTE Financial Chatbot :speech_balloon:")
st.subheader("Welcome to ANOTE!")
st.success("This is our Private GPT. We aim to help data teams within the finance sector to answer questions on documents such as 10-Ks.\n\n Come chat about your documents while keeping your data private and secure. ")
st.info("You can select between either of the available options: ")

# Create three columns 
col1, col2 = st.columns([1,1])

with col1:
    img = st.image('images/apichat.png')
    st.markdown('<a href="/EdgarAPIChatbot"></a>', unsafe_allow_html=True)
    st.warning("Specify the ticker of a company to ask questions based on the 10K of the company.")



with col2:
    st.image('images/pdfchat.png')
    st.markdown('<a href="/PDFChatbot"></a>', unsafe_allow_html=True)
    st.warning("Specify the ticker of a company to ask questions based on the 10K of the company.")


st.info("You may navigate to either dashboard from the left sidebar.")
