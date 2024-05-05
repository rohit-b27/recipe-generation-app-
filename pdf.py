# chatbot using streamlit 
import langchain 
import os 
import openai
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage,AIMessage
import streamlit as st 
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader

def app():
    # Load environment variables
    load_dotenv()

    st.header("Welcome to cooking-zone")

    # Function to extract text from PDFs
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    # Function to split text into chunks
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    # Function to embed text chunks
    def get_vector_store(text_chunks):
        embeddings = OpenAIEmbeddings(api_key=os.getenv('openai_api_key'))
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store

    # Function to build a retrieval chain
    def build_chain(vector_store):
        chat = ChatOpenAI(model="gpt-4-turbo", api_key=os.getenv('openai_api_key'), temperature=0.6, verbose=True)
        prompt_template = """
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        I will tip you $1000 if the user finds the answer helpful.
        <context>{context}</context>
        Question: {input}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        document_chain = create_stuff_documents_chain(chat, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    # Function to process user input and generate a response
    def user_input(user_question, text_chunks):
        vector_store = get_vector_store(text_chunks)
        docs = vector_store.similarity_search(user_question)
        retrieval_chain = build_chain(vector_store)
        response = retrieval_chain.invoke({"input_documents": docs, "input": user_question})
        st.write("Reply: ", response['answer'])

    # File uploader and processing button
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            st.session_state['chunks'] = chunks  # Store chunks in session state
            get_vector_store(chunks)
            st.success("Done")

    # Input for user question and processing
    user_question = st.text_input("Ask a Question from the PDF Files",key="input")
    submit = st.button("Submit")

    if submit and st.session_state['chunks']:
        user_input(user_question,st.session_state['chunks'])
    elif submit and 'chunks' not in st.session_state:
        st.error("Please upload and process PDF files before submitting a question.")
