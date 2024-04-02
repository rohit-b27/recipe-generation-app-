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
load_dotenv()


def app():
    #streamlit UI 
 st.header("Welcome to cooking-zone")

 chat = ChatOpenAI(model="gpt-3.5-turbo",api_key=os.getenv('openai_api_key'),temperature=0.6,verbose=True)
 

 if 'session' not in  st.session_state:
    st.session_state['session'] = [SystemMessage(content='You are a nutritionist and multi-cusine Chef Assitant')]

 #function for getting the response
 def response(question):
    st.session_state['session'].append(HumanMessage(content=question))
    answer = chat.invoke(st.session_state['session'])
    st.session_state['session'].append(AIMessage(content=answer.content))
    return answer.content

 input =st.text_input("Input: ", key='Input')

 submit = st.button("Submit")

 response = response(input)

 if submit:
    st.subheader("Here's the response:")
    st.write(response)