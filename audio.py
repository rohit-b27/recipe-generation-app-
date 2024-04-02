import langchain 
import os 
import openai
from langchain_community.llms import HuggingFaceHub
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage,AIMessage
import streamlit as st 
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder 

load_dotenv()

def app():
    load_dotenv()
    
    client = OpenAI(api_key=os.getenv("openai_api_key"))
    
    
    def generate_audio_to_text(audio_file):
        with open(audio_file, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
             model = "whisper-1",
             file = audio_file
            )
            response_text = transcription.text
            return response_text

    
    chat = ChatOpenAI(model="gpt-3.5-turbo",api_key=os.getenv('openai_api_key'),temperature=0.5,verbose=True)
 

    if 'session' not in  st.session_state:
        st.session_state['session'] = [
        SystemMessage(
            content="""You are a nutritionist and multi-cusine Chef Assitant, 
                     provide the detailed recipe steps based on the human message, 
                     the human message may be dish name or might be ingredients name, depending on it
                     answer it properly. If it is dish name, mention the required ingredients and then recipe to make it.
                     If the message is ingredients, tell them a recipe using those ingredients.
                     If any additional specifications are mentioned, answer the question accordingly.
                     If the question is not related to this area, tell them this is out of your capabilites."""
                     )
        ]
    

    #function for getting the response
    def response_text_to_text(question):
        st.session_state['session'].append(HumanMessage(content=question))
        answer = chat.invoke(st.session_state['session'])
        st.session_state['session'].append(AIMessage(content=answer.content))
        return answer.content
    
    st.header('Recipe-generation through audio')
    recorded_audio = audio_recorder(text="click to record",icon_size="1x")
    if recorded_audio is not None:
        audio_file = "audio.mp3"
        with open(audio_file,"wb") as f:
            f.write(recorded_audio)
        question_text = generate_audio_to_text(audio_file)
        st.write("Transcribed text: ", question_text)
    

    


        
    
   
    
    