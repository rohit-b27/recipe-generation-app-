import streamlit as st 
from streamlit_option_menu import option_menu
import chat, image, audio

st.set_page_config(page_title="welcome to Cooking-zone")

class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self, title, function):
        self.apps.append({
            "title":title,
            "function":function
        })
    def run():
        with st.sidebar:
            app = option_menu(
                menu_title= "Getting started",
                options= ['Chat', 'Image','Audio'],
                icons= ['chat-dots','card-image','mic'],
                menu_icon='chat-text-fill',
                default_index= 1)
            
        if app == 'Chat':
                chat.app()
        if app == 'Image':
                image.app()
        if app == 'Audio':
                audio.app()
                
    run()
            