import streamlit as st 
from streamlit_option_menu import option_menu
import chat, image, audio, pdf

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
                menu_title= "Input",
                options= ['Chat', 'Image','Audio','pdf'],
                icons= ['chat-dots','card-image','mic','file-earmark-pdf-fill'],
                menu_icon='chat-text-fill',
                default_index= 1)
            
        if app == 'Chat':
                chat.app()
        if app == 'Image':
                image.app()
        if app == 'Audio':
                audio.app()
        if app =='pdf':
                pdf.app()        
    run()
            