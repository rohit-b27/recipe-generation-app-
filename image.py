import streamlit as st 
import google.generativeai as genai 
from PIL import Image
import os 
from dotenv import load_dotenv
load_dotenv()





def app():
    genai.configure(api_key = os.getenv('google_api_key'))
    
    ## loading gemini pro vision 
    model = genai.GenerativeModel('gemini-pro-vision')
    
    def get_gemini_response(input,image,prompt):
        response = model.generate_content([input, image[0], prompt])
        return response.text
    
    
    st.header('Decode the taste of an image')
    input = st.text_input("Input: ", key= "input", placeholder= "prompt: ")
    file_uploaded = st.file_uploader("upload the image: ", type=['jpg','jpeg','png'])
    
    
    
    def input_image_details(file_image):
        if file_image is not None:
            #read the image:
            bytes_data = file_image.getvalue()
            image_parts = [
                {
                    "mime_type": file_image.type, 
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
        
           
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption="uploaded Image.", use_column_width=True)
    submit = st.button("Generate the Recipe")
    submit1 = st.button("Nutritional-Value")
    
    input_prompt = """ You are a professional multi-cusine international Chef. 
    We will upload an image of a dish, and you have to generate the recipe of the dish,
    based on the input prompt. If there is no prompt given generate the dish name, ingredients required 
    and recipe steps as a default procedure."""
    
    nutritional_prompt = """ You are an expert nutritionist, where you need to extract the food items from the image and
    give the calorie count of each ingredient and mention nutritional composition, i.e percentage of protients, carbohydrates, fats, fibers,
    vitamins, minerals, etc, in the dish. If any prompt is given by user tailor the response accordingly."""
    
    if submit:
        image_data = input_image_details(file_uploaded)
        response = get_gemini_response(input_prompt, image_data, input)
        st.subheader("Response: ")
        st.write(response)
        
    if submit1:
        image_data = input_image_details(file_uploaded)
        response = get_gemini_response(nutritional_prompt, image_data, input)
        st.subheader("Response: ")
        st.write(response)
        
    
        
        

    