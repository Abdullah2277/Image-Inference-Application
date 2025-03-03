import streamlit as st
import pyttsx3
import google.generativeai as genai
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()  # To Load environment variables from .env

class ImageInferenceApp:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Configuring Gemini API key

    def speak_result(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except RuntimeError:
            pass

    def process_image(self, model_name, image_bytes, prompt):
        try:
            if model_name == "maxiw/Phi-3.5-vision":
                from gradio_client import Client, handle_file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_file_path = tmp_file.name

                client = Client("maxiw/Phi-3.5-vision")
                result = client.predict(
                    image=handle_file(tmp_file_path),
                    text_input=prompt,
                    model_id="microsoft/Phi-3.5-vision-instruct",
                    api_name="/run_example"
                )
                return result

            elif model_name == "gemini-2.0-flash":
                model = genai.GenerativeModel('gemini-2.0-flash')
                img = Image.open(io.BytesIO(image_bytes))
                response = model.generate_content([prompt, img])
                return response.text
            else:
                return "Invalid model selected."

        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.title('Image Inference App')

    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False

    app = ImageInferenceApp()

    model_name = st.selectbox('Select Model:', [
        "gemini-2.0-flash",
        "maxiw/Phi-3.5-vision"
    ])

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if not st.session_state.image_uploaded:
            st.success("Image uploaded successfully!")
            app.speak_result("Image uploaded successfully!")
            st.session_state.image_uploaded = True

        image_bytes = image_file.getvalue()

    default_prompt = "Describe the image for a blind person."
    prompt = st.text_area('Enter your prompt here...', value=default_prompt)

    if st.button('Submit'):
        if model_name and image_file and prompt:
            st.info("Image submitted for interpretation...")
            app.speak_result("Image submitted for interpretation...")
            app.speak_result("Please wait...")

            with st.spinner("Please wait..."):
                result = app.process_image(model_name, image_bytes, prompt)

            st.text_area('Output:', value=result, height=200)

            if result:
                app.speak_result(result)
            else:
                st.warning("No result to speak.")
        else:
            st.error("Please fill in all fields.")

if __name__ == '__main__':
    main()