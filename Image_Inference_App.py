import streamlit as st
import pyttsx3
from gradio_client import Client, handle_file
from PIL import Image
import tempfile

class ImageInferenceApp:
    def __init__(self):
        self.client = None
        self.tts_engine = pyttsx3.init()  # Initialize the TTS engine

    def speak_result(self, text):
        """Convert the given text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()  # Wait until the speech is finished

    def process_image(self, model_name, image_path, prompt):
        try:
            if model_name == "maxiw/Phi-3.5-vision":
                result = self.client.predict(
                    image=handle_file(image_path),
                    text_input=prompt,
                    model_id="microsoft/Phi-3.5-vision-instruct",
                    api_name="/run_example"
                )
            elif model_name == "HuggingFaceM4/Docmatix-Florence-2":
                result = self.client.predict(
                    image=handle_file(image_path),
                    text_input=prompt,
                    api_name="/process_image"
                )
            return result
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.title('Image Inference App')
    
    if "image_uploaded" not in st.session_state:
        st.session_state.image_uploaded = False

    app = ImageInferenceApp()

    model_name = st.selectbox('Select Model:', [
        "HuggingFaceM4/Docmatix-Florence-2",
        "maxiw/Phi-3.5-vision"
    ])

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if not st.session_state.image_uploaded:
            st.success("Image uploaded successfully!")  # Display the success message
            app.speak_result("Image uploaded successfully!")  # Speak the success message
            st.session_state.image_uploaded = True

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(image_file.getbuffer())
            tmp_file_path = tmp_file.name

    default_prompt = "Describe the image for a blind person."
    prompt = st.text_area('Enter your prompt here...', value=default_prompt)

    if st.button('Submit'):
        if model_name and image_file and prompt:
            app.client = Client(model_name)
            st.info("Image submitted for interpretation...")  # Display the info message
            app.speak_result("Image submitted for interpretation...")
            
            # Speak "Please wait..." only once before starting the spinner
            app.speak_result("Please wait...")  # Speak the waiting message
            
            with st.spinner("Please wait..."):  # Spinner while processing
                result = app.process_image(model_name, tmp_file_path, prompt)
            
            st.text_area('Output:', value=result, height=200)
            
            if result:
                app.speak_result(result)  # Speak the result
            else:
                st.warning("No result to speak.")
        else:
            st.error("Please fill in all fields.")

if __name__ == '__main__':
    main()

# To run the app, use the command:
# streamlit run c:/Users/PMLS/Downloads/AIES_project_streamlit_app.py
