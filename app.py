import streamlit as st
from ImageModel import promptgen,text2image  # your voice to text function
from io import BytesIO  # your text to image function

def app():
    st.title("Audio2Art: Transforming Audio Prompts into Visual Creations")

    # File uploader for .wav files
    upload_file = st.file_uploader("Choose your .wav audio file", type=["wav"])

    # Model selection
    option = st.selectbox(
        "Select model for Image Generation:",
        [
            "nota-ai/bk-sdm-small",
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
            "prompthero/openjourney",
            "hakurei/waifu-diffusion",
            "stabilityai/stable-diffusion-2-1",
            "dreamlike-art/dreamlike-photoreal-2.0"
        ]
    )

    # Form submission
    with st.form("my_form"):
        submit = st.form_submit_button(label="Submit Audio File!")

    # Processing after submit
    if submit:
        with st.spinner(text="Generating Image ... It may take some time."):
            prompt = promptgen(upload_file)
            im, start, end = text2image(prompt, option)
            buf = BytesIO()
            im.save(buf, format="PNG")
            byte_im = buf.getvalue()

            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)

            st.success("Processing time: {:02}:{:02}:{:05.2f}".format(int(hours), int(minutes), seconds))
            st.image(im)

            st.download_button(
                label="Click here to download",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png",
            )

    # Sidebar Info
    st.sidebar.markdown("## Guide")
    st.sidebar.info("This UI uses Wav2Vec and Text2Image Transformers to convert your audio prompt into an image!")
    st.sidebar.markdown("1. Upload a .wav file\n2. Choose your model\n3. Click submit\n4. See how the AI generates an image from your voice!")
    st.sidebar.write("1. Give an audio file to see how the AI generates an image from your voice!")
    st.sidebar.write("2. try different prompts!")

if __name__ == "__main__":
    app()
