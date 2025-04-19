import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Set up the Streamlit app
st.title("Text-to-Image Generator with Stable Diffusion")
st.write("Enter a text prompt to generate images using Stable Diffusion.")

# Initialize the model
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

pipe = load_model()

# Style presets
style_presets = {
    "Photorealistic": "highly detailed, photorealistic, 4k resolution",
    "Cartoon": "cartoon style, vibrant colors, smooth shading",
    "Abstract": "abstract art, surreal, colorful patterns"
}

# User inputs
prompt = st.text_input("Enter your prompt", value="a futuristic city at sunset")
style = st.selectbox("Select style", options=list(style_presets.keys()))
num_images = st.slider("Number of images to generate", 1, 4, 1)
guidance_scale = st.slider("Guidance scale", 5.0, 15.0, 7.5, step=0.5)
num_inference_steps = st.slider("Inference steps", 10, 100, 50, step=5)
image_size = st.selectbox("Image size", ["512x512", "768x768"])

# Parse image size
width, height = map(int, image_size.split("x"))

# Combine prompt with style
full_prompt = f"{prompt}, {style_presets[style]}"

# Generate images
if st.button("Generate Images"):
    with st.spinner("Generating images..."):
        try:
            images = pipe(
                full_prompt,
                num_images=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width
            ).images

            # Display images
            for i, img in enumerate(images):
                st.image(img, caption=f"Generated Image {i+1}")
                
                # Save image
                save_path = f"generated_image_{i+1}.png"
                img.save(save_path)
                
                # Provide download button
                with open(save_path, "rb") as file:
                    st.download_button(
                        label=f"Download Image {i+1}",
                        data=file,
                        file_name=save_path,
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Error generating images: {e}")

# Note about compute requirements
st.markdown("""
**Note**: This app requires significant computational resources. For best performance, run locally with a GPU or use a paid hosting service for deployment.
""")