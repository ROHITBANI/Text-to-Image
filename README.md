# Text-to-Image Generator with Stable Diffusion
This is a Streamlit app that generates images from text prompts using the Stable Diffusion model from Hugging Face. Users can customize the output with style presets, guidance scale, inference steps, image size, and the number of images to generate.
## Features
### Text Prompt Input: Enter any text prompt (e.g., "a futuristic city at sunset").
### Style Presets: Choose from Photorealistic, Cartoon, or Abstract styles.
### Customization:
### Guidance scale (5.0 to 15.0) to control adherence to the prompt.
### Number of inference steps (10 to 100) for image quality.
### Image size (512x512 or 768x768).
Generate 1 to 4 images per prompt.
Save Images: Download generated images as PNG files.
### Streamlit Interface: User-friendly web app for easy interaction.
# Model Explanation
The app uses the runwayml/stable-diffusion-v1-5 model from Hugging Face's Diffusers library, a pre-trained Stable Diffusion model. Stable Diffusion is a latent diffusion model that generates high-quality images from text prompts by iteratively denoising a random noise vector guided by the text embedding. The model runs with torch.float16 precision to optimize performance on GPUs.
# Prerequisites
Python 3.8+
A GPU with at least 8GB VRAM (recommended for faster generation)
CUDA (if running on GPU)
