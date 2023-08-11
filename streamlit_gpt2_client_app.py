import sys
import numpy as np
import streamlit as st
import tritonclient.http
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer
from PIL import Image

st.image('https://www.activeeon.com/images/activeeon-logo.svg', width=200)

default_txt2txt = "Artificial Intelligence (AI) is a set of techniques "
if "default_txt2txt" not in st.session_state:
    st.session_state["default_txt2txt"] = default_txt2txt

default_txt2img = "A small cabin on top of a snowy mountain in the style of Disney, artstation"
if "default_txt2img" not in st.session_state:
    st.session_state["default_txt2img"] = default_txt2img

# This creates a dropdown menu where users can select one option from a list.
application = st.selectbox(
    'Choose your application',
    ('Text Generation', 'Image Generation')
)

def generate_text(prompt, model_name, triton_url, max_length=50):
    # Create the Triton Inference Server client
    try:
        client = InferenceServerClient(url=triton_url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)
    # Check if the server is live
    if client.is_server_live():
        print("Triton Server is live!")
    else:
        print("Error: Triton Server is not live.")
        sys.exit()
    # Check if the model is ready
    if client.is_model_ready(model_name):
        print(f"{model_name} is ready!")
    else:
        print(f"Error: {model_name} is not ready.")
        sys.exit()
    # Tokenize the input prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
    # Prepare an array to store the generated tokens
    output_tokens = input_ids.copy()
    for _ in range(max_length - input_ids.shape[1]):
        # Create input and output metadata for the Triton client
        infer_input = InferInput("input_ids", output_tokens.shape, "INT64")
        infer_input.set_data_from_numpy(output_tokens)  # Ensure output tokens is int64
        infer_output = InferRequestedOutput("output")
        # Get the logits from the Triton server
        response = client.infer(model_name, model_version="1", inputs=[infer_input], outputs=[infer_output])
        logits = response.as_numpy("output")
        # Sample a token from the logits
        max_logits = np.max(logits[:, -1, :], axis=-1, keepdims=True)
        probs = np.exp(logits[:, -1, :] - max_logits) / np.sum(np.exp(logits[:, -1, :] - max_logits), axis=-1, keepdims=True)
        next_token_np = np.array([np.random.choice(len(probs[0]), p=probs[0])], dtype=np.int64).reshape(1, -1)
        # Concatenate the new token to the previous tokens
        output_tokens = np.concatenate((output_tokens, next_token_np), axis=1)
    # Decode the generated tokens into text
    generated_text = tokenizer.decode(output_tokens[0])
    return generated_text

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    # grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generate_image(prompt, model_name, triton_url, model_version = "1", 
                   batch_size = 1, samples = 1, rows = 1, cols = 1, steps = 45, guidance_scale = 7.5, seed = 0):
    triton_client = tritonclient.http.InferenceServerClient(url=triton_url, verbose=False)
    assert triton_client.is_model_ready(
    model_name = model_name, model_version=model_version), f"model {model_name} not yet ready"
    # model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    # model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
    # Input placeholder
    prompt_in = tritonclient.http.InferInput(name="PROMPT", shape=(batch_size,), datatype="BYTES")
    samples_in = tritonclient.http.InferInput("SAMPLES", (batch_size, ), "INT32")
    steps_in = tritonclient.http.InferInput("STEPS", (batch_size, ), "INT32")
    guidance_scale_in = tritonclient.http.InferInput("GUIDANCE_SCALE", (batch_size, ), "FP32")
    seed_in = tritonclient.http.InferInput("SEED", (batch_size, ), "INT64")
    images = tritonclient.http.InferRequestedOutput(name="IMAGES", binary_data=False)
    # Setting inputs
    prompt_in.set_data_from_numpy(np.asarray([prompt] * batch_size, dtype=object))
    samples_in.set_data_from_numpy(np.asarray([samples], dtype=np.int32))
    steps_in.set_data_from_numpy(np.asarray([steps], dtype=np.int32))
    guidance_scale_in.set_data_from_numpy(np.asarray([guidance_scale], dtype=np.float32))
    seed_in.set_data_from_numpy(np.asarray([seed], dtype=np.int64))
    response = triton_client.infer(model_name=model_name, model_version=model_version, 
        inputs=[prompt_in,samples_in,steps_in,guidance_scale_in,seed_in], outputs=[images])
    images = response.as_numpy("IMAGES")
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    st.image(image_grid(pil_images, rows, cols), caption=prompt)

def valid_inputs(inference_endpoint, model_name, prompt):
    valid = True
    if not inference_endpoint:
        st.error("Inference endpoint is empty!")
        valid = False
    if not model_name:
        st.error("Model name is empty!")
        valid = False
    if not prompt:
        st.error("Prompt is empty!")
        valid = False
    return valid

if application == "Text Generation":
    inference_endpoint = st.text_input('Inference Endpoint')
    model_name = st.text_input('Model Name')
    prompt = st.text_area('Text Generation', st.session_state["default_txt2txt"])
    col1, col2, _, _, _, _, _ = st.columns(7)
    # Create a button for the user to generate text
    with col1:
        reset_button = st.button('Reset')
    # Create a button for the user to generate text
    with col2:
        generate_button = st.button('Generate')
    # If the button is pressed and there's a prompt, generate the text
    if generate_button and valid_inputs(inference_endpoint, model_name, prompt):
        generated_text = generate_text(prompt, model_name, inference_endpoint)
        combined_text = f'{generated_text}'
        st.session_state["default_txt2txt"] = combined_text
        st.experimental_rerun()
    if reset_button:
        st.session_state["default_txt2txt"] = default_txt2txt
        st.experimental_rerun()

if application == "Image Generation":
    inference_endpoint = st.text_input('Inference Endpoint')
    model_name = st.text_input('Model Name')
    prompt = st.text_area('Image Generation', st.session_state.get("default_txt2img", ""), height=50)
    col1, col2, _, _, _, _, _ = st.columns(7)
    # Create a button for the user to reset the prompt
    with col1:
        reset_button = st.button('Reset')
    # Create a button for the user to generate the image
    with col2:
        generate_button = st.button('Generate')
    # If the "Generate" button is pressed and there's a prompt, generate the image
    if generate_button and valid_inputs(inference_endpoint, model_name, prompt):
        generate_image(prompt, model_name, inference_endpoint)
    # If the "Reset" button is pressed, reset the prompt
    if reset_button:
        st.session_state["default_txt2img"] = default_txt2img
