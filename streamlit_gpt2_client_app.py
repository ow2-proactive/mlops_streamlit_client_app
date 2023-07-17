import sys
import numpy as np
import streamlit as st

from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer

if "default" not in st.session_state:
    st.session_state["default"] = "Artificial Intelligence (AI) is a set of techniques "

st.image('https://www.activeeon.com/images/activeeon-logo.svg', width=200)

inference_endpoint = st.text_input('Inference Endpoint', '')
model_name = st.text_input('Model Name', 'gpt2')

def generate_text(prompt, model_name, triton_url=inference_endpoint, max_length=50):
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

prompt = st.text_area('Text Generation', st.session_state["default"])

col1, col2, _,_,_,_,_ = st.columns(7)

# Create a button for the user to generate text
with col1:
    clear_button = st.button('Clear')

# Create a button for the user to generate text
with col2:
    generate_button = st.button('Generate')

# If the button is pressed and there's a prompt, generate the text
if generate_button and prompt.strip() != "":
    generated_text = generate_text(prompt, model_name)
    combined_text = f'{generated_text}'
    st.session_state["default"] = combined_text
    st.experimental_rerun()

if clear_button:
    st.session_state["default"] = ""
    st.experimental_rerun()
