# Streamlit GPT2 and Stable Diffusion client app
Streamlit GPT2 and Stable Diffusion client app for the MLOps Dashboard

This is the client app used to perform inferences on the GPT2 and Stable Diffusion model deployed with the MLOps Dashboard from Activeeon.

For more info:
- (English) https://www.activeeon.com/blog/deploy-and-monitor-llms-with-proactive/
- (French) https://www.activeeon.com/blog/mlops-dashboard/

### Requirements
- Activeeon's MLOps Dashboard
- [GPT2 model from Hugging Face](https://huggingface.co/gpt2)
- [Stable Diffusion Triton Model](https://github.com/kamalkraj/stable-diffusion-tritonserver)

### Setup
Create a Python virtual env and install the project dependencies
```bash
rm -rf streamlit_app_env
python3 -m venv streamlit_app_env
source streamlit_app_env/bin/activate

python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt
```

### Run
```bash
streamlit run streamlit_gpt2_client_app.py
```

![Animated GIF](https://github.com/ow2-proactive/mlops_streamlit_gpt2_app/blob/34afcdb1d7c0647e7da29c104d8bb7d53d813e86/images/streamlit_client.gif)
