rm -rf streamlit_app_env
python3 -m venv streamlit_app_env
source streamlit_app_env/bin/activate

python3 -m pip install --upgrade pip

# Install project dependencies
python3 -m pip install -r requirements.txt
