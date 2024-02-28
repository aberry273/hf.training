# Create a virtual environment
python -m venv .env

# Activate the virtual environment
source .env/bin/activate
.env/scripts/activate.bat

# Deactivate the virtual environment
source .env/bin/deactivate
.env/scripts/deactivate.bat

# Install packages
pip install -r requirements.txt 

# Login to HF to save mdoel
huggingface-cli login
pip3 install --upgrade huggingface_hub