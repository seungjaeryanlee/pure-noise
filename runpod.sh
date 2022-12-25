## Shell file to setup new Runpod instance

# Initialize conda to show Python
/opt/conda/bin/conda init

# Install Python packages
/opt/conda/bin/pip install -r requirements.txt

# Download tmux for JupyterLab
apt update
apt install tmux -y

# Download libGL for OpenCV
apt update
apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

# Download and prepare data
# Ignore `terminate called without an active exception` error and continue running
/opt/conda/bin/python prepare_data.py || true

# Setup git
git config --global user.name "Ryan Lee"
git config --global user.email "seungjaeryanlee@gmail.com"
