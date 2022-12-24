## Shell file to setup new Runpod instance

# Initialize conda to show Python
/opt/conda/bin/conda init

# Install Python packages
/opt/conda/bin/pip install -r requirements

# Download tmux for JupyterLab
apt update
apt install tmux -y

# Download libGL for OpenCV
apt update
apt install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Download and prepare data
/opt/conda/bin/python prepare_data.py

# Setup git
git config --global user.name "Ryan Lee"
git config --global user.email "seungjaeryanlee@gmail.com"
