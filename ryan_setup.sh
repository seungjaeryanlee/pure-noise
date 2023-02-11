# Install dependencies
pip install -r requirements.txt
apt update
apt install tmux nano -y

# Setup git
git config credential.helper store
git config --global user.name "Ryan Lee"
git config --global user.email "seungjaeryanlee@gmail.com"
