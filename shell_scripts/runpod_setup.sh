#!/bin/bash

# Step 1 We need to install tmux and vim for persistent session and editing
apt-get update
apt-get install tmux vim -y

# Step 2 Configure GIT
git config --global user.name 'your_name'
git config --global user.email 'your_email'

# Step 3 Activate virtual env
source /path_to_venv/bin

# Step 3 [Optional: required to experiment with ImageNet, especially if it is downloaded from Hugging Face]
huggging-cli login 
# This will prompt to enter your huggingface token
# Enter you hugging-face token

