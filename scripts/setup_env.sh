#!/bin/bash
ml load Python/3.11.5-GCCcore-13.2.0
. ./prompt_venv/bin/activate

export HF_HOME="/mnt/proj2/open-29-45/poludmik/textvision/cache"
export HF_DATASETS_CACHE="/mnt/proj2/open-29-45/poludmik/textvision/cache"

# Dynamically extract the virtual environment name from $VIRTUAL_ENV
VENV_NAME=$(basename "$VIRTUAL_ENV")

# Set the PS1 prompt to use the dynamic virtual environment name
PS1="($VENV_NAME) \\[\e[1;32m\\][\\u@\\h.karolina \\[\e[1;33m\\]\\W\\[\e[0m\\]]\\\$ "

# squeue -u poludmik -l