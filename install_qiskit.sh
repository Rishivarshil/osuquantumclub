#!/bin/bash


if ! command -v conda &> /dev/null; then
    echo "Miniconda not found. Installing..."
    curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    source $HOME/miniconda/etc/profile.d/conda.sh
else
    echo "Miniconda already installed."
    source $(conda info --base)/etc/profile.d/conda.sh
fi

if ! conda env list | grep -q "qiskit"; then
    echo "Creating environment 'qiskit'..."
    conda create -n qiskit python=3.11 -y
else
    echo "Environment 'qiskit' already exists."
fi

echo "Installing Qiskit packages into 'qiskit'..."
conda activate qiskit
pip install --upgrade pip
pip install qiskit-aer qiskit-ibm-runtime "qiskit[visualization]" jupyter matplotlib numpy

if ! command -v code &> /dev/null; then
    echo "VS Code not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install --cask visual-studio-code
    else
        wget -qO vscode.deb https://update.code.visualstudio.com/latest/linux-deb-x64/stable
        sudo apt install ./vscode.deb -y
        rm vscode.deb
    fi
else
    echo "VS Code is already installed."
fi

# --- Step 5: Create a starter notebook ---
NOTEBOOK_DIR="$HOME/QiskitNotebooks"
mkdir -p "$NOTEBOOK_DIR"
cat > "$NOTEBOOK_DIR/hello_quantum.ipynb" <<EOF
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from qiskit import QuantumCircuit\\n",
    "qc = QuantumCircuit(2)\\n",
    "qc.h(0)\\n",
    "qc.cx(0,1)\\n",
    "qc.measure_all()\\n",
    "qc.draw('mpl')"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF

