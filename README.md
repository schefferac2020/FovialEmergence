# The Emergence of Fovial Image Sampling 
EECS 553 Final Project


## Setup
Create a new conda env
```bash
conda create -n fovial -y python=3.11
conda activate fovial
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Create Cluttered MNIST Dataset
```bash
cd dataset
python3 make_dataset.py
```