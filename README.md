# The Emergence of Fovial Image Sampling 
An implementation of [this paper](https://arxiv.org/pdf/1611.09430). An agent with limited photoreceptors 
is tasked with recognizing digits in cluttered environments in as few "glimpses" as possible.
Trained with backpropagation, the agent learns an emergent sampling lattice with higher precision at the center 
of the gaze. This learned lattice closely resembles the density of ganglion cells
in the primate retina.


![](./readme/time_example.png)

## Emergent Fovial Lattice 
The following shows the sampling lattice (learned through backpropagation) evolving as the algorithm is trained on a [*Cluttered MNIST dataset*](./dataset/). The circles represent 144 unique "receptive fields" where the radius of the circles represent the standard deviation of a gaussian kernel used to sample the image. The larger the circle, the larger the receptive field.  

![](./readme/output_video.gif)


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
python3 make_dataset.py -h
```