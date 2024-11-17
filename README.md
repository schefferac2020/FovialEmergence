# The Emergence of Fovial Image Sampling 
An implementation of [this paper](https://arxiv.org/pdf/1611.09430)


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

## TODO
- [x] Create glimpse module with gaussian kernel
- [ ] Get Simple RNN to work with just MNIST (not using our glimpse module)
- [ ] Get RNN working using glimpse module (not updating the sigmas or mus)
- [ ] Get RNN working using glimpse module while updating all parameters
- [ ] Look into if we have to do some normalization thing?


## Potential Future Steps
- [ ] Moving images?
- [ ] Use some other dataset besides MNIST
- [ ] Could extend to use Reinforcement Learning
- [ ] Use a special that can change the number of kernels over time. 
