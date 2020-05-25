# Learning Neural Network Weights with the Ensemble Kalman Filter

This repository contains code for learning the weights of a neural network using the Ensemble Kalman Filter. There are three main experiments:

1. Comparison to backpropagation: `generate_comparison.py`
2. Impact of precision threshold: `varying_r.py`
2. Transition from ENKF to backpropagation: `varying_pretrain.py`

Each file accepts a set of command line arguments which determine the dataset, model architecture, and ENKF hyperparameters. For example:
```
python generate_comparison.py --dataset=boston_housing --model=fcn --r=0.01 --initial_noise=0.03 --batch_size=16  --timesteps=25 --num_epochs=5 --num_particles=50
```
Running this from the command line will compare backprop with ENKF on the boston housing dataset with a fully connected network architecture using the specified learning hyperparameters.
