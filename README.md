# Perceptron & Neural Network project
Python university project for implementing a basic perceptron and a neural network for solving several problems.


## Requirements
Several libraries are used. Mainly `numpy`, `pandas` and `sklearn(shuffle util)`.
If when running any dependency issues arise, `PIP` tool should be used for solving such issues.  
Implemented and tested on `Python 3.7`

## Running
Navigate to bundles root dir with `cd src/ar/edu/itba/sia/group3`.
Next run each of the requested exercises. Each file name in this directory indicates the exercise.  
All files can be configured inside. And run with `python file`

### Configurations
- learning_rate => % of delta W to add to weighs on neural network
- momentum => momentum variable. Ideally in [0,1]
- iteration_limit => number of epochs
- features => number of input arguments
- activation_function => the function to be used from `src/ar/edu/itba/sia/group3/Functions/Activations_Functions.py`.
- restart_condition => used for restarting perceptron on error not changing
- layer_info_list => defines the neural network structure.

### More on layer_info_list
 Firtst element is first hidden layer, the following are the remaining hidden layers in order. Final element is the output. All elements have 2 int parameters. First int is number of neurons in layer, second int is number of neurons or inputs in previous layer.
 
## Example runs
