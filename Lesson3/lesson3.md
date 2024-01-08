# Lesson 3 : Neural Net foundations

## what is [timm](https://timm.fast.ai/) library
- you can use torch image models with timm library 

## How does a neural net really work?
- feature training from data 

### How do we fit a function to data?
- recognize patterns in the data examples we give it


### Gradient Decent 
> weight - gradient of object function 

### SGD

> In standard Gradient Descent, the entire dataset is used to compute the gradient of the loss function with respect to the model parameters. This can be computationally expensive for large datasets.

> In Stochastic Gradient Descent, instead of using the entire dataset for each update, a random subset (mini-batch) of the data is used. This introduces stochasticity into the optimization process.
The algorithm computes the gradient of the loss on the mini-batch and updates the parameters accordingly.


>Algorithm Steps:
1. Initialize the model parameters randomly.
2. Shuffle the training dataset.
3. Divide the dataset into mini-batches.
4. For each mini-batch:
    1. Compute the gradient of the loss with respect to the parameters using the current mini-batch.
    2. Update the parameters in the opposite direction of the gradient.
Repeat the process until convergence or a specified number of iterations.

### Linear Layer -> 1 Linear Regression? 
> 

### Non-Linear activation function??
> 

### Loss landscape
> can compute each weight or very high dimension landscape 
