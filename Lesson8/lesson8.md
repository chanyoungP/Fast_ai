# Lesson 8 Convolutional Neural Network

## 0. Convolution

### 1. how it works?
> sliding window(filter) to extract feature <br>
then what feature?? -> we **_train_** what feature gonna help to understanding images  (this is the differnce between ML and DL)

### 2. how to compute ?? 
> dot product and sum <br>
it is actually Correlational operation in CNN  <br>
but **we don't care cause we train the weight**

## 1. Max Pooling

### 1. how it works? 
> mainly doing Down sampling of the features 

- Convolutional also do downsampling then why we need maxpooling layer? 

    - Spatial Down-sampling
        > only remain maximum value -> downsampling

    - Reduction of Computational Complexity
        > By retaining only the maximum value within each region, it captures the most essential information while discarding less relevant details.

    - Translation Invariance:
        > Even if an object in the input **feature map shifts slightly**, **the maximum value within the pooling region will remain the same**, making the network more robust to small variations in position.


    - Localization vs. Summarization:
        > Convolutional operations primarily focus on feature extraction and can capture local patterns. On the other hand, **MaxPooling** is more concerned with summarizing the relevant information by keeping the most important activations. **It acts as a form of feature selection.**

    - Parameter Reduction
        > MaxPooling helps in controlling the number of parameters and computations in the subsequent layers

    - Hierarchical Feature Representation
        > The combination of convolutional layers and pooling layers helps in building hierarchical feature representations. Convolutional layers capture local patterns, and pooling layers progressively summarize and abstract these patterns at different levels.

## 2. DropOut in CNN

### 1. how it works?
- Randomly drop unit of output features not a filter weights 
(bits of the activations)

### 2. why? 
- if we drop unit of the activation then the computer force to learn the underlying real representation rather than overfitting (like data augmentation)

- we can think drop out like this way 
    > drop out is data augmentation for the activations
    > (data augmentation is for the input)












