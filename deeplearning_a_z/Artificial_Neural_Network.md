[TOC]



# Artificial Neural Network

In this part we will learn: 

1.  [The intuition of ANNs](## ANN Intuition)
2.  How to Build an ANN
3.  How to predict an outcome of a single observation

For more in-depth material, check out [The Ultimate Guide to Artificial Neural Netowrk (ANN)](https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann)

## ANN Intuition

In this section, we will begin into the depth of ANN. Here are few things we will cover in this section: 

*   [The Neuron](# The-Neuron)
*   [The Activation Function](#The-Activation-Function)
*   How do Neural Networks Work?
*   How do Neural Networks Learn?
*   Gradient Descent
*   Stochastic Gradient Descent
*   Backpropogation

### The Neuron

In the previous chapter we covered what a neuron. In this section we will see how we can recreate this in a computer. A neuron consists of a nucleus and have multiple branches coming out of it. They look something like this: 

<img src="Artificial_Neural_Network.assets/image-20210303093705824.png" alt="image-20210303093705824" style="zoom:80%;" />

Neurons by themselves are quite useless. They cannot do anything by themselves. However, when they are connected with other neurons, they can do a lot. The **dendrites** are the receivers of the neurons while the **axon** are the transmitters. An electric signal passes between neurons and that is how they communicates with each other. 

### Neurons in Machines

We create the neuron conceptually as follows: 

<img src="Artificial_Neural_Network.assets/image-20210303094254301.png" alt="image-20210303094254301" style="zoom:80%;" />

Here we have the input layer, which is indicated by the yellow neurons. This is not the case in a neuron in the brain but we make use of **input neurons** that take information from input values and passing to the neuron. Such a neuron is presented by the color green. The neuron then processes these signals and passes to the output layer. The output layer is presented by a red. 

Let's look at in more detail: 

*   **Input layer** - Are independent variables. Each neuron in yellow is a single independent variable. If there are $m$ features, we will have $m$ neurons in the input layer. **The independent variables need to be standardized and scaled**. To know more about why need to standardized and scale, read this [article](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf).
*   **Output Layer** - The output layer can be numerical, binary, or categorical variable with multiple categories. 

It is important to remember that at any given point, the ANN will take a single observation, process it and output the prediction of that single observation. 

<img src="Artificial_Neural_Network.assets/image-20210303094854383.png" alt="image-20210303094854383" style="zoom:80%;" />

Each of the connection between neurons have associated **weights**. The learning of Neural Network is learning what weights each synapse should be in order for the prediction to be as close to actual observation. 

The **neuron** takes the weighted sum of all the input values from the input feature. We then assign an **activation function** to the weighted sum of the observations. The result of the activation is then passed to the output. 

So, we can think of the whole process in four steps: 

1.  The input layer passes the features to the neurons. Each input has an associated weight. The feature and the weights are passed to the neuron. 
2.  The neuron takes a weighted sum of the input layer.
3.  The neuron passes this weighted sum through an activation function
4.  The output of the activation function is passed to the output layer

<img src="Artificial_Neural_Network.assets/image-20210303095651391.png" alt="image-20210303095651391" style="zoom:80%;" />

### The Activation Function

There are four differnet types of activation functions that exists. 

*   **Threshold Function**. It is a very simple function which is zero for some values and then is equal to 1 for the rest of the values. This is sort of a binary function

    <img src="Artificial_Neural_Network.assets/image-20210303095806883.png" alt="image-20210303095806883" style="zoom:80%;" />

    

*   **Sigmoid Function** The sigmoid function is used in logisitic regression. It is a smooth function that gradually increases from 0 to 1, unlike the threshold function. This is often used to prediction 

    <img src="Artificial_Neural_Network.assets/image-20210303095916108.png" alt="image-20210303095916108" style="zoom:80%;" />

    

*   **Rectifier Function**: This is one of the most used in the ANN.

    <img src="Artificial_Neural_Network.assets/image-20210303095955528.png" alt="image-20210303095955528" style="zoom:80%;" />

    

*   **Hyperbolic Tangent**: This is another smooth function that looks like the sigmoid function. 

    <img src="Artificial_Neural_Network.assets/image-20210303100032082.png" alt="image-20210303100032082" style="zoom:80%;" />

The activation function is applied in the hidden layer but depending on the type of an output, we will also apply another activation at the output layer such that the output is as expected. The common practice is to apply the rectifier in the hidden layers and the sigmoid for the output layer if the expected output is binary or categorical. 

<img src="Artificial_Neural_Network.assets/image-20210303100628821.png" alt="image-20210303100628821" style="zoom:80%;" />

### How do Neural Networks Work?

To explain how a neural network works we will use a tutorial to predict property prices. One caveat is that we assume that the neural network is already trained. Here we want to see what a trained neural network does. 

In terms of property we have the following features: 

*   Area
*   Bedrooms
*   Distance to city
*   Age

These features will be our input neurons. This would look something like this: 

<img src="Artificial_Neural_Network.assets/image-20210303102823989.png" alt="image-20210303102823989" style="zoom:80%;" />

We can get the price based on a linear combination of features and weights. This is similar to what we have in linear regression. So, how is neural network different from standard multiple linear regression? 

The power of neural network comes from the use of hidden layer. The hidden layer or layers gives power to the neural networks. So, in our example, we add a hidden layer. Now let's consider the first neuron in the hidden layer: 

<img src="Artificial_Neural_Network.assets/image-20210303103137072.png" alt="image-20210303103137072" style="zoom:80%;" />

We pass all the features to this neuron and associated non-zero weights. Let's say this this neuron gives more weightage to Area and Distance to city and less to the other. This could be because this neuron thinks that going away from the city center will result in decrease in the price for a given area. So, this neuron may pick such a condition. We don't know but it is something the neuron has picked through learning. Remember, we have already taught this neural network with training data. 

We pass all the features to each of the neurons in the hidden layer. Just as the first neuron, we find that each neurons give different weights to each of the input features. This is the power that the neural network has, the combination of the features. The neural network will pick up combinations of features that we would not have thought of. The final neural network would look something like this: 

<img src="Artificial_Neural_Network.assets/image-20210303104030919.png" alt="image-20210303104030919" style="zoom:80%;" /> 

These synapses shown here are the synapses that have non-zero weights. Finally, all the neurons in the hidden layer send their output to the output layer, where the final predictionis made. 

This is in short how a neural network works. 

### How do Neural Networks Learn?

The neural network is given a large amount of data. For example, in order to train a neural network to distinguish between a cat and a dog, it is given thousands of images of cat and dog. We then ask it to go and learn from these images. The neural network finds features that allows it to distinguish and be trained. 

A single-layered neural network that we have seen so far is called a **perceptron**. Such a perceptron looks like this: 

<img src="Artificial_Neural_Network.assets/image-20210303104826658.png" alt="image-20210303104826658" style="zoom:80%;" />

The perceptron learns in the following way: 

1.  The perceptron takes features from a single instance. It goes, along with associated weights, to the neuron.
2.  The activation function is applied to the weighted sum of the features
3.  The resultant information is then sent to the output layer. 
4.  The output layer compares the prediction $\hat{y}$ with the actual value $y$. 
5.  How close the prediction is to the actual value is determined by the loss function. The typical loss function is $L = 1/2(\hat{y} - y)^2$
6.  We repeat steps 1 - 5 for each of the observations in the dataset while keeping track of each prediction. 
7.  We compute the cost function, which is the sum of loss functions.
8.  We adjust the weights of the perceptron
9.  We repeat steps 1-8 until the cost function has been minimized. 

We can see these steps visually as follows. Suppose we have a dataset that has 8 rows. It would look something like this: 

<img src="Artificial_Neural_Network.assets/image-20210303105828717.png" alt="image-20210303105828717" style="zoom:80%;" />

We have the neural network on the left. It is the same neural network but shown 8 times corresponding to the 8 observations in the dataset. Now as we go through each observation, we compute the predictions for each observation. We now have 8 predictions corresponding to the 8 observations: 

<img src="Artificial_Neural_Network.assets/image-20210303110036459.png" alt="image-20210303110036459" style="zoom:80%;" />

We can now compute the **cost function**, which is a sum of **loss function**, for the dataset. 

Once we have the cost function, we adjust the weights of the percepton. And with these new weights, we again go through all the observations. We compute the cost function again, change the weights and repeat. We stop when we have have minimized the cost function. The method used to update the weights is called **backpropogation**. 

>   Going through all of the observations in the dataset is called an epoch. To train a neural network, we run through multiple epochs. 

You can learn from about the cost functions [here](https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)

### Gradient Descent

As we saw in the previous section, the backpropogation along with the cost function updates the weights. However, the right weights that reduce the cost function to the minimum is often one in the total number of combinations. Finding this value is often not possible due to the curse of dimensionality. That is where the gradient descent comes in. 

The gradient descent is an iterative algorithm that finds the minimum of the cost function by computing the gradient of the cost function at a given point. Gradient descent works well when the function is smooth and really well when the cost function has just one minima. 

So rather than looking at each combination of weights, we simply look at weights that decrease the cost function. 

### Stochastic Gradient Descent

There are two reasons why the gradient descent by itself is not good: 

1.  The gradient descent uses all of the observations to find the gradient of the cost function
2.  It is sensitive to a local minima and therefore it requires the cost function to be convex. 

Stochastic gradient descent addresses both of these issues by computing the gradient of the cost function by using just a single observation. The weights are adjusted by using a single observations rather than adjusting them after going through all the observations. The gradient descent is also known as **batch gradient descent**. 

Here's how the weights are updated in either of the methods: 

<img src="Artificial_Neural_Network.assets/image-20210303113504773.png" alt="image-20210303113504773" style="zoom:80%;" />

Stochastic gradient descent is also faster as it uses less data. 

There is a nice article on [Gradient Descent](https://iamtrask.github.io/2015/07/27/python-network-part2/) that you can read. 

### Backpropagation

So far we have see that we move forward and backward through the network. This movement through the network does the following: 

*   **Forward Propogation**: This is done starting from the input layer, moving through the hidden layers and get to the output layer with a prediction. We calculate the errors. 
*   **Backpropagation**: This is done by moving from the output layer, through the hidden layers to the input layers. This process involves updating weights

We can see this graphically as follows:

<img src="Artificial_Neural_Network.assets/image-20210303114706295.png" alt="image-20210303114706295" style="zoom:80%;" />

<img src="Artificial_Neural_Network.assets/image-20210303114641155.png" alt="image-20210303114641155" style="zoom:80%;" />

The backpropagation updates all the weights at the same time rather than updating each individual weight. 

#### Training a Neural Network

Let's go through the steps taken to train a neural network: 

1.  Randomly initialize the weights to a small numbers, close to 0 but not exactly 0
2.  Input the first observation of your dataset in the input layer each feature in one input node. 
3.  Forward propogate and predict the result, $\hat{y}$
4.  Compare the predicted result to the actual result. Measure the generated error
5.  Back-propogate and update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights. 
6.  Repeat steps 1 - 5. Repeating can be done in two ways: 
    1.  Update the weights after each observation (Reinforcement Learning). 
    2.  Update the weights only after a batch of observations (Batch Learning)
7.  When the whole training set passed through ANN, that makes an epoch. Redo more epochs. 