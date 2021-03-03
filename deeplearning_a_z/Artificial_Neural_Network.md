# Artificial Neural Network

In this part we will learn: 

1.  [The intuition of ANNs](## ANN Intuition)
2.  How to Build an ANN
3.  How to predict an outcome of a single observation

For more in-depth material, check out [The Ultimate Guide to Artificial Neural Netowrk (ANN)](https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann)

## ANN Intuition

In this section, we will begin into the depth of ANN. Here are few things we will cover in this section: 

*   [The Neuron](### The Neuron)
*   [The Activation Function](The Activation Function)
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
