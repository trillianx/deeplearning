[TOC]



# Course 1: Neural Network & Deep Learning

Deep learning, just like machine learning, has transformed the world as we know it. Deep learning has become very important to the companies and has become a sought after skill. 

## Week 1: Introduction to Deep Learning

There are a total of five courses in this specialization. The five courses are the following: 

*   **Neural Networks & Deep Learning**: In this course we will learn about the neural networks, how to build a deep neural network, and how to train it on data. 
*   **Improving Deep Neural Networks: Hyperparameter tuning, Regularization & Optimization**: In this course we will learn about the tools to fine tune the networks. We will learn tools such as regularization, hypertuing, and and advanced optimization algorithms. 
*   **Structuring your Machine Learning Project**: In this course, we will learn how to structure our ML project. We will learn best practices and learn about end-to-end DL. 
*   **Convolutional Neural Network**: In this course we will talk about CNN. These models are often applied to images. 
*   **Natural Language Processing: Building sequence models**: In this course we will learn models that learn overtime. We will apply these models to sequence data and speech recognition and music generation. 

### What is a Neural Network? 

The term **deep learning** refers to training Neural Networks, sometimes very large neural networks. So, what is a neural network? To understand this, consider an example of predicting housing prices. In the figure below, we have just one variable, `size`. This is the size of a house and the output variable is `price`. So, based on the size of the house, we predict its price.

![IMG_937A569299F8-1](Neural_Network_and_Deep_Learning.assets/IMG_937A569299F8-1.jpeg)

We have few observations marked in purple. We fit a model to the data. For given size, there are no houses and so the price for smaller sizes is zero. It has a non-zero value after a certain point and zero value before that point for a given size. This is an example of very simple neural network. A neural network for this example is drawn as follows: 

![IMG_F52BFCB8878B-1](Neural_Network_and_Deep_Learning.assets/IMG_F52BFCB8878B-1.jpeg)

This neural network takes a single input and predicts an output. The neuron in neural network is represented by the circle. The model that was fit on the data in the above figure is called the **Rectified Linear Unit** or ReLU. The ReLU function is zero for some values and then increases linearly. 

Once we have a single neuron, we can easily stack them into multiple neurons and create a model that is more complex. For example, in the housing example above, rather than just the size, we include more features such as  `# of bedrooms`,  `zipcode`,  `family size`,  `walkability`,  `wealth`, `school quality`. The neural network in this case would look something like this: 



![IMG_EBFE795A9750-1](Neural_Network_and_Deep_Learning.assets/IMG_EBFE795A9750-1.jpeg)

This neural network is quite complex. It is to be noted that input features are the one of the left hand side of the image. The features in between such as `family size` etc...is automatically decided by the neural network. 

We note that each neuron is connected with each other. There are 4 input features on the left. This is known as the **input layer**.  The second neuron is called the **output layer**, which is followed by the output variable, `price`. 

>   The first neurons are called **hidden units** because they are not visible to us. All we have are the inputs and the outputs. 

>   When each neuron is connected with each other, we say that the neural network is said to be **densely connected**. 

The neural network decides which features it wants to take in the middle as long as we give it the features in the input layer. This is the hallmark of neural networks. 

### Supervised Learning with Neural Networks

Supervised learning consists of features and associated labels. In machine learning, we use both the input features and the output label to train the model. This is why we call this type of learning supervised learning. Below is a table containing some examples of supervised learning and associated type of neural networks that are typically used for those type of problems. 

| Input         | Output            | Application         | Type of Neural Network       |
| ------------- | ----------------- | ------------------- | ---------------------------- |
| Home Features | Price             | Real Estate         | Standard Neural Network      |
| Ad, User Info | CTR               | Online Advertising  | Standard Neural Network      |
| Image         | Object            | Photo Tagging       | Convolutional Neural Network |
| Audio         | Text Transmission | Speech Recognition  | Recurrent Neural Network     |
| English       | French            | Machine Translation | Recurrent Neural Network     |
| Image, Radar  | Car Positions     | Autonomous Driving  | Hybrid Neural Network        |

We will learn these types of neural networks in this course. The neural networks are represented pictorally as, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_21F4FF563E95-1.jpeg" alt="IMG_21F4FF563E95-1" style="zoom:30%;" />

<img src="Neural_Network_and_Deep_Learning.assets/IMG_41903E3F8EDD-1.jpeg" alt="IMG_41903E3F8EDD-1" style="zoom:30%;" />

<img src="Neural_Network_and_Deep_Learning.assets/IMG_1519C82CAD23-1.jpeg" alt="IMG_1519C82CAD23-1" style="zoom:30%;" />

Machine learning applications work both for **structured data** and **unstructured data**. The examples of structured data include tables, spreadsheets while audio files or individual words are examples of unstructured data. It has been harder for machines to work with unstructured data. However, thanks to neural networks, working with unstructured data is much more possible. We will learn to apply neural networks to both structured and unstructured data.

### Why is Deep Learning Taking off? 

Deep learning has been around since the 1970s. However, it has been taking off now because of the following figure: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_FCF4B60B428B-1.jpeg" alt="IMG_FCF4B60B428B-1" style="zoom:50%;" />

The figure shows the amount of labeled data in the x-axis and the performance of the model in the y-axis. For traditional ML algorithms, increasing the data does increase the performance in the beginning. However after some amount of data, the performance plateaus. On the other hand, for neural networks, the increase in data continues to increase their performance. Of course, the performance also depends on the complexity of the neural network. The only caveat is that the larger the neural network, the better it does with lots of data but longer it may take to train it. 

In the small training set regime, the traditional ML may be equally good as the neural network. The biggest breakthrough in the domain of AI has been the move from the **sigmoid function** to the **Rectificed Linear Unit (ReLU) function**.

 ![IMG_69F8D6B088C5-1](Neural_Network_and_Deep_Learning.assets/IMG_69F8D6B088C5-1.jpeg)

The ReLU is better suited than the sigmoid function because at the extremities the gradient descent takes a very long time to converge. The shape of the sigmoid function makes it hard. Because of this, the GD algorithm takes a lot of steps to converge. However, in the case of ReLU this does not happen. The ReLU function is also called the **activation function** for the neural network.

>   Why do we need an activation function? An activation function decides, whether a neuron should be activated or not by calculating weighted sum of the bias of it. The purpose of the activation function is to introduce non-linearlity into the output of a neuron.
>
>   The non-linearity is essential in a neural network because without it, a neural network would simply be a linear regression model. 

The other advantage of using neural networks is the iterative process through which they go in order to improve their performance. This is illustrated in the figure below: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_39DF49BCC063-1.jpeg" alt="IMG_39DF49BCC063-1" style="zoom:50%;" />

Faster computation has really helped the researchers to iterate through the loop quickly and implement ideas. 

## Week 2: Neural Network Basics

In this week, we will learn to set up a machine learning problem with a neural network mindset. We will also learn to use vectorization to speed up our models. 

The key concepts that are covered in this week are: 

*   Build a logistic regression model, structured as a shallow neural network
*   Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent
*   Implement computationally efficient, highly vectorized version of the models
*   Understand how to compute derivatives for logistic regression using a backpropagation mindset
*   Be able to implement vectorization across multiple training examples 

### Notation Used

This section describes the notations used for the entire course.

*   $x$: represents a single training example,  $x \in \R^{n_x}$. In other words, the sample $x$ is a single sample that consists of $n_x$ features. 
*   $y$: is the output label. 
*   $(x, y)$: is a single training example. 
*   $X \in \R^{n_x \times m}$: is the input matrix which has $n_x$ features and $m$ samples. This is a $(n_x,m)$ matrix. 
*   $x^{(i)} \in \R^{n_x}$ : is the $i$th example represented as a column vector, $(n_x, 1)$. 
*   $Y \in \R^{n_y \times m}$ is the label matrix which has $n_y$ labels and $m$ samples. This is a $(m, 1)$ matrix
*   $y^{(i)} \in \R^{n_y}$: is the output label for the $i$th example and represented by a single value
*   The superscript $(i)$ will denote the $i$th training example while the superscript $[l]$ will denote the $l$th **training layer**. 
*   $m$: are the number of examples in the dataset
*   $n_x$: the input size, also known as the features
*   $n_y$: the output size (or the number of classes)
*   $n_h^{[l]}$: the number of hidden units of the $l$th layer
*   $L$: the number of layers in the network

### Logistic Regression as a Neural Network

In neural networks, the training set is processed directly without looping through each of the examples. In particular this is also done without the use of `for` loops. Another imporant thing to note about NN is that there are two processes that happen. There is a forward pass from the input to the output which is called the **forward propagation** and then there is a backward pass which is called the **backpropagation**. We will implement this for logisitc regression. 

The logistic regression is a machine learning algorithm that is used for binary classification problems. For example, if we wish to create a model that recognizes cats in the photo. The output of the model should result in two categories: "cats" and "no cats".  Think of the following figure. We give the NN an image and the NN will output either a `0` or a `1`. 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_A5C260F11A63-1.jpeg" alt="IMG_A5C260F11A63-1" style="zoom:50%;" />

Now the image itself is composed of three color filteres, red, blue, and green. Suppose the image is $64 \times 64$ pixels. So, then we will have three images in the three colors or pixel intensities. 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_DB04E4DEA293-1.jpeg" alt="IMG_DB04E4DEA293-1" style="zoom:50%;" />

Now, the image by itself is a single example. We can convert all the pixel colors, which there are $64 \times 64 \times 3 = 12, 288$  of them into a single column vector. This would be our single sample. In other words, we concatenate all the pixel intensities. So, what we have done is to represent a picture by a single vector called the **feature vector** because it has 12,288 features. We represent the dimensions of the features by $n_x = 12288$. Thus, there are 12, 288 features. 

In binary classfication problem, we take the image and convert that into a feature vector $X$ and predict its corresponding label, $y$, which is either `0` or `1`.  

*   For the logistic regression, a $(x, y)$  represents a single training example with $x \in \R^{n_x}$ and $y \in \{0,1\}$, the corresponding label. 

*   The training set will comprise of $m$ training examples: $[(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}),..., (x^{(m)}, y^{(m)})]$
    Note that each $x^{(i)}$ is a feature vector with 12, 288 features. 

*   The lowercase $m$ are the number of training examples. 

*   All the training examples and the features can be concisely represented as follow:
    ![IMG_11E33BB75A8D-1](Neural_Network_and_Deep_Learning.assets/IMG_11E33BB75A8D-1.jpeg)

    So, our matrix, $\bold{X} \in \R^{n_x \times m}$ dimensional matrix. This is slightly different from the traditional machine learning scenario where $\bold{X} = \bold{X}^T$. 

    >   The rows are the features while the columns are the training examples. 

*   The notation for the labels will be used in the following way: 

    ![IMG_BB12E068EB58-1](Neural_Network_and_Deep_Learning.assets/IMG_BB12E068EB58-1.jpeg)

    In this case, $\bold{Y} \in \R^{1 \times m}$. 

Returning to Logistic regression, the problem is as follows: 

We are given a sample vector $x^{(i)}$ and we want to predict its corresponding label $\hat{y}$ such that $\hat{y} = P(y = 1|x)$. Note that the label $y$ or its predicted value $\hat{y}$ are bound between 0 and 1. We make use of the parameters $w \in \R^{n_x}, b \in \R$. As we might do in linear regression, we can express the output as a linear combination of the weights $w$ and the constant $b$:
$$
\hat{y} = w^Tx + b
$$
However, given the nature of linear regression, the output values are going to be real numbers. Instead we wish to limit our outputs to between 0 and 1. This is where the **sigmoid function** comes in. 

![IMG_402A885D5663-1](Neural_Network_and_Deep_Learning.assets/IMG_402A885D5663-1.jpeg)

The sigmoid function is given by the following equation: 
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
We can easily verify that this function is bound between `0` and `1`. With this in mind, we write the output as, 
$$
\hat{y} = \sigma(z) = \sigma(w^Tx + b)
$$
At times it is convenient to absorb the constant, $b$ into the weights by setting $w_0 = b$ and $x_0=1$. By doing so, the above equation then becomes: 
$$
\hat{y} = \sigma(w^Tx)
$$
where $w = [w_0, w_1, ..., w_{n_x}]$ with $w_0 = b$. This is what is typically done in ML.

>    **we will choose to keep $w$ and $b$ separate.** 

As seen from the above equation, Eq. 4, the logistic regression boils down to determining the parameters $w$ and $b$ such that the predictions $\hat{y}$ are as close to the actual labels, $y$. The determination of how close the predictions are to the actual labels is done by the use of **loss function**. 

We could use the RSS as the loss function as we do in linear regression. But this is generally not used because using this in the logisitic regression causes the loss function to be non-convex. A non-convex function is generally hard for GD algorithm to work well. 

The loss function for logistic regression that we use is given by, 
$$
\mathcal{L}(\hat{y}, y) = -[y\ log (\hat{y}) + (1-y)\ log(1-\hat{y})]
$$
We can see that this loss function penalizes the predictions that are not close to the actual label. For example, when $y=1$, the second term is zero. The loss function will be small when $\hat{y}$ approaches $1$ else, it will be large. Similarly, when $y=0$, then the first term is zero and the second term is simply $-log(1-\hat{y})$. Again, the loss function is small when $\hat{y}$ approaches 0 else it will be large. 

The loss function is generally defined for a single sample. The **cost function** is defined as the average or the sum of the loss function across all samples. The cost function for logistic regression is defined as, 
$$
\mathcal{J}(w,b) = \frac{1}{m}\sum_{i=1}^m\mathcal{L}(\hat{y}, y) = -\frac{1}{m}\sum_{i=1}^m(y^{(i)}\ log \hat{y}^{(i)} + (1-y^{(i)})\ log(1-\hat{y}^{(i)}))
$$

Note that $\hat{y} = f(w, b)$, therefore $\mathcal{J} = f(w, b)$. 

### Gradient Descent

The gradient descent allows us to find optimal model parameters such that the cost function is the lowest. Let's see how we can apply the gradient descent algorithm to logistic regression. 

 To illustrate the gradient descent algorithm, consider a one-dimensional case, 



<img src="Neural_Network_and_Deep_Learning.assets/IMG_BB84A6106EC3-1.jpeg" alt="IMG_BB84A6106EC3-1" style="zoom:50%;" />

The curve is the loss function. We see that the curve is smooth and has just one minimum value. Such a curve is called a **convex function**. When a curve has more than one minimum, often called local minima, the curve is **non-convex**. 

The gradient descent for the convex function can start anywhere. The gradient descent of the cost function at a particular point is the slope at that point.  The gradient descent equation is given by, 
$$
w_{update} = w_{old} - \alpha\frac{\partial\mathcal{J}}{\partial w} \\[15pt]
b_{update} = b_{old} - \alpha\frac{\partial\mathcal{J}}{\partial b}
$$
And this is run for all the samples. Let's consider the simple case, 1D and see what happens. Consider the figure below: 

![IMG_EAC647EED6E9-1](Neural_Network_and_Deep_Learning.assets/IMG_EAC647EED6E9-1.jpeg)

In this case, we have just one dimension, so our above equation, Equation 7 becomes: 
$$
w_{update} = w_{old} - \alpha\frac{d\mathcal{J}}{dw}
$$
In the case of A., the derivative, the slope at point A, is positive so the updated $w$ will decrease and move towards the minimum value. On the other hand, for B, the slope is negative so the above equation will increase updated $w$ and move it towards the minimum value.

The $\alpha$ is called the **learning rate**. It is the step that is taken each time in moving towards the minimum value. The learning rate is optimized through training. 

In general, the cost function is not 1D but multi-dimensional so we take the partial derivative of the cost function based on parameters. 

>   In this course we wll use $\partial/\partial w$ will be written as $dw$ in the code.  

### Computation Graph

Deep learning involves the use of forward propogation and backward propogation. The computation graphs aid us in visualizing these two movements in deep learning. To illustrate this point, consider a function, 
$$
J(a, b, c) = 3(a + bc)
$$
We can break the calculation of $J$ as,

 $u = bc; \\ v = a + u;\\ J = 3v$ 

We can illustrate this calculation as follows: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_76F34D2639CF-1.jpeg" alt="IMG_76F34D2639CF-1" style="zoom:50%;" />

The backward propogation is starting from the right and going left which involves the use of derivatives. 

Let's look at various derivatives: 
$$
\frac{\partial J}{\partial v} = 3 \\[10pt]
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a} = 3 \cdot1 = 3 \\[10pt]
\frac{\partial J}{\partial b} = \frac{\partial J }{\partial v }\frac{\partial v}{\partial u}\frac{\partial u}{\partial b} = 3\cdot 1\cdot c = 3c \\[10pt]
\frac{\partial J}{\partial c} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial c} = 3\cdot 1 \cdot b = 3b
$$

>   The backpropogation consists of taking the derivative going back, e.g., taking the derivative of $J$ w.r.t. $v$ results in doing a backpropogation of 1 step. Taking a derivative of $J$ w.r.t. $u$ corresponds to doing a backpropogation of 2 steps and so on. 

When it comes to converting these calculations to code, we wish to simply the notations. The notation we will use are the following: 

*   For derivative for the final output variable, such as the cost function, $J$,  with respect to a given variable, such as $u$ or $v$ or others, we will simply use `d<var>`. 

    For example, $dJ/dv$, as we have above will simply be $dv$. Similarly, $dJ/da$ will be $da$. In other words, when we see a single variable name with a `d` in front, it reflects the derivative of the loss function with respect to that variable. 

The backpropogation will look something like this in our example, 

![IMG_F6C6E266A217-1](Neural_Network_and_Deep_Learning.assets/IMG_F6C6E266A217-1.jpeg)

So, we can see how we can reach back from $J$ all the way to a, we need to compute $\partial J/\partial a$ and so forth. 

So, if we started with $a = 5, b = 3, c=2$, we end up with $da = 3, db = 6, dc=9$ through backpropogation. 

### Logistic Regression Gradient Descent (1 Training Example)

To compute the gradient descent for logistic regression, we first setup the equations we have: 
$$
z = w^Tx + b \\[10pt]
\hat{y} = a = \sigma(z) \\[10pt]
\mathcal{L}(\hat{y},y) = -(ylog\hat{y} + (1-y)log(1-\hat{y})
$$
The above are the set of equations for 1 sample. Let's create a computation graph for two features, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_BA8CA07B1F17-1.jpeg" alt="IMG_BA8CA07B1F17-1" style="zoom:50%;" />

In this case, the model has three parameters, $w_1, w_2, b$. Through backpropogation, we can modify these parameters such that the loss function $\mathcal{L}(a, y)$ is minimized. So, let's compute the derivative of the loss function with respect to these parameters: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_17B69BA0E35D-1.jpeg" alt="IMG_17B69BA0E35D-1" style="zoom:33%;" />

This involves computing each of the separate derivatives. Let's work these out separately: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_CF519B8BD73E-1.jpeg" alt="IMG_CF519B8BD73E-1" style="zoom:33%;" />

Now the second derivative in the chain rule: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_BB254B87E129-1.jpeg" alt="IMG_BB254B87E129-1" style="zoom:33%;" />

Finally, the last derivative in the chain: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_950BDA9EFA96-1-9601314.jpeg" alt="IMG_950BDA9EFA96-1" style="zoom:33%;" />

Which then comes out to be the product of individual results. We can for now write this as a shorthand notation: 
$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial w_1} = x_1dz
$$
Similarly, we have: 
$$
\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial w_2} = x_2dz \\[15pt]
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial b} = 1dz
$$
We can of course compute $dz$ as, 



<img src="Neural_Network_and_Deep_Learning.assets/IMG_0252F5562358-1.jpeg" alt="IMG_0252F5562358-1" style="zoom:33%;" />

And therefore, the updates will be: 
$$
dz = a - y \\[15pt]
w_{1updated} = w_1 -\alpha(x_1dz) \\[15pt] 
w_{2updated} = w_2 -\alpha(x_2dz) \\[15pt]
b_{updated} = b -\alpha(dz) \\[15pt]
$$

### Logistic Regression Gradient Descent ($m$ Training Example)

Logistic regression example we did above was used for one training example. In this section, we expand this to include $m$ training examples. Note that we still have two features $x_1, x_2$ and associated weights, $w_1, w_2$ and the bias, $b$. 

The code for setting this for $m$ examples would be: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_14FAF1EDB43B-1.jpeg" alt="IMG_14FAF1EDB43B-1" style="zoom:33%;" />

This would be the code for doing one pass through all of the samples. In order to do this multiple times, we will need to write another loop that goes through multiple passes. Another drawback of this code is that the weights are updated for just two features. If we have $n$ features, we will need to write a look that will update all the weights in a single pass. As you can see there are far too many nested loops to do this quickly. That is where **vectorization** becomes incredibly important. Vectorization allows us to do the same calculations without the need for for loop. 

### Vectorization

Vectorization of code allows us to do computation much faster. This is especially important when we have a lot of data. The vectorization we have seen in the logisitic regression is the computation of the dot product, $z = w^tx + b$. There are two ways to compute the dot product: using the `for` loop and through vectorization by implementing the `numpy` arrays. 

```python
# Vectorization version: 
c = np.dot(w, x) + b

# Non-vectorization version
for i in range(len(x)):
    z += w[i]*x[i]
z += b
```

If we were to time this, we will see that the difference between vectorization and non vectorization is 300x. Therefore, vectorization is very important in deep learning. 

Coming back to our code for logistic regression, we had only used two features. If we used more features, $n$, then we need to use another loop. However, we can make a small change and completely avoid that loop as follows: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_3E2402B01494-1.jpeg" alt="IMG_3E2402B01494-1" style="zoom:33%;" />

In the next section we will see how we can make the code even more efficient. 

### Vectorizing Logistic Regression

Having sold one part of the problem to remove a for loop, we attempt to remove the second for loop that runs through all of the $m$ examples. 

Now, note that in the equation, 
$$
z^{(i)} = w^Tx^{(i)} + b
$$
we compute the $z$ for each sample. This is what we have for the first row after the for loop in the above figure. Here, $x^{(i)}$ is a feature vector for the $i$th sample. The weight $w$ is also a vector with elements equal to that of the dimensions of $x^{(i)}$. But rather than multiplying each row of features of a given sample, we can put all the features and all samples into a matrix. Something like this, 

 <img src="Neural_Network_and_Deep_Learning.assets/IMG_A003DB7B53B6-1.jpeg" alt="IMG_A003DB7B53B6-1" style="zoom:33%;" />

So, we have $m$ samples or observations. There are a total of $n_x$ features. The row that is shown corresponds to a given feature across all the $m$ samples. The above equation can then be written as, 
$$
Z = W^TX + B
$$
where $X$ is a matrix $X \in \R^{n_x \times m}$, $W$ is a column vector, $W \in \R^{n_x \times 1}$ and $B$ is a column vector, $B \in \R^{1 \times 1}$.  Equation 16 then corresponds to a matrix multiplication with a vector and addition with a scalar. 

So, we go from, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_F75A99A9E8DE-1.jpeg" alt="IMG_F75A99A9E8DE-1" style="zoom:33%;" />

To this, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_1BC51208FE92-1.jpeg" alt="IMG_1BC51208FE92-1" style="zoom:33%;" />

Note that $b$ is a single element but through **broadcasting** in Python, it is made into a row vector with dimensions equal to the $n_x$. Let's check the dimensions, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_3DFEBB8DFE2C-1.jpeg" alt="IMG_3DFEBB8DFE2C-1" style="zoom:33%;" />

In python this operation is performed as follows, 

```python
z = np.dot(w.T, X) + b
```

That it. This gives us all the $z$ values across all the samples. Similarly, the prediction, $a$ can be computed as: 

```python
a = 1/(1 + np.exp(-z))
```

This will result in $1 \times m$ dimension vector, which is what we expect. 

Next, let's see how we can vectorize the backpropogation. 

The first one is $dz$. This is very easy to do as we already have $a$ and $y$ and row vectors. We simply subtract one from each other to get a row vector. 

```python
dz = a - y
```

Next are the updates to $b$ and $w$: 

In the case of $db$, we have effectively, $db = 1/m \sum_{i=1}^m dz^{(i)}$, which we can write as: 

```python
db = (np.sum(dz))/m
```

For the updates to weights, we have: $dw = (1/m) X \cdot dz^T$. which we can write as, 

```python
dw = (np.dot(X, dz^T))/m
```

With that we can write the python code as follows, 

```python
J = 0, dw = np.zeros(n,1)
# Forward Propogation: 
Z = np.dot(w.T, X) + b
A = 1/(1 + np.exp(Z))
# Back Propogation: 
dz = A - Y
dw = 1/m(np.dot(X, dz.T))
db = 1/m(np.sum(dz))
w = w - alpha * dw
b = b - alpha * db
```

We have gotten rid of the for loops but for going through multiple iterations of gradient descent, we will need to implement a for loop. 

### Broadcasting in Python

Suppose we have the following table of food and associated macronutrients in 100g of the food: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20200909143956303.png" alt="image-20200909143956303" style="zoom:50%;" />

Rather than computing the grams of each macronutrient, we wish to compute their percentage of total weight. We can easily do this using vectorization as follows: 

```python
import numpy as np

# Set up the matrix: 
 calories = np.matrix([[56, 1.2, 1.8],
                       [0.0, 104, 135],
                       [4.4, 52, 99],
                       [68, 8, 0.9]])
# Each of the array is actually a row, so we take a transpose of it:
calories = calories.T
print(calories)
```

The output looks something like this: 

```python
matrix([[ 56. ,   0. ,   4.4,  68. ],
        [  1.2, 104. ,  52. ,   8. ],
        [  1.8, 135. ,  99. ,   0.9]])
```

Next we take the sum of each of the columns: 

```python
sum_val = np.sum(calories, axis=0)
```

When `axis=1`, we would take sum of the rows rather than the columns. 

Finally, we divide the `sum_val` with the matrix: 

```python
result = 100 * np.round(calories / sum_val.reshape(1, 4),2)
print(result)
```

```python
array([[95.,  0.,  3., 88.],
       [ 2., 44., 33., 10.],
       [ 3., 56., 64.,  1.]])
```

Note that when we divided the `sum_val` with a matrix, we **cast** it as a 1 by 4 matrix. This is not required as `sum_val` is already in the correct dimension. However, if one is unsure, it is important to use the `.reshape()` method. 

So, one thing is clear. How did we divide a $3 \times 4$ matrix with a $1 \times 4$ matrix? This is where the **broadcasting** comes into picture. Through broadcasting, Python took the $1 \times 4$ matrix and converted it to a $3 \times 4$ matrix, thus making the calculation possible. It did so by repeating the row, two times. 

>   *   If you have a matrix with dimensions, $(m, n)$ and do any mathematical operation with a $(1, n)$ matrix or with $(m, 1)$ matrix, you will get a $(m, n)$ matrix
>   *   If you have a matrix with dimension $(m, 1)$ (or a $(1, m)$) and do a mathematical operation with a scalar, you will end with a $(m, 1)$ or $(1, m)$ matrix. 



### Quiz: Week 2

1.  What does a neuron compute in logistic regression? 

2.  What is the logistic loss function? 

3.  Give me an example of `.reshape()` where we have 4 elements in a column

4.  Given the following: 

    ```python
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a + b
    ```

    What are the dimensions of c? 

5.  Given the following: 

    ```python
    a = np.random.randn(4, 3) # a.shape = (4, 3)
    b = np.random.randn(3, 2) # b.shape = (3, 2)
    c = a*b
    ```

    What are the dimensions of c? 

6.  Suppose you have $n_x$ input features per example. Recall that $X = [x^{(1)} x^{(2)} ... x^{(m)}]$ What is the dimension of X?

7.  Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?



### Answers

1.  The neuron computes the linear function followed by the activation function
2.  <img src="Neural_Network_and_Deep_Learning.assets/image-20200914152052867.png" alt="image-20200914152052867" style="zoom:50%;" />
3.  `a.reshape(1,4)`
4.  `c.shape(2,3)`
5.  `Not Possible`
6.  $(n_x, m)$
7.  (32 x 32 x 3, 1)

