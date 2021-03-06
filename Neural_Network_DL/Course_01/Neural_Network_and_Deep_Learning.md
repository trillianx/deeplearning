[TOC]



# Course 1: Neural Network & Deep Learning

Deep learning, just like machine learning, has transformed the world as we know it. Deep learning has become very important to the companies and has become a sought after skill. 

## Week 1: Introduction to Deep Learning

There are a total of five courses in this specialization. The five courses are the following: 

*   **Neural Networks & Deep Learning**: In this course we will learn about the neural networks, how to build a deep neural network, and how to train it on data. 
*   **Improving Deep Neural Networks: Hyperparameter tuning, Regularization & Optimization**: In this course we will learn about the tools to fine tune the networks. We will learn tools such as regularization, hypertuning, and advanced optimization algorithms. 
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

This neural network is quite complex. It is to be noted that input features are the one of the left hand side of the image. The features in between such as `family size` etc... are automatically decided by the neural network. The user has no input into that.

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

## Quiz 1

1.  What does the analogy "AI is the new electricity" to?
2.  Which of these reasons for Deep Learning recently taking off? 
    1.  We have access to a lot more data.
    2.  Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition. 
    3.  We have access to a lot more computational power.
    4.  Neural networks are a brand new field. 
3.  Recall the diagram of iterating over different ML ideas. Which of the statements below are true? 
    1.  Being able to try out ideas quickly allows deep learning engineers to iterate more quickly. 
    2.  Faster computation can help speed up how long a team takes to iterate to a good idea. 
    3.  It is faster to train on a big dataset than a small dataset. 
    4.  Recent progress in deep learning algorithms has allowed us to train good models faster (e.g. CPU/GPU)
4.  When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False?
5.  Which of these plots represents a ReLU activation function?
6.  Images for cat recognition is an example of “structured” data, because it is represented as a structured array in a computer. True/False?
7.  A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data because it contains data coming from different sources. True/False?
8.  Why is an RNN (Recurrent Neural Network) used for machine translation, say translating English to French? (Check all that apply.)
    1.  It can be trained as a supervised learning problem.
    2.  It is strictly more powerful than a CNN
    3.  It is applicable when the input/output is a sequence
    4.  RNNs represent the recurrent process of Idea->Code->Experiment-Idea->...
9.  In the diagram that shows various neural networks and traditional machine learning, what is the x-axis and y-axis?
10.  Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right),  which of the following are true? (Check all that apply.)
     1.  Increasing the size of a NN generally does not hurt an algorithm's performance, and it may help significantly
     2.  Decreasing the training set size generally does not hunt an algorithm's performance, and it may help significantly. 
     3.  Decreasing the size of a NN generally does not hurt an algorithm's performance, and it may help significantly
     4.  Increasing a training set size generally does not hurt an algorithm's performance, and it may help significantly



### Answers 

1.  Similar to electricity starting about 100 years ago, AI is transforming multiple industries
2.  Points, 1, 2, and 3.
3.  Points 1, 2 and 4.
4.  False
5.  The one that is zero for all negative values and then linear (y=x) for all positive values. 
6.  False
7.  False
8.  1 and 3.
9.  X-axis: Amount of data; Y-axis: Algorithm performance
10.  1 and 4.



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
*   $x^{(i)} \in \R^{n_x}$ : is the $i$th observation or instance represented as a column vector, $(n_x, 1)$. 
*   $Y \in \R^{n_y \times m}$ is the label matrix which has $n_y$ labels and $m$ samples. This is a $(m, 1)$ matrix
*   $y^{(i)} \in \R^{n_y}$: is the output label for the $i$th example and represented by a single value
*   The superscript $(i)$ will denote the $i$th training example while the superscript $[l]$ will denote the $l$th **training layer**. 
*   $m$: are the number of examples in the dataset
*   $n_x$: the input size, also known as the features
*   $n_y$: the output size (or the number of classes)
*   $n_h^{[l]}$: the number of hidden units of the $l$th layer
*   $L$: the number of layers in the network

### Logistic Regression as a Neural Network

In neural networks (NN), the training set is processed directly without looping through each of the examples. In particular this is also done without the use of `for` loops. Another imporant thing to note about NN is that there are two processes that happen. There is a forward pass from the input to the output which is called the **forward propagation** and then there is a backward pass from the output to the inputs which is called the **backpropagation**. We will implement this for logisitc regression. 

The logistic regression is a machine learning algorithm that is used for binary classification problems. For example, if we wish to create a model that recognizes cats in the photo. The output of the model should result in two categories: "cats" and "no cats".  Think of the following figure. We give the NN an image and the NN will output either a `0` or a `1`. 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_A5C260F11A63-1.jpeg" alt="IMG_A5C260F11A63-1" style="zoom:50%;" />

Now the image itself is composed of three color filters: red, blue, and green. Suppose the image is $64 \times 64$ pixels. So, then we will have three images in the three colors or pixel intensities. 

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
>   This is a vectorized form. So, all the features are updated at the same time when we go through each iteration. The above is run multiple times until convergence. 

 Let's consider the simple case, 1D and see what happens. Consider the figure below: 

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

The computation graph comes in handy when you wish to optimize an output variable such as $J$ in the case. In the case of logistic regression $J$ is the cost function that we are trying to minimize. 

What we see here is that from left-to-right pass, we compute the value of $J$ and as we will see in the next few steps, the right-to-left pass, we compute the derivatives. 

The backward propogation is starting from the right and going left which involves the use of derivatives. 

Let's look at various derivatives: 
$$
\frac{\partial J}{\partial v} = 3 \\[10pt]
\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a} = 3 \cdot1 = 3 \\[10pt]
\frac{\partial J}{\partial b} = \frac{\partial J }{\partial v }\frac{\partial v}{\partial u}\frac{\partial u}{\partial b} = 3\cdot 1\cdot c = 3c \\[10pt]
\frac{\partial J}{\partial c} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial c} = 3\cdot 1 \cdot b = 3b
$$

>   The backpropogation consists of taking the derivative going back, e.g., taking the derivative of $J$ w.r.t. $v$ results in doing a backpropogation of 1 step. Taking a derivative of $J$ w.r.t. $u$ corresponds to doing a backpropogation of 2 steps and so on. 

When it comes to converting these calculations to code, we wish to simplify the notations. The notation we will use are the following: 

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

In this case, the model has three parameters, $w_1, w_2, b$. Through backpropogation, we wish to modify the features weights such that the loss function $\mathcal{L}(a, y)$ is minimized. So, we have the following relationships for the feature weights: 
$$
dw_1 = \frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w_1} \\[15pt]

dw_2 = \frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w_2} \\[15pt]

db = \frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial b} \\[15pt]

dz = \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a}\frac{\partial \mathcal{a}}{\partial z}
$$
These are the main equations that connect the loss function, $\mathcal{L}(a, y)$ with the weights, $w_1$, $w_2$ and the intercept, $b$.  

So, let's compute the derivative of the loss function with respect to these parameters: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_17B69BA0E35D-1.jpeg" alt="IMG_17B69BA0E35D-1" style="zoom:33%;" />

This involves computing each of the separate derivatives. Let's work these out separately: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_CF519B8BD73E-1.jpeg" alt="IMG_CF519B8BD73E-1" style="zoom:33%;" />

Now the second derivative in the chain rule: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_BB254B87E129-1.jpeg" alt="IMG_BB254B87E129-1" style="zoom:33%;" />

Finally, the last derivative in the chain: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_950BDA9EFA96-1-9601314.jpeg" alt="IMG_950BDA9EFA96-1" style="zoom:33%;" />

Which then comes out to be the product of individual results. We can for now write this as a shorthand notation: 
$$
\frac{\partial \mathcal{L}}{\partial w_1} = dw_1 = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial w_1} = x_1dz
$$
Similarly, we have: 
$$
\frac{\partial \mathcal{L}}{\partial w_2} = dw_2 = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial w_2} = x_2dz \\[15pt]
\frac{\partial \mathcal{L}}{\partial b} = db = \frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial b} = 1dz
$$
We can of course compute $dz$ as, 



<img src="Neural_Network_and_Deep_Learning.assets/IMG_0252F5562358-1.jpeg" alt="IMG_0252F5562358-1" style="zoom:33%;" />

And therefore, the updates will be: 
$$
dz = a - y \\[15pt]
w_{1updated} = w_1 - \alpha \left(\frac{\partial \mathcal{L}}{\partial w_1}\right)= w_1 -\alpha(x_1dz) = w_1 - \alpha x_1(a-y)\\[15pt] 
w_{2updated} =  w_2 - \alpha \left(\frac{\partial \mathcal{L}}{\partial w_2}\right) = w_2 -\alpha(x_2dz) = w_2 - \alpha w_2(a-y)\\[15pt]
b_{updated} = b -\alpha(dz) = b - \alpha (a-y)\\[15pt]
$$

We now see the use of backpropogation. The backpropogation is used to change the feature weights in order to reduce the cost function. 

### Logistic Regression Gradient Descent ($m$ Training Example)

Logistic regression example we did above was used for one training example. In this section, we expand this to include $m$ training examples. Note that we still have two features $x_1, x_2$ and associated weights, $w_1, w_2$ and the bias, $b$. 

The code for setting this for $m$ examples. Here the example assumes two features and associated two weights.  

```python
def sigma(z):
    result = 1/(1+math.exp(z))
    return result

J = 0
dw1 = 0
dw2 = 0
db = 0


# We go through each training example: 
for i in range(len(m)):
    z[i] = w[i] * x[i] + b
    a[i] = sigma(z[i])
    J += -(y[i] * math.log(a[i]) + (1-y[i]) * log(1-a[i]))
    
    # Now the updates: 
    dz[i] = a[i] - y[i]
    dw1 += x[i] * dz[i]
    dw2 += x[i] * dz[i]
    db += dz[i]

# Finally divide everything by m
J = J/m
dw1 = dw1 / m
dw2 = dw2 / m
db = db / m

# Now we update the weights: 
w1 = w1 - alpha * dw1
w2 = w2 - alpha * dw2
b = b - alpha * db
```

All of this is done for one pass of the gradient descent. So, we have to encase this code from line 11 to the end in another for loop to run the gradient descent through multiple epochs. 

 If we have $n$ features, we will need to write a loop that will update all the weights in a single pass. As you can see there are far too many nested loops to do this quickly. That is where **vectorization** becomes incredibly important. Vectorization allows us to do the same calculations without the need for for loop. 

### Vectorization

Vectorization of code allows us to do computation much faster. This is especially important when we have a lot of data. The vectorization we have seen in the logisitic regression is the computation of the dot product, $z = w^Tx + b$. There are two ways to compute the dot product: using the `for` loop and through vectorization by implementing the `numpy` arrays. 



```python
# Non-vectorized form:
z = 0
for i in range(n):
    z += w[i] * X[i]
z += b
```

This would be really slow. In python the above operation is performed as follows, 

```python
z = np.dot(w, X) + b
```

Let's see this in action: 

```python
import numpy as np
import time

a = np.random.randn(1000000)
b = np.random.randn(1000000)

def non_vectorized():
    start = time.time()
    z = 0
    for i in range(len(a)):
        z += a[i] * b[i]
    end = time.time()
    return end - start

def vectorized():
    start = time.time()
    z = np.dot(a, b)
    end = time.time()
    return end - start

if __name__ == '__main__':
    nv = []
    v = []
    for i in range(100):
        nv.append(non_vectorized())
        v.append(vectorized())
    print("Average Non-vectorized: ", np.mean(nv))
    print("Average Vectorized: ", np.mean(v))
```

And here is the response: 

```python
Average Non-vectorized:  506.9623470306397
Average Vectorized:  6.353855133056641
```

The total time taken is in milliseconds. We see that the vectorized form is nearly 100 times faster than the non-vectorized version. 

Therefore, it is important to vectorize your code when working with deep learning. Another advantage of vectorization is that the good can be parallelized and will run much faster when working with GPUs. The CPUs can also do this, GPUs are designed to make the calculations parallel. 

>   When possible avoid explicit for loops

Let's look at another example. Suppose, you have the following product: 
$$
u = exp(v)
$$
Here, $v$ is a n-dimensional vector. So, in order to carry out this operation, we will need to use a for loop and store the information in an array. It would be something like this: 

```python
# Non-vectorized form
u = []
for i in range(len(v)):
    u.append(math.exp(v[i]))
  
# Vectorized form:
u = np.exp(v)
```

### Vectorizing Logistic Regression

Having solved one part of the problem to remove a for loop, we attempt to remove the second for loop that runs through all of the $m$ examples. 

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
where $X$ is a matrix $X \in \R^{n_x \times m}$, $W$ is a column vector, $W \in \R^{n_x \times 1}$ and $B$ is a column vector, $B \in \R^{1 \times 1}$.  Equation 18 then corresponds to a matrix multiplication with a vector and addition with a scalar. 

So, we go from, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_F75A99A9E8DE-1.jpeg" alt="IMG_F75A99A9E8DE-1" style="zoom:33%;" />

To this, which is taking a transpose of $w$: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_1BC51208FE92-1.jpeg" alt="IMG_1BC51208FE92-1" style="zoom:33%;" />

Note that $b$ is a single element but through **broadcasting** in Python, it is made into a row vector with dimensions equal to the $n_x$. Let's check the dimensions, 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_3DFEBB8DFE2C-1.jpeg" alt="IMG_3DFEBB8DFE2C-1" style="zoom:33%;" />

We would then write the equation 18 in python as: 

```python
z = np.dot(W.T, X) + b
```

That's it. This gives us all the $z$ values across all the samples. Similarly, the prediction, $a$ can be computed as: 

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
dw = (np.dot(X, dz.T))/m
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

# Gradient Descent update
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

### Numpy Vectors

Broadcasting can be confusing at times. So, let's see how we can avoid this confusion. 

Consider the following example: 

```python
import numpy as np

a = np.random.randn(5)
```

This creates a vector with 5 randomly generated values: 

```python
array([-1.02353477, -0.0083769 ,  0.4808782 , -2.23798533,  0.85900304])
```

If you look at the shape of this vector, you will find: 

```python
a.shape
(5,)
```

This is neither a column or a row. So, doing a transpose will have no effect on this vector. So, it is important not to use such a data structure. Instead, use the following: 

```python
b = np.random.randn(5, 1)
```

This gives us a column vector. The shape agrees with what we created: 

```python
b.shape

(5, 1)
```

And this is how it looks: 

```python
array([[-0.15191493],
       [ 1.08354514],
       [ 0.21466363],
       [ 1.32826618],
       [ 0.57910842]])
```

Doing a transpose now works: 

```python
b.T
array([[-0.15191493,  1.08354514,  0.21466363,  1.32826618,  0.57910842]])
```

In short, make sure to explicitly state the right dimension of a vector. In other words, commit to creating a row or a column vector explicitly. 

You can also use `.reshape()` to convert a vector into a form you want. 

### Python Basics with Numpy Lab

#### Sigmoid Function

Let's build the sigmoid function

```python
# Basic function

def basic_sigmoid(x):
    s = 1/(1 + math.exp(-x))
    return s
```

We hardly use `math` package in deep learning because the the above function expects a single value. When working in deep learning, often we pass an array. Passing an array in the above function results in type error. 

Let's implement a sigmoid function using Numpy

```python
# Function using Numpy

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s
```

This function can take an array. For example, 

```python
x = np.array([1, 2, 3])
sigmoid(x)

array([ 0.73105858,  0.88079708,  0.95257413])
```

#### Gradient of Sigmoid

The gradient of the sigmoid is given by, 
$$
\sigma^{'}(x) = \sigma(x)(1 - \sigma(x))
$$

```python
# Gradient of Sigmoid Function

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid
    function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use
    it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s*(1-s)
    return ds
    
```

#### Reshaping Arrays

Two common numpy functions used in deep learning are:

*    `np.shape` used to get the shape of an array 
*   `np.reshape()` used to reshape an array into some other dimensions

Implement `image2vector()` that takes an input of shape (l, h, 3) and returns a vector of shape (l\*h\*3, 1). 

```python
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    length = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]
    v = image.reshape(length * height * depth, 1)
    return v
```

#### Normalizing Rows

Normalization of vectors is performed by dividing the components of the vector by its norm. So, given a matrix, 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210211145445353.png" alt="image-20210211145445353" style="zoom:80%;" />

The normalized matrix will have the following form: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210211145515608.png" alt="image-20210211145515608" style="zoom:80%;" />

Where each element in the above matrix was divided by the norm, $||x|| = 5/\sqrt{56}$. 

Let's implement `normalizeRows()` function to normalize the rows of a matrix. 

```python
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit
    length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / x_norm
    return x_normalized
```

#### Softmax Function

Implement a softmax function using numpy. You can think of softmax as a normalizing function used when your algorithm needs to classify two or more classes. You will learn more about softmax in the second course of this specialization.

**Instructions**:

<img src="Neural_Network_and_Deep_Learning.assets/image-20210211150629628.png" alt="image-20210211150629628" style="zoom:150%;" />

```python
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. 
    # Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically 
    # use numpy broadcasting.
    s = x_exp/x_sum

    ### END CODE HERE ###
    
    return s
```

**What you need to remember:**

*   `np.exp(x)` works for any `np.array(x)` and applies the exponential function to every coordinate
*   the sigmoid function and its `gradientimage2vector()` is commonly used in deep learning
*   `np.reshape` is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. 
*   numpy has efficient built-in functions
*   broadcasting is extremely useful

#### Implement L1 and L2 loss functions

Let's implement regularization functions using lumpy: 

```python
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(y-yhat))
    
    return loss
```

So, if we are given, 

```python
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

1.1
```

Let's implement L2 norm: 

```python
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.sum((y - yhat)**2)
    
    return loss
```



### Quiz: Week 2

1.  What does a neuron compute in logistic regression? 

2.  What is the logistic regression loss function? 

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

8.  Consider the following code snippet:

    ```python
    # a.shape = (3,4)
    # b.shape = (4,1)
    
    for i in range(3):
      for j in range(4):
        c[i][j] = a[i][j] + b[j]
    ```

    How do you vectorize this? 

9.  Consider the following code:

    ```python
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a*b
    ```

    What will be c?

10.  Consider the following computation graph: 
     <img src="Neural_Network_and_Deep_Learning.assets/image-20210308100414735.png" alt="image-20210308100414735" style="zoom:67%;" />

     What is the output of J?



### Answers

1.  The neuron computes the linear function followed by the activation function

2.  <img src="Neural_Network_and_Deep_Learning.assets/image-20200914152052867.png" alt="image-20200914152052867" style="zoom:50%;" />

3.  `a.reshape(1,4)`

4.  `c.shape(2,3)`

5.  `Not Possible`

6.  $(n_x, m)$

7.  (32 x 32 x 3, 1)

8.  ```python
    c = np.dot(a.T, b)
    ```

9.  This will invoke broadcasting so b is copied three times to become (3,3) and * invokes a matrix multiplication operation of two 3x3 matrices so c.shape will be (3,3)

10.  The output is: 
     $$
     J = (a*b) + (a*c) - (b+c) \\[15pt]
     = a(b + c) - (b + c) \\[15pt]
     = (a - 1)(b + c)
     $$
     

### DL Lab Week 2

In the lab this week we implemented a NN for identifying whether a photo is that of a cat or not. We made use of the logisitic regression and the sigmoid activation function. These were done in separate steps and were then added together in a single function. Let's dive down to these in more detail. 

For this lab, the mathematics behind running a NN for a single instance, $x^{(i)}$ is the following: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210310085525261.png" alt="image-20210310085525261" style="zoom:50%;" />

Once the above are computed for all training examples, we would then take the mean of the cost functions. 

#### Building the Part NN

The main steps for building a NN are:

1.  Define the model structure (such as number of input features)
2.  Initialize the model's parameters
3.  Loop:
    1.  Calculate current loss (forward propagation)
    2.  Calculate current gradient (backward propagation)
    3.  Update parameters (gradient descent)

We start by first creating helper functions. 

##### Helper Functions

The activation function we will use is the sigmoid: 

```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1 + np.exp(-z))
    
    return s
```

##### Initialize the Parameters

We initialize the weights and the bias: 

```python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
  
    w = np.zeros(dim).reshape(dim, 1)
    b = 0

    # Check to see that the initialization worked
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```

##### Forward and Backward Propagation

Now we write a function to compute the forward and backward propagation. We will use the two formulas: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210310090107520.png" alt="image-20210310090107520" style="zoom:80%;" />

We will associate these with two variables `dw` and `db`. 

```python
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    # Get the total sample length
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X) + b
    # compute activation
    A = sigmoid(z)
    # compute cost
    cost = (-1/m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))     
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*(np.dot(X,(A - Y).T))
    db = (1/m) * np.sum(A - Y)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```

##### Optimization

Now that we have created a pipeline to run the forward and backward propogation, we need to use gradient descent to find the optimal values of weights and bias: 

```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```

##### Implement the Predict Method

Now that we have trained our NN, we implement a predict method: 

```python
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat
    # being present in the picture
    z = np.dot(w.T, X) + b
    A = sigmoid(z) 

    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction
```

##### Create a Final Model

We can put all the above functions into a single function to run the whole thing: 

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # initialize parameters with zeros (≈ 1 line of code)
    w = np.zeros(X_train.shape[0]).reshape(X_train.shape[0], 1)
    b = 0

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 -
                  np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 -
                  np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```

This completes the assignment for Week 2.

## Week 3: Shallow Neural Network

In this week, we will learn to build a neural network with one hidden layer, using forward propagation and backpropagation. The learning objectives in this week are: 

-   Describe hidden units and hidden layers
-   Use units with a non-linear activation function, such as tanh
-   Implement forward and backward propagation
-   Apply random initialization to your neural network
-   Increase fluency in Deep Learning notations and Neural Network Representations
-   Implement a 2-class classification neural network with a single hidden layer

In the previous week, we saw how we can implement logistic regression as a neural network for binary classification. In this calculations we had the following neural network:

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214094554492.png" alt="image-20210214094554492" style="zoom:80%;" />

We start with three features. These go into the neuron. Two steps are performed in the neuron: (1) the weighted sum of features and the bias. $z$,  and (2) the application of activation function on $z$. Finally, the cost function is computed. 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214094645812.png" alt="image-20210214094645812" style="zoom:80%;" />



### Neural Network in this Week

The above diagram is a scematic presentation of a neural network. What we now wish to introduce is something a little more involved and akin to what you would see in a NN. The neural network generally consists of multiple neurons. So, rather than the features passing the information to a single neuron we saw earlier, they pass to multiple neurons as we see in the figure below.

![IMG_B8AF9B0CD9F5-1](Neural_Network_and_Deep_Learning.assets/IMG_B8AF9B0CD9F5-1.jpeg)

You will notice here that the features not only go through the neurons but there is an interaction that happens between features at each of the neurons. The bunch of neurons right after the input is called the **first layer**. Each neuron in this layer performs the two calculations based on the initial weights and the intercept: 

1.  Computation of $z$
2.  Computation of $\sigma(z)$

Now the result from this step goes into the **second layer**. So, in our example, each neuron from the first layer return $\sigma(z)$ and these become inputs to the neuron in the second layer. Each of these inputs have their own associated weights. This single neuron does exactly the same thing: it takes the weighted sum of the inputs and passes that through another activation function. This result is then our prediction, $\hat{y}$. 

We will denote weights and intercept used in this layer by a superscript value:

![IMG_12DADBA2EC4E-1](Neural_Network_and_Deep_Learning.assets/IMG_12DADBA2EC4E-1.jpeg)

In this case we have two layers. The first layer uses the initial weights and the intercept to calculate $z$ and $\sigma(z)$. This is something we are well familiar with: 

![IMG_90725955D9C7-1](Neural_Network_and_Deep_Learning.assets/IMG_90725955D9C7-1.jpeg)

But notice how the first layer outputs $a^{[1]}$, which is nothing but the activation coming from the first layer. This now takes the roll of $x$ as it did in the first layer. In other words, the $z^{[2]}$ is simply the weighted sum of the activations.  The second layer computes $z$ and $\sigma(z)$ based on new weights it has computed from the first layer. 

>   $z^{[2]}$ is simply the weighted sum of the activations. 

The computation is then passed to the loss function. 

The backward propogation happens by going through the second layer first where the weights associated with the activation functions are changed slightly and then going through the first layer where the weights associated with the features are changed slightly. 

>   The superscripts square brackets are used to identify the layer while the superscript round brackets are used to identify the $i$th sample. 

### Neural Network Representation

Let's dig deeper and see what a two-layer neural network with a single hidden layer looks like.

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214102723841.png" alt="image-20210214102723841" style="zoom:60%;" />

This 2-layer neural network has the following components: 

*   **Input Layer** - This is the layer where all the features go into the neural network
*   **Hidden Layer** - This layer is "hidden" from the user. It does the calculations based on the input values.
*   **Output Layer** - This layer is the layer where the loss function is computed and the prediction is given back to the user. 

We will use another set of convention to make our life easier for computation. 

#### Activation Functions

We will use the symbol $a^{l}$ to represent a set of activations. For the input layer the activation is defined with $l=0$. 

Now let's use the above figure to construct a NN with a hidden layer.  The above figure has three features. These form a column vector as seen below: 
$$
a^{[0]} = \begin{bmatrix} 
x_1\\
x_2\\
x_3 \\
\end{bmatrix}
$$
We associate that to an activation function $a^{[0]}$.  Next we look at the hidden layer.

The hidden layer will generate a set of activations based on the input layer. We have chosen to take 4 neurons or units in the hidden layer. Each unit will generate an activation function which will be associated with weights and a bias. Thus overall, we can write the activation function of the hidden layer as a column vector of individual unit activation function:
$$
a^{[1]} = f(w^{[1]},b^{[1]}) = \begin{bmatrix} 
a^{[1]}_1\\
a^{[1]}_2\\
a^{[1]}_3 \\
a^{[1]}_4 \\
\end{bmatrix}
$$
Finally, we have the activation corresponding to the output layer. This is our prediction:
$$
a^{[2]} = f(w^{[2]},b^{[2]}) = \hat{y}\\
$$
>   The current example is a 2-layer NN because the input layer is not counted. 

Notice that the hidden layer and output layer both have two parameters: $w, b$. These are the weights and the bias. 

#### Dimensions of Activation Functions

What are the dimensions of these activations? In our example, the dimensions would be: 

*   **Input Layer**: $(3,1)$. This is easy to understand. We have three input features. So, the input layer is a column vector.
*   **Hidden Layer**:  is a function of feature weights with dimension (4, 3) . It has these dimensions because there are 3 weights corresponding to 3 features. In addition, it has 4 neurons. In a similar manner, the dimension of the intercept will be (4,1).
*   **Output Layer**: is a function of feature weights so the dimension would be (1,4) and the intercept so the output would be (1,1). 

We will go over in the next section about what each of the activations do. 

#### Computing a Neural Network's Output

As we saw in the previous week, the neural network for logistic regression that we created in the previous week computes two things: 

*   $z = w^Tx + b$
*   $a = \sigma(z)$

We represent this as following: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214105701818.png" alt="image-20210214105701818" style="zoom:80%;" />

The neural network does this a multiple times. 

Let's look at the activation functions in the hidden layers: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214105937454.png" alt="image-20210214105937454" style="zoom:80%;" />

>    The difference between a neural network with one neuron and multiple neurons is that the same calculation is perfomed multiple times. 

The convention followed is: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214135515865.png" alt="image-20210214135515865" style="zoom:80%;" />

We can write operations and activations in the hidden layer for our 2-layer neural network as follows: 

![IMG_5C7A4AA0CA28-1](Neural_Network_and_Deep_Learning.assets/IMG_5C7A4AA0CA28-1.jpeg)

which we can write more compactly as:
$$
Z^{[1]}_1 = w^{[1]T}_1x + b^{[1]}_1; \ a^{[1]}_1 = \sigma(Z^{[1]}_1) \\[15 pt]
Z^{[1]}_2 = w^{[1]T}_2x + b^{[1]}_2; \ a^{[1]}_2 = \sigma(Z^{[1]}_2) \\[15 pt]
Z^{[1]}_3 = w^{[1]T}_3x + b^{[1]}_3; \ a^{[1]}_3 = \sigma(Z^{[1]}_3) \\[15 pt]
Z^{[1]}_4 = w^{[1]T}_4x + b^{[1]}_4; \ a^{[1]}_4 = \sigma(Z^{[1]}_4) \\[15 pt]
$$
Note that $x$ is a column vector with elements: $x = [x_1, x_2, x_3]$. 

So, we see that the two computations are done for each neuron in the hidden layer. The first layer is associated with the square brackets for the weight vector, the intercept vector, and the activation function $\sigma(Z^{{[1]}}_1)$ all corresponding to the first neuron. We then do the same for each of the other neurons. 

Thus the activations for the first hidden layer, $a^{[1]}$ will have 4 rows corresponding to the $Z$ and $a = \sigma(Z)$. In other words, we will write this as,
$$
a^{[1]} = \begin{bmatrix} 
 a^{[1]}_1\\
 a^{[1]}_2\\
 a^{[1]}_3 \\
 a^{[1]}_4 \\
\end{bmatrix}
$$
The convention is a little screwed up here, but the computation of $Z^{[l]}_i$ is done in the following way: 

![IMG_5C7A4AA0CA28-1 2](Neural_Network_and_Deep_Learning.assets/IMG_5C7A4AA0CA28-1%202.jpeg)

The above matrix multiplication results in the activations $a^{[1]}$ which has a dimension of $(4,1)$. When this activations pass through the sigmoid function, we have the same dimension. 

We can write all of the above four equation in terms of: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214140208091.png" alt="image-20210214140208091" style="zoom:80%;" />

Here: 
$$
a^{[0]} = X
$$
This is the input vector of dimensions $(3, 1)$. 

We now move to the second layer. In this case, we will have: 

 <img src="Neural_Network_and_Deep_Learning.assets/image-20210214140652866.png" alt="image-20210214140652866" style="zoom:80%;" />

This second layer can be thought of as a single neuron: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214141004015.png" alt="image-20210214141004015" style="zoom:80%;" />

### Vectorizing Across Multiple Examples

In the previous example, we say how to pass one instance or observation through a neural network. The value $a^{[0]}$ was a vector with three features, $[x_1, x_2, x_3]$ of a given instance. Now we will see how to pass multiple instances. 

In the previous sections, we saw that for a 2-layer neural network, we needed the following set of equations: 
$$
Z^{[1]} = W^{[1]}a^{[0]} + B^{[1]} \\[15pt]
a^{[1]} = \sigma(Z^{[1]}) \\[15pt]

Z^{[2]} = W^{[2]}a^{[1]} + B^{[2]}\\[15pt]
a^{[2]} = \sigma(Z^{[2]})
$$
Now for $m$ training examples, we will repeat the above calculations $m$ times. We will make use of another convention to take into consideration the $m$ training examples. The activations will have the following notation: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214142114214.png" alt="image-20210214142114214" style="zoom:80%;" />

With this we are ready to compute the activations across two layers. Typically, we would use a for loop to run through our $m$ training examples. Here's how we may do it: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214142606660.png" alt="image-20210214142606660" style="zoom:50%;" />

This would be a non-vectorized way of doing it. It would also be slower than the vectorized here. The good thing is that we can easily vectorize the above. Here's how: 

Note that our $a^{[0]}$ consists of a column vector with 3 features for a given instance. So, we can place our instances as columns in a given matrix, the computation of $W^{[1]}x^{(i)}$ is simply a dot product of a row of $W$ and a column of $x^{(i)}$. Now if $x^{(i)}$ were a matrix, the computation will not change but be the same. This is because a dot product of two matrices is simple a scalar multiplication of a row of one matrix and the column of a second matrix at a given time. 

Thus we can can write the vectorized form of the above equation as: 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214143632462.png" alt="image-20210214143632462" style="zoom:80%;" /> 

Where $X$ has the dimension of $(n, m)$, where there are $n$ features and $m$ instances. In other words, we have:

![IMG_D233E8FEC9A5-1](Neural_Network_and_Deep_Learning.assets/IMG_D233E8FEC9A5-1.jpeg)

When we move horizontally in $X$, we went through various training examples while when we move top to bottom, we go through features of a given training example.

When we move horizontally in $A$ and $Z$, we go through training examples while going vertically, top to bottom we go through different nodes or the neurons for a given training example. 

So, the first element in $A$ corresponds to the activation of the first hidden unit that has the first training instance, the one below will correspond to the second hidden unit that has the first training instance and so on.

### Activation Function

The sigmoid function that we have used so far is called an **activation function**. However, an activation does not have to be a sigmoid function. For example, one example of activation function is **tanh(z)**. This is given bym 
$$
tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$
The form of this activation function is similar to sigmoid function with the exception that it ranges from -1 to 1. 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214154738818.png" alt="image-20210214154738818" style="zoom:80%;" />

The tanh function can be thought of as a sigmoid function that is shifted by 0.5. In general the tanh function works really well for the hidden layers in a neural network. 

In a given neural network we can have a combination of activation functions. For example, for one layer can have a ReLU function while the other layer can have a tanh function. 

Both activation functions discussed so far have a shortcoming. Because they approach their limits, the Gradient Descent takes a very long time to converge. Therefore, the ReLU function is better suited as an activation function. The ReLU function is either linear or equal to zero. This helps the GD converge very quickly as the slope of ReLU is different from zero. 

A version of ReLU is a **leaky ReLU**, which is not often used. 

<img src="Neural_Network_and_Deep_Learning.assets/image-20210214155824212.png" alt="image-20210214155824212" style="zoom:80%;" />

We choose the activation function based on what the output should be. If the output needs to be between 0 and 1, then choose a sigmoid function. But for all hidden layers, choose the ReLU function. 

>   Use the sigmoid function for the output layer and use the tanh and ReLU for hidden layers. This is when you wish the output to be between 0 and 1. 

To summarize: 

| Activation Function | Pros                                   | Cons                     |
| ------------------- | -------------------------------------- | ------------------------ |
| Sigmoid             | Helpful for binary categories          | GD does not work well    |
| Tanh                | Much better tahn Sigmoid               | GD does not work well    |
| ReLU                | Better for GD                          | Zero for negative values |
| Leaky ReLU          | Better for GD; Non-zero for -ve values | None                     |



### Why we need non-linear Activation Function? 

Let's see why we use a non-linear function by using a linear function instead. With a 2-layer NN, we have the following equations that we have seen earlier: 
$$
Z^{[1]} = W^{[1]}a^{[0]} + b^{[1]} \\[15pt]
a^{[1]} = g^{[1]}(Z^{[1]}) \\[15pt]
Z^{[1]} = W^{[1]}a^{[1]} + b^{[1]} \\[15pt]
a^{[2]} = g^{[2]}(Z^{[2]}) \\[15pt]
$$
Here we assume that our activation function, $g^{[l]}$ is different in each layer. Now suppose that we use an identify activation function, which is simply $g(z) = z$. Such a identity function is a linear function. Let's see what happens to the above equations: 

![IMG_39EF11116931-1](Neural_Network_and_Deep_Learning.assets/IMG_39EF11116931-1.jpeg)

Now we replace $a^{[1]}$ in equation three to get: 

<img src="Neural_Network_and_Deep_Learning.assets/IMG_39EF11116931-1%202.jpeg" alt="IMG_39EF11116931-1 2" style="zoom:80%;" />

Now calling $W^{'} = (W^{[2]}W^{[1]})$ and $b^{'} = W^{[2]}b^{[1]} + b^{[1]}$, we can write the above equation as:

<img src="Neural_Network_and_Deep_Learning.assets/IMG_39EF11116931-1%203.jpeg" alt="IMG_39EF11116931-1 3" style="zoom:80%;" />

We see that this is simply a linear function of input features and biases. This looks very similar to the logisitc function we worked out before. In other words, having a hidden layer does not play any part when the activation function is linear. Thus not, using a hidden layer is exactly the same when using it. We can easily extrapolate this to multiple hidden layers. When we have these and we use a linear function, it is as good as not using any hidden layers. 

The use of linear function boils down to simply using a linear regression to solve the problem. Then why even use a neural network? So, unless you use a non-linear function, NN is not very useful. However, note that when you are doing a linear regression, you will need to use a linear activation function, right before the output. But the hidden layers should be non-linear.

### Derivatives of Activation Functions

The derivatives of the activation functions are listed in the table below:

| Activation Function | Mathematical Form             | Derivative                        |
| ------------------- | ----------------------------- | --------------------------------- |
| Sigmoid             | $g(z) = \frac{1}{1 + e^{-z}}$ | $g(z)(1-g(z))$                    |
| Tahnh               | $g(z) = tanh(z)$              | $(1-g(z)^2)$                      |
| ReLU                | $g(z) = max(0,z)$             | 0 if $z < 0$ else 1 if $z \geq 0$ |
| Leaky ReLU          | $g(z) = max(0.01z, z)$        | 0.01 if $z < 0$ else 1 if $z > 0$ |

### Gradient Descent for Neural Networks

The following equations for forward propagation and backward propagation work for any neural network. For simplicity, we will use a NN with only two layers: 

#### Forward Propagation Equations

$$
Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]} \\[15pt]
A^{[1]} = g^{[1]}(Z^{[1]}) \\[15pt]
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\[15pt]
A^{[2]} = g^{[2]}(Z^{[2]}) \\[15pt]
$$

#### Backward Propagation Equations

$$
dz^{[2]} = A^{[2]} - Y \\[15pt]
dw^{[2]} = \frac{1}{m}dz^{[2]}A^{[1]T} \\[15pt]
db^{[2]} = \frac{1}{m}\sum^{m}_{i=1}dz^{[2]} \\[15pt]
dz^{[1]} = W^{[2]T}dz^{[2]} \times dg^{[1]}(z^{[1]}) \\[15pt]
dw^{[1]} = \frac{1}{m}dz^{[1]}X^T \\[15pt]
db^{[1]} = \frac{1}{m}\sum^{m}_{i=1}dz^{[1]}
$$

Note the similarity between $db^{[i]}$ and $dw^{[i]}$. 



### Week 3: Quiz

1.  Which of the following is true? 

    1.  $a^{[2]}_4$ is the activation output of the 2nd layer for the 4th training example
    2.  $a^{[2]}_4$ is the activation output by the 4th neuron of the second layer
    3.  $a^{[2](12)}$ denotes the activation vector of the 2nd layer for the 12th training example
    4.  X is a matrix in which each row is one training example
    5.  X is a matrix in which each column is one training example
    6.  $a^{[2]}$ denotes the activation vector of the 2nd layer
    7.  $a^{[2](12)}$ denotes activation vector of the 12th layer on the 2nd training example

2.  The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?

3.  Which of these is a correct vectorized implementation of forward propgation for layer $l$, where $1 \leq l \leq L$. 

4.  You are building a binary classifier for recognizing cucumbers (y=1) vs watermelons (y=0). Which one of these activation functions would you recommend using for the output layer?

    1.  ReLU
    2.  Leaky ReLU
    3.  Sigmoid
    4.  tanh

5.  Consider the following code. What is the shape of B?

    ```python
    A = np.random.randn(4,3)
    B = np.sum(A, axis = 1, keepdims = True)
    ```

6.  Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true?

    1.  Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. 
    2.  Each neuron in the first hidden layer will perform the same computation in the first iteration. But after one iteration of gradient descent they will learn to compute different things because we have “broken symmetry”.
    3.  Each neuron in the first hidden layer will compute the same thing, but neurons in different layers will compute different things, thus we have accomplished “symmetry breaking” as described in lecture. 
    4.  The first hidden layer’s neurons will perform different computations from each other even in the first iteration; their parameters will thus keep evolving in their own way.

7.  Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

8.  You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen? 

    1.  It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small. 
    2.  This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α*α* to be very small to prevent divergence; this will slow down learning. 
    3.  This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow. 
    4.  This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.

9.  Consider the following NN: 
    ![image-20210310172529163](Neural_Network_and_Deep_Learning.assets/image-20210310172529163.png)
    What would be the dimensions of $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$?

10.  In the same network as the previous question, what are the dimensions of $Z^{[1]}, a^{[1]}$?

### Quiz Answers

1.  Choices: 2, 3, 5, 6 are true
2.  True.Yes. As seen in lecture the output of the tanh is between -1 and 1, it thus centers the data which makes the learning simpler for the next layer.
3.  That would be $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$; $A^{[l]} = g^{[l]}(Z^{[l]})$. 
4.  As the output should be between 0 and 1, we will use the sigmoid function
5.  We sum the rows of A so that will create a column vector with 4 elements. As we use `keepdims=True`, the dimensions will be (4,1)
6.  Answer is the first because the computation will be exactly the same.
7.  False. Yes, Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector. 
8.  The answer is 3. Yes. tanh becomes flat for large values, this leads its gradient to be close to zero. This slows down the optimization algorithm.
9.  $W^{[1]}: (4,2); b^{[1]}:(4,1)$ while $W^{[2]}:(1,4);  b^{[2]}:(1,1)$
10.  Both would be $(4,m)$



