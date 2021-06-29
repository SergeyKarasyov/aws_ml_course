# aws_ml_course

## what is ML
* ML is a subset of AI, it is about automatic learning and improvement from experience without any specific coding added with the time. It is how computers learns from data to discover patterns and make predictions.
* there are several subsets of ML
* Supervised learning is about learning a pattern from a dataset where each entry has a label. it may be image classification
* Unsupervised learning is without labels, it's about finding patterns in the data.
* Reinforcement learning. If system acts wrong - give penalty, if it works ok - reward it. Algorithms wants to maximize the reward.


## components of the ml
*  ml model, is a generic program or block of code, made specific by the data used to train it. It is used to solve different problems. It maybe represented by a linear regression function.
*  model training algorithm. thinks over what should be the goal, changes model to look more similar to the goal, repeats until when model meats goals/requirements.
*  model inference algorithm. inspection of result or usage of the model.

## major steps in ML
* define the problem
* build the dataset
* train the model
* evaluate the model
* use the model

## defining the problem
* should be specific (does adding $50 to price increase the amount of sold items)
* supervised learning: classification, regression(predicting a continuous quantity output for an example), predictive(a mapping function from inputs to outputs).
* a categorical label has a discrete set of possible values. In ml when you wnat to identify the type of flower based on a picture your would train your model usuing images that have been labeled with the categories of flower.
* a continuous(regression) label does not have a discrete set of possible values, which often means you are working with numerical data. You can predict amount of products being sold, price etc. Potentially may have an infinite amount of possible results.
* unsupervised learning: clustering, neural networks.
* clustering is a task of entries split in groups 
* categorization is answering a question is a cat or is not a cat
* label is a data that already has a solution 
* kinds of problems possibly be solved by ML: Is this email spam or not spam?, Is this product a book, movie, or clothing?, For this product, how many units will sell?


## build a dataset
* %80 time of ml engineers is spent on preparing the data
* working with data includes: data collection, data inspection, summary statics, data visualization
* with statistical tools like mean, inner-quartile range, standard deviation you could know the scope, scale and shape of the dataset
* impute is a commion term referring to differnt statistical tools which can be used to calculate missing values
* outliers are data points that are significantly different from others in the scope
* there is industry solutions. example is sklearn lib https://sklearn.org/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py

## model training
* split datast to: mostly to a training set and maybe 20% to a test set(evaluate your model)
* iteratively update model parameters to minimize some loss function
* model parameters are configuration that changes model behaviour. Sometimes are called as weights or biases . If to think about function as a linear, the parameters may be orientation and location of the line.
* a loss function is a measurement how close the model is to its goal. like the average distance between your model's predicted result and actual number.
* hyperparameters are not changed during training, they are to influence the speed or correctness of the model. E.g. number of features to compare or number of clusters
* Practitioners use a process called model selection to determine which model or models to use
* linear modules are popular, good for tasks of classiffication
* tree based models, is like a tree of if else branches, they split the world into regions, the regions borders are trained. XGBoost is the commonly used implementation.

### neural networks, deep learning use case
* deep learning models, is about concept of brain work. Neural network is composed of neurons connected by weights(math representation of how much data can flow from one neuron to another). Training is about finding best weights.
* FFNN(feed forward neural network) the most straightforward solution. Structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the layer.
* CNN (convolutional neural network) represents nested filters over grid-organized data. they are popular for processing images.
* RNN/LSTM(Recurrent neural network, long short term memory) model types are structured to effectively represent 'for loops' in traditional computing collecting state while iterating over some object. They can be used for processing sequences of data.
* Transformer. a modern replacement for rnn/lstm, enables training over larger datasets involving sequences of data

### machine learning and python
* statistical models can be done via scikit-learn
* for deep learning models, mxnet, tensorflow, pytorch