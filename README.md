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

## evaluate model
* https://scikit-learn.org/stable/modules/model_evaluation.html
### metrics
* recall
* precision
* log loss - measure of how likely model thinks the predictions are correct. Model thinks that there 5% probability of buyer to buy shirt.
* mean absolute error
* hinge loss
* r^2
* quantile loss
* f1 score
* kl divergence
* accuracy - fraction of predictions a model gets rigth

### methods to evaluate regression model
* Mean absolute error (MAE): This is measured by taking the average of the absolute difference between the actual values and the predictions. Ideally, this difference is minimal.
* Root mean square error (RMSE): This is similar MAE, but takes a slightly modified approach so values with large error receive a higher penalty. RMSE takes the square root of the average squared difference between the prediction and the actual value.
* Coefficient of determination or R-squared (R^2): This measures how well-observed outcomes are actually predicted by the model, based on the proportion of total variation of outcomes.

## model inference
* even after this step you can change algos, data, requirements

# examples 
* supervised - predict prices
* unsupervised - isolate micro-genres of books
* neural network - raw images analysis, try to found chemical spills
* https://machinelearningmastery.com/ridge-regression-with-python/

### unsupervised evaluation metrics:
* V - measure
* Silhouette coefficient - how well the data was clusterred
* Rand index
* Mutual information
* Contingency matrix
* Completeness
* Homogenity
* Fowlkes-Mallows
* Calinski-Harabasz index
* Pair confusion matrix
* Davies-Boulding index

### unsupervised learning - terminology
* bag of words - is a technique to extract features from the text. counts how many times a word appears in the doc.
* data vectorization - transform of non-numeric data into a numerical format so that it can be used by the machine
* silhouette coefficient - a score from -1 to 1 that describes the clusters found during modelling. 0 - overlapping clusters, less then zero - points are assigned to wrong clusters, more then zero - success
* stop words - a list of words removed during building datasat, like 'a', 'the'

### reading on unsupervised
* https://machinelearningmastery.co/
* https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/

### readings on deep learning/neural networks
* https://aws.amazon.com/blogs/machine-learning/protecting-people-through-virtual-boundaries-computer-vision/

## intro into ml glossary
* Discrete: A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).
* Machine learning, or ML, is a modern software development technique that enables computers to solve problems by using examples of real-world data. 
  

# machine learning in aws
### aws services pretrained
* health ai(amazon prescribed medical, speech to text)
* industrial ai(monitron, failure of machines prediction)
* anomaly detection(lookout for metrics)
* chatbot(amazon lex)
* personalization(amazon personalize, recomendation system)
* forecasting(amazon forecast)
* fraud(amazon fraud detector)
* code development(codeguru & devops guru)\
* vision(amazon recognition)
* speech(amazon polly)
* text(amazon textract)
* contact centers(amazon lens, analyze users speech with contact center)
* search(amazon kendra, search in unstructured data)

### other products
* amazon sagemaker studio - build train and deploy ml models
* amazon sagemaker distributed training - partitions model and training data for faster learning
* amazon sagemake clarify - detects bias across the ML workflow and explaining model behaviour

### readings on aws ml services
* https://aws.amazon.com/machine-learning/ai-services/?utm_source=Udacity&utm_medium=Webpage&utm_campaign=Udacity%20AWS%20ML%20Foundations%20Course

# Reinforcement learning 
* Agent: The piece of software you are training is called an agent. It makes decisions in an environment to reach a goal.
* Environment: The environment is the surrounding area with which the agent interacts.
* Reward: Feedback is given to an agent for each action it takes in a given state. This feedback is a numerical reward.
* Action: For every state, an agent needs to take an action toward achieving its goal.
* Action space: The set of all valid actions or choices, available to an agent as it interacts with the environment.
* Hyperparameteres: Are variables that control the performance of your agent during training. E.g. learning rate. 
* The reward function's purpose is to encourage the agent to reach its goal. 
* reward graph: plot of rewards from each episode
* 
### strategies of reinforcement learning in aws deep racer
* soft actor critic(encourage exploration), data-efficient, in aws deep racer works only in discrete action spaces
* proximal policy optimization(very stable), data hungry, in aws deep racer works in discrete and continuous action spaces

### reward graph(aws deepracer)
* Average reward. This graph represents the average reward the agent earns during a training iteration. The average is calculated by averaging the reward earned across all episodes in the training iteration. 
* Average percentage completion (training). The training graph represents the average percentage of the track completed by the agent in all training episodes in the current training. It shows the performance of the vehicle while experience is being gathered.
* Average percentage completion (evaluation).  While the model is being updated, the performance of the existing model is evaluated. The evaluation graph line is the average percentage of the track completed by the agent in all episodes run during the evaluation period.
* Best model line. This line allows you to see which of your model iterations had the highest average progress during the evaluation. The checkpoint for this iteration will be stored. A checkpoint is a snapshot of a model that is captured after each training (policy-updating) iteration.
* Reward primary y-axis. This shows the reward earned during a training iteration. To read the exact value of a reward, hover your mouse over the data point on the graph.
* Percentage track completion secondary y-axis. This shows you the percentage of the track the agent completed during a training iteration.
* Iteration x-axis. This shows the number of iterations completed during your training job.

### Adjust Hyperparameteres
* A lower learning rate makes learning take longer but can help increase the quality of your model.

# Generative AI
* are mostly unsupervised models
* non generative techniquue: discriminative(cat vs dog, aka cat-dog detector)
* generative technique: creates new data from the training data set(create new image of cat)
* A generative model aims to answer the question,"Have I seen data like this before?"
### types of generative ai models:
* generative adversarial networks(GANs) : uses two neural networks(generator network produces new data, discriminator measures similarity) to generate new content
* Autoregressive models(AR-CNN): used to stydy systems that evolve over the time, aka prediction.
* Transformer-based models :  used to study data that is sequentional in nature(pixels in picture, words in text).
### AWS DeepComposer
* it uses generative ai techniques
* It consists of a USB keyboard that connects to your computer to input melody and the AWS DeepComposer console, which includes AWS DeepComposer Music studio to generate music, learning capsules to dive deep into generative AI models
### GANs(in scope of DeepComposer)
* A GAN is a type of generative machine learning model which pits two neural networks against each other to generate new content: a generator and a discriminator.
* A generator is a neural network that learns to create new data resembling the source data on which it was trained.
* A discriminator is another neural network trained to differentiate between real and synthetic data. Critiques how realistic the data is.
* The generator and the discriminator are trained in alternating cycles. The generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.
* The generator takes in a batch of melody as the input and generates a a music as the output by adding accompaniments to each of the input music tracks.
* The discriminator then takes these generated music tracks and predicts how far they deviate from the real data present in the training dataset. This deviation is called the generator loss. This feedback from the discriminator is used by the generator to incrementally get better at creating realistic output.
* As the generator gets better at creating music accompaniments, it begins fooling the discriminator. So, the discriminator needs to be retrained as well. The discriminator measures the discriminator loss to evaluate how well it is differentiating between real and fake data.

 ### AR-CNN(in scope of DeepComposer)
* Piano roll: A two-dimensional piano roll matrix that represents input tracks. Time is on the horizontal axis and pitch is on the vertical axis.
* Edit event: When a note is either added or removed from your input track during inference.
### Glossary from lesson 3
 *   Action: For every state, an agent needs to take an action toward achieving its goal.
 *   Agent: The piece of software you are training is called an agent. It makes decisions in an environment to reach a goal.
 *   Discriminator: A neural network trained to differentiate between real and synthetic data.
 *   Discriminator loss: Evaluates how well the discriminator differentiates between real and fake data.
 *   Edit event: When a note is either added or removed from your input track during inference.
 *   Environment: The environment is the surrounding area within which the agent interacts.
 *   Exploration versus exploitation: An agent should exploit known information from previous experiences to achieve higher cumulative rewards, but it also needs to explore to gain new experiences that can be used in choosing the best actions in the future.
 *  Generator: A neural network that learns to create new data resembling the source data on which it was trained.
 *   Generator loss: Measures how far the output data deviates from the real data present in the training dataset.
 *   Hidden layer: A layer that occurs between the output and input layers. Hidden layers are tailored to a specific task.
 *   Input layer: The first layer in a neural network. This layer receives all data that passes through the neural network.
 *   Output layer: The last layer in a neural network. This layer is where the predictions are generated based on the information captured in the hidden layers.
 *   Piano roll: A two-dimensional piano roll matrix that represents input tracks. Time is on the horizontal axis and pitch is on the vertical axis.
 *   Reward: Feedback is given to an agent for each action it takes in a given state. This feedback is a numerical reward.

# Software engineering practices
## Clean and Modular Code
* Production code: Software running on production servers to handle live users and data of the intended audience.
* Production-quality code, which describes code that meets expectations for production in reliability, efficiency, and other aspects. Ideally, all code in production meets these expectations, but this is not always the case.
* Clean code: Code that is readable, simple, and concise. Clean production-quality code is crucial for collaboration and maintainability in software development.
* Modular code: Code that is logically broken up into functions and modules. Modular production-quality code that makes your code more organized, efficient, and reusable.
* Module: A file. Modules allow code to be reused by encapsulating them into files that can be imported into other files.

## Refactoring code

*  Refactoring: Restructuring your code to improve its internal structure without changing its external functionality. This gives you a chance to clean and modularize your program after you've got it working.
*  Since it isn't easy to write your best code while you're still trying to just get it working, allocating time to do this is essential to producing high-quality code. Despite the initial time and effort required, this really pays off by speeding up your development time in the long run.
*  You become a much stronger programmer when you're constantly looking to improve your code. The more you refactor, the easier it will be to structure and write good code the first time.

## Writing clean code: Meaningful names
* Be descriptive and imply type: For booleans, you can prefix with is_ or has_ to make it clear it is a condition. You can also use parts of speech to imply types, like using verbs for functions and nouns for variables.
* Be consistent but clearly differentiate: age_list and age is easier to differentiate than ages and age.
* Avoid abbreviations and single letters: You can determine when to make these exceptions based on the audience for your code. If you work with other data scientists, certain variables may be common knowledge. While if you work with full stack engineers, it might be necessary to provide more descriptive names in these cases as well. (Exceptions include counters and common math variables.)
* Long names aren't the same as descriptive names: You should be descriptive, but only with relevant information. For example, good function names describe what they do well without including details about implementation or highly specific uses. 

## Writing clean code: Nice whitespace
* Organize your code with consistent indentation: the standard is to use four spaces for each indent. You can make this a default in your text editor.
* Separate sections with blank lines to keep your code well organized and readable.
* Try to limit your lines to around 79 characters, which is the guideline given in the PEP 8 style guide. In many good text editors, there is a setting to display a subtle line that indicates where the 79 character limit is. 
