# Machine Learning with Python 
This repository contains Machine Learning Projects in Python programming language. 
All the projects are done on Jupyter Notebooks.

## Libraries Required 
The following libraries are required to successfully implement the projects.
  - Python 3.6+
  - NumPy (for Linear Algebra)
  - Pandas (for Data Preprocesssing)
  - Scikit-learn (for ML models)
  - Matplotlib (for Data Visualization)
  - Seaborn (for statistical data visualization)

The projects are divided into various categories listed below -

## Supervised Learning 
  - [**Linear Regression**]()
     - [Linear Regression Single Variables.](https://github.com/suubh/Machine-Learning-in-Python/blob/master/Linear%20Regression/LinearRegressionSingle%20Variables.ipynb) : A Simple Linear Regression Model to model the linear relationship between Population and Profit for plot sales.
     - [Linear Regression Multiple Variables.](https://github.com/suubh/Machine-Learning-in-Python/blob/master/Linear%20Regression/LinearRegressionMultipleVariables.ipynb) : In this project, I build a Linear Regression Model for multiple variables for predicting the House price based on acres and number of rooms.
   
  - [**Logistic Regression**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/Logistic%20Regression/Logistic/Untitled.ipynb) : In this project, I train a binary Logistic Regression classifier to predict whether a student will get selected on the basis of mid semester and end semester marks.
  
  - [**Support Vector Machine**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/SVM/Untitled.ipynb) : In this project, I build a Support Vector Machines classifier for predicting Social Network Ads . It predicts whether a user with age and estimated salary will buy the product after watching the ads or not. It uses the Radial Basic Function Kernal of SVM. 
  
  - [**K Nearest Neighbours**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/K-NN/Untitled.ipynb) : K Nearest Neighbours or KNN is the simplest of all machine learning algorithms. In this project, I build a kNN classifier on the Iris Species Dataset which predict the three species of Iris with four features *sepal_length, sepal_width, petal_length* and *petal_width*.
  
  - [**Naive Bayes**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/TextClassification/Textclassification.ipynb) : In this project, I build a Naïve Bayes Classifier to classify the different class of a message from sklearn dataset called [*fetch_20newsgroups*](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html).
  
  - [**Decision Tree Classification**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/Decision%20Tree/Untitled.ipynb) :  In this project, I used the Iris Dataset and tried a Decision Tree Classifier which give an accuracy of 96.7% which is less than KNN.
  
  - [**Random Forest Classification**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/RandomForest/RandomForest.ipynb) : In this project I used Random Forest Classifier and Random Forest Regressor on the Social Network Ads dataset. 

## Unsupervised Learning 
  - [**K Means Clustering**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/K-means/creditcard.ipynb) : K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences.It is one of the most detailed projects, In this project, I implement K-Means Clustering  on Credit Card Dataset to cluster different credit card users based on the features.I scaled the data using *StandardScaler* because normalizing(scale in range 0 to 1) will improves the convergence.I also implemented the [*Elbow Method*](https://en.wikipedia.org/wiki/Elbow_method_(clustering)) to search for the best numbers of clusters.For visualizing the dataset I used [*PCA(Principal Component Analysis)*](https://en.wikipedia.org/wiki/Principal_component_analysis) for dimensionality reduction as the dataset features were large in number.In the end I used [*Silhouette Score*]() which is used to calculate the performance of clustering . It ranges from -1 to 1 and I got a score of 0.203.

## NLP( Natural Language Processing )
  - [**Text Analytics**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/TextAnalytics/textAnalytics.ipynb) : It is a project for Introduction to Text Analytics in NLP.I performed the important steps -
    - ***Tokenization***
    - ***Removal of Special Characters***
    - ***Lower Case***
    - ***Removing StopWords***
    - ***Stemming*** 
    - ***Count Vectorizer***  ( which generally performs all the steps mentioned above except Stemming)
    - ***DTM (Document Term Matrix)***
    - ***TF-IDF (Text Frequency Inverse Document Frequency)***
    
  - [**Sentiment Analysis**](https://github.com/suubh/Machine-Learning-in-Python/tree/master/Sentiment%20Analysis) : I applied Sentiment analysis in MovieReview (Dataset from nltk library) and RestaurentReview Datasets to predict the positive and negative review . I used Naive Bayes Classifier (78.8%) and Logistic Regression (84.3%) to build the models and for prediction. 
 
## Data Cleaning and Preprocessing
  - [**Data Preprocessing**](https://github.com/suubh/Machine-Learning-in-Python/blob/master/Data%20Preprocessing/Untitled.ipynb) : I perform various data preprocessin and cleaning methods which are mentioned below -
    - ***Label Encoding*** : It converts each category into a unique numeric value ranging from 0 to n(size of dataset).
    - ***Ordinal Encoding*** : Categories to ordered numerical values.
    - ***One Hot Encoding*** : It creates a dummy variable with value 0 to n(unique value count in the column) for each category value.Extra columns are created.

## Some Comparisons on Datasets
| **Social Network Ads**     | **Accuracy**|
| ----------- | ----------- |
| Support Vector Machine     | 90.83%     |
| Random Forest Classifier   | 90.0%      |
| Random Forest Regressor    | 61.8%      |

| **Iris Dataset**     | **Accuracy** |
| ----------- | ----------- |
| KNN     | 98.3%     |
| Decision Tree   | 96.7%      |





