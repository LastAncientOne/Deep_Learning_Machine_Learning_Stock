<img src="DL_Title.PNG">

# Deep Learning and Machine Learning for Stock Predictions  
Description: This is for learning, studying, researching, and analyzing stock in deep learning (DL) and machine learning (ML). Predicting Stock with Machine Learning method or Deep Learning method with different types of algorithm. Experimenting in stock data to see how it works and why it works or why it does not work that way. Using different types of stock strategies in machine learning or deep learning. Using Technical Analysis or Fundamental Analysis in machine learning or deep learning to predict the future stock price. In addition, to predict stock in long terms or short terms.  

Machine learning is a subset of artificial intelligence involved with the creating of algorithms that can change itself without human intervention to produce an output by feeding itself through structured data. On the other hand, deep learning is a subset of machine learning where algorithms created, but the function are like machine learning and many of the different type of algorithms give a different interpretation of the data. The network of algorithms called artificial neural networks and is similar to neural connections that exist in the human brain.  

<h3 align="left">Languages and Tools:</h3>
<p align="left"> </a> <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.r-project.org/" target="_blank"> <img src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/r/r.png" alt="R" width="40" height="40"/> </a> <a href="https://www.mathworks.com/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png" alt="Matlab" width="40" height="40"/> </a> <a href="https://www.microsoft.com/en-us/microsoft-365/excel" target="_blank"> <img src="https://zapier-images.imgix.net/storage/services/296388d714e0dcd78105c9b165ca751e.png?auto=format&ixlib=react-9.0.2&ar=undefined&fit=crop&h=105&w=105&q=50&dpr=1g" alt="Excel" width="40" height="40"/>  </a> <a href="https://www.automateexcel.com/vba-code-examples/" target="_blank"> <img src="https://nakedsecurity.sophos.com/wp-content/uploads/sites/2/2015/09/vba-957.jpg?w=780&h=408&crop=1" alt="VBA" width="40" height="40"/> </a> <a href="https://powerbi.microsoft.com/en-us/" target="_blank"> <img src="https://www.k2e.com/wp-content/uploads/2018/12/Power-BI-Logo.png" alt="Power BI" width="40" height="40"/> </a> <a href="https://www.tableau.com/" target="_blank"> <img src="https://pbs.twimg.com/profile_images/1268207088683020288/d9agkn4h.jpg" alt="tableau" width="40" height="40"/> </a> <a href="https://nteract.io/" target="_blank"> <img src="https://avatars.githubusercontent.com/u/12401040?s=200&v=4" alt="Nteract" width="40" height="40"/> </a> <a href="https://anaconda.org/" target="_blank"> <img src="https://www.clipartkey.com/mpngs/m/227-2271689_transparent-anaconda-logo-png.png" alt="Anaconda" width="40" height="40"/> </a> <a href="https://www.spyder-ide.org/" target="_blank"> <img src="https://www.pinclipart.com/picdir/middle/180-1807410_spyder-icon-clipart.png" alt="Spyder" width="40" height="40"/> </a> <a href="https://jupyter.org/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Jupyter_logo.svg" alt="Jupyter Notebook" width="40" height="40"/> </a> <a href="https://notepad-plus-plus.org/" target="_blank"> <img src="https://logos-download.com/wp-content/uploads/2019/07/Notepad_Logo.png" alt="Notepad++" width="40" height="40"/> </a> </p>

### Three main types of data: Categorical, Discrete, and Continuous variables
  1. Categorical variable(Qualitative): Label data or distinct groups.    
    Example: location, gender, material type, payment, highest level of education  
  2. Discrete variable (Class Data): Numerica variables but the data is countable number of values between any two values.  
    Example: customer complaints or number of flaws or defects, Children per Household, age (number of years)  
  3. Continuous variable (Quantitative): Numeric variables that have an infinite number of values between any two values.
    Example: length of a part or the date and time a payment is received, running distance, age (infinitly accurate and use an infinite number of decimal places)  

### Data Use  
  1. For 'Quantitative data' is used with all three centre measures (mean, median and mode) and all spread measures.  
  2. For 'Class data' is used with median and mode.  
  3. For 'Qualitative data' is for only with mode.  

### Two types of problems: 
  1. Classification (predict label)  
  2. Regression (predict values)  

### Bias-Variance Tradeoff  
#### Bias  
- Bias is the difference between our actual and predicted values.  
- Bias is the simple assumptions that our model makes about our data to be able to predict new data.  
- Assumptions made by a model to make a function easier to learn.   
#### Variance  
- Variance is opposite of bias.  
- Variance is variability of model prediction for a given data point or a value that tells us the spread of our data.  
- If you train your data on training data and obtain a very low error, upon changing the data and then training the same.   

### Overfitting, Underfitting, and the bias-variance tradeoff  
Overfitted is when the model memorizes the noise and fits too closely to the training set. Good fit is a model that learns the training dataset and genernalizes well with the old out dataset. Underfitting is when it cannot establish the dominant trend within the data; as a result, in training errors and poor performance of the model. 

#### Overfitting:   
Overfitting model is a good model with the training data that fit or at lease with near each observation; however, the model mist the point and random noise is capture inside the model. The model have low training error and high CV error, low in-sample error and high out-of-sample error, and high variance.  
  1. High Train Accuracy   
  2. Low Test Accuracy
#### Avoiding Overfitting:  
  1. Early stopping - stop the training before the model starts learning the noise within the model.   
  2. Training with more data - adding more data will increase the accuracy of the modelor can help algorithms detect the signal better.     
  3. Data augmentation - add clean and relevant data into training data.  
  4. Feature selection - Use important features within the data. Remove features. 
  5. Regularization - reduce features by using regularization methods such as L1 regularization, Lasso regularization, and dropout.  
  6. Ensemble methods - combine predictions from multiple separate models such as bagging and boosting.       
  7. Increase training data.  
#### Good fit:  
Good fit:   
  1. High Train Accuracy   
  2. High Test Accuracy   
#### Underfitting:  
Underfitting model is not perfect, so it does not capture the underlying logic of the data. Therefore, the model does not have strong predictive power with low accuracy. The model have large training set error, large in-sample error, and high bias.  
  1. Low Train Accuracy  
  2. Low Test Accuracy   
#### Avoiding Underfitting:  
  1. Decrease regularization - reduce the variance with a model by applying a penalty to the input parameters with the larger coefficients such as L1 regularization, Lasso regularization, dropout, etc.   
  2. Increase the duration of training - extending the duration of training because stopping the training early will cause underfit model.  
  3. Feature selection - not enough predictive features present, then adding more features or features with greater importance would improve the model.  
  4. Increase the number of features - performing feature engineering  
  5. Remove noise from the data    


## Python Reviews
Step 1 through step 8 is a reviews in python.  
After step 8, everything you need to know that is relate to data analysis, data engineering, data science, machine learning, and deep learning.   

## List of Machine Learning Algorithms for Stock Trading  
### Most Common Regression Algorithms  
1. Simple Linear Regression Model  
2. Logistic Regression  
3. Lasso Regression    
4. Support Vector Machines  
5. Polynomial Regression  
6. Stepwise Regression  
7. Ridge Regression  
8. Multivariate Regression Algorithm    
9. Multiple Regression Algorithm  
10. K Means Clustering Algorithm  
11. Naïve Bayes Classifier Algorithm  
12. Random Forests  
13. Decision Trees  
14. Nearest Neighbours   
15. Lasso Regression  
16. ElasticNet Regression  
17. Reinforcement Learning  
18. Artificial Intelligence    
19. MultiModal Network  
20. Biologic Intelligence  

### Different Types of Machine Learning Algorithms and Models  
Algorithms is a process and set of instructions to solve a class of problems. In addition, algorithms perform a computation such as calculations, data processing, automated reasoning, and other tasks. A machine learning algorithms is a method that provides the systems to have the ability to automatically learn and improve from experience without being formulated.   

# Prerequistes  
Python 3.5+  
Jupyter Notebook Python 3  

## :black_square_button: Add more of algorithms and different types of algorithms   

## Authors  
### * Tin Hang

## Disclaimer  
&#x1F53B; Do not use this code for investing or trading in the stock market. However, if you are interest in the stock market, you should read :books: books that relate to stock market, investment, or finance. On the other hand, if you into quant or machine learning, read books about &#x1F4D8; machine trading, algorithmic trading, and quantitative trading. You should read &#x1F4D7; about Machine Learning and Deep Learning to understand the concept, theory, and the mathematics. On the other hand, you should read academic paper and do research online about machine learning and deep learning on :computer:  

