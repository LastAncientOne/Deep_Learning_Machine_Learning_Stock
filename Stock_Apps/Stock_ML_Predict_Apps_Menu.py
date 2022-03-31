# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:44:59 2019

@author: Tin
"""

import numpy as np
import matplotlib.pyplot as plt

import datetime

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.svm import SVR 

import warnings
warnings.filterwarnings("ignore")

# yahoo finance used to fetch data 
import yfinance as yf
yf.pdr_override()

options = " Stock Linear Regression Prediction, Stock Logistic Regression Prediction, Support Vector Regression, Exit".split(",")

# Input Start Date
def start_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    start = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    start = start.strftime('%Y-%m-%d')
    return start

# Input End Date
def end_date():
    date_entry = input('Enter a ending date in MM/DD/YYYY format: ')
    end = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    end = end.strftime('%Y-%m-%d')
    return end

# Input Symbols
def input_symbol():
    symbol = input("Enter symbol: ").upper()
    return symbol

# Logistic Regression
def stock_logistic_regression():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
 
    df = df.drop(['Date'], axis=1)
    X = df.loc[:, df.columns != 'Adj Close']
    y = np.where (df['Adj Close'].shift(-1) > df['Adj Close'],1,-1)

    split = int(0.7*len(df))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = LogisticRegression()
    model = model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print(metrics.confusion_matrix(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))
    print(model.score(X_test,y_test))
    cross_val = cross_validate(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print(cross_val)
    print(cross_val.mean())
    ans = ['1', '2'] 
    user_input=input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)   
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input=input("Command: ")
    if user_input=="1":
        menu()
    elif user_input=="2":
        exit()    
        

# Linear Regression
def stock_linear_regression():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    n = len(df.index)
    X = np.array(df['Open']).reshape(n,-1)
    Y = np.array(df['Adj Close']).reshape(n,-1)
    lr = LinearRegression()
    lr.fit(X, Y)
    lr.predict(X)
    
    plt.figure(figsize=(12,8))
    plt.scatter(df['Adj Close'], lr.predict(X))
    plt.plot(X, lr.predict(X), color = 'red')
    plt.xlabel('Prices')
    plt.ylabel('Predicted Prices')
    plt.grid()
    plt.title(sym + ' Prices vs Predicted Prices')
    plt.show()
    print('_____________Summary:_____________')       
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    print("")
    ans = ['1', '2'] 
    user_input=input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)   
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input=input("Command: ")
    if user_input=="1":
        menu()
    elif user_input=="2":
        exit()    
        

# Support Vector Regression
def stock_svr():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    dates = np.reshape(df.index,(len(df.index), 1)) # convert to 1xn dimension
    x = 31
    x = np.reshape(x,(len(x), 1))
    prices = df['Adj Close']
    svr_lin  = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    # Fit regression model
    svr_lin .fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.figure(figsize=(12,8))
    plt.scatter(dates, prices, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    print('Linear Model:', svr_rbf.predict(x)[0])
    print('RBF Model:', svr_lin.predict(x)[0])
    print('Polynomial Model:', svr_poly.predict(x)[0])
    print("")
    ans = ['1', '2'] 
    user_input=input("""                  
What would you like to do next? Enter option 1 or 2.  
1. Menu
2. Exit
Command: """)   
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input=input("Command: ")
    if user_input=="1":
        menu()
    elif user_input=="2":
        exit()    
        

#***********************************************************************************************************************#
#******************************************************* Menu **********************************************************#
#***********************************************************************************************************************#  
def menu():
    ans = ['1', '2', '3', '4', '0'] 
    print(""" 
              
                           MENU
                  MACHINE LEARNING PREDICTION        
                  ---------------------------
                  1.Linear Regression
                  2.Logistic Regressions
                  3.Support Vector Regression
                  4.Beginning Menu
                  0.Exit the Program
                  """)
    user_input = input("Command (0-3): ") 
    while user_input not in ans:
        print("Error: Please enter a valid option 0-3")
        user_input=input("Command: ")             
    if user_input == '1':
        stock_linear_regression()
    elif user_input == '2':
        stock_logistic_regression()
    elif user_input == '3':
        stock_svr()
    elif user_input == "4":  
        beginning()
    elif user_input == "0":
        exit() 
        
        
#***********************************************************************************************************************#    
#*************************************************** Start of Program **************************************************# 
#***********************************************************************************************************************#  
def beginning():
    print()
    print("----------Welcome to Portfolio Measurements--------")
    print("""
Please choose option 1 or 2
              
1. Menu
2. Exit Program 

---------------------------------------------""")
    ans = ['1', '2'] 
    user_input=input("What is your Option?: ")    
    while user_input not in ans:
        print("Error: Please enter a a valid option 1-2")
        user_input=input("Command: ")
    if user_input=="1":
        menu()
    elif user_input=="2":
        exit()
  
    
#***********************************************************************************************************************#     
beginning()      
