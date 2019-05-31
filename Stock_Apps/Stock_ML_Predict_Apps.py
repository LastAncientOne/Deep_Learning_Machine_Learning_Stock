# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:21:27 2019

@author: Tin
"""
import numpy as np
import matplotlib.pyplot as plt

import datetime

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()

options = " Stock Linear Regression Prediction, Stock Logistic Regression Prediction, Exit".split(",")

# Input Start Date
def start_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    start = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    start = start.strftime('%Y-%m-%d')
    return start

# Input End Date
def end_date():
    date_entry = input('Enter a starting date in MM/DD/YYYY format: ')
    end = datetime.datetime.strptime(date_entry,'%m/%d/%Y')
    end = end.strftime('%Y-%m-%d')
    return end

# Input Symbols
def input_symbol():
    symbol = input("Enter symbol: ").upper()
    return symbol

# Features Analysis
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
    cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
    print(cross_val)
    print(cross_val.mean())
    return


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
    print('Summary:')       
    print('Estimate intercept coefficient:', lr.intercept_)
    print('Number of coefficients:', len(lr.coef_))
    print('Accuracy Score:', lr.score(X, Y))
    return

    
def main():
    run_program = True
    while run_program:
        print("__________Stock Price Prediction__________")
        print("____________Linear Regression_____________")
        print("Choose Options:")
        for i in range(1, len(options)+1):
            print("{} - {}".format(i, options[i-1]))
        choice = int(input())
        
        if choice == 1:
             stock_linear_regression()
        elif choice == 2:
             stock_logistic_regression()
        elif choice == 3:
             run_program = False             


if __name__ == "__main__":
    main()
