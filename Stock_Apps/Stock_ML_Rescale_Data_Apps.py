# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:21:27 2019

@author: Tin
"""
import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")

# yahoo finance used to fetch data 
import yfinance as yf
yf.pdr_override()

options = " Data Preprocessing, Exit".split(",")

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

def preprocessing_dataset():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    array = df.values
    X = array[:,0:5]
    Y = array[:,5]
    # initialising the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # learning the statistical parameters for each of the data and transforming
    rescaledX = scaler.fit_transform(X)
    np.set_printoptions(precision=3)
    print('Rescaled values between 0 to 1')
    print(rescaledX[0:5,:])
    print("")
    # Splitting the datasets into training sets and Test sets
    X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
    sc_X = StandardScaler()
    # Splitting the datasets into training sets and Test sets
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    print("Training Dataset")
    print(X_train)
    print("")
    print(Y_train)
    print("")
    print("Testing Dataset")
    print(X_test)
    print("")
    print(Y_test)
    return

    
def main():
    run_program = True
    while run_program:
        print("")
        print("__________Preprocessing Dataset__________")
        print("")
        print("Choose Options:")
        print("")
        for i in range(1, len(options)+1):
            print("{} - {}".format(i, options[i-1]))
        choice = int(input())
        
        if choice == 1:
             preprocessing_dataset()
        elif choice == 2:
             run_program = False             


if __name__ == "__main__":
    main()