# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:21:27 2019

@author: Tin
"""
import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
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

# Rescaled Dataset
def Rescale_Dataset():
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
# Binarize Data 
def Binarize_Dataset():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    array = df.values
    X = array[:,0:5]
    Y = array[:,5]
    # initialising the binarize
    binarizer = Binarizer(threshold = 0.0).fit(X)
    binaryX = binarizer.transform(X)
    np.set_printoptions(precision=3)
    print('Binarize values equal or less than 0 are marked 0 and all of those above 0 are marked 1')
    print(binaryX[0:5,:])
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
# Standardize Data  
def Standardize_Dataset():
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    array = df.values
    X = array[:,0:5]
    Y = array[:,5]
    # initialising the standardize
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    np.set_printoptions(precision=3)
    print('Standardize values with a mean of 0 and a standard deviation of 1')
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
                     PREPROCESSING DATASET       
                  ---------------------------
                  1. Rescaled Data
                  2. Binarize Data 
                  3. Standardize Data  
                  4. Beginning Menu
                  0. Exit the Program
                  """)
    user_input = input("Command (0-3): ") 
    while user_input not in ans:
        print("Error: Please enter a valid option 0-3")
        user_input=input("Command: ")             
    if user_input == '1':
        Rescaled_Dataset()
    elif user_input == '2':
        Binarize_Dataset()
    elif user_input == '3':
        Standardize_Dataset()
    elif user_input == "4":  
        beginning()
    elif user_input == "0":
        exit() 
        
        
#***********************************************************************************************************************#    
#*************************************************** Start of Program **************************************************# 
#***********************************************************************************************************************#  
def beginning():
    print()
    print("----------Welcome to Preprocessing Dataset--------")
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