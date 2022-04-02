# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:21:27 2019

@author: Tin
"""
import numpy as np
import pandas as pd
import datetime
from sys import exit

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")

# yahoo finance used to fetch data 
import yfinance as yf
yf.pdr_override()

options = " Feature Selection, Exit".split(",")

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


#***********************************************************************************************************************#
# Univariate Selection
def Univariate_Selection():
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)

    # Feature extraction
    test = SelectKBest(score_func=chi2, k=3)
    fit = test.fit(X, Y)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    new_features = fit.transform(X)
    # Show results
    print("")
    print('Original number of features:', X.shape[1])
    print('Reduced number of features:', new_features.shape[1])
    # Summarize selected features
    print(new_features[0:5,:])
    print("")
    US = pd.DataFrame(fit.scores_, columns = ["Univariate_Selection"], index=features.columns)
    US = US.reset_index()
    print("")
    print(US.sort_values('Univariate_Selection',ascending=0))
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
# Recursive Feature Elimination
def Recursive_Feature_Elimination():
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    # Feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    print("")
    Selected = pd.DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)
    Selected = Selected.reset_index()
    print(Selected[Selected['RFE'] == True])
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
# Principal Component Analysis
def Principal_Component_Analysis():
    from sklearn.decomposition import PCA
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    # feature extraction
    pca = PCA(n_components=3)
    fit = pca.fit(X)
    # summarize components
    print(("Explained Variance: %s") % fit.explained_variance_ratio_)
    print(fit.components_)
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
# Feature Importance
def Feature_Importance():
    from sklearn.ensemble import ExtraTreesClassifier
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    ET = pd.DataFrame(model.feature_importances_, columns = ["Extra Trees"], index=features.columns)
    ET = ET.reset_index()
    print(ET.sort_values(['Extra Trees'],ascending=0))
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
# Random Forest Classifier
def Random_Forest_Classifier():
    from sklearn.ensemble import RandomForestClassifier
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    clf = RandomForestClassifier()
    clf.fit(X,Y)
    RFC = pd.DataFrame(clf.feature_importances_, columns = ["RFC"], index=features.columns)
    RFC = RFC.reset_index()
    print(RFC.sort_values(['RFC'],ascending=0))
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
# Chi Square on Features
def Chi_Square_on_Features():
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    model = SelectKBest(score_func=chi2, k=5)
    fit = model.fit(X, Y)
    print(fit.scores_)
    print("")
    chi_sq = pd.DataFrame(fit.scores_, columns = ["Chi_Square"], index=features.columns)
    chi_sq = chi_sq.reset_index()
    print(chi_sq.sort_values('Chi_Square',ascending=0))
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
# L1 Feature Selection
def L1_Feature_Selection():
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    array = features.values
    X = array.astype(int)
    Y = df['Adj Close'].values.astype(int)
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lsvc,prefit=True)
    l1 = pd.DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
    l1 = l1.reset_index()
    print(l1[l1['L1'] == True])
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
# Multicollinearity Variance Inflation Factor
def Multicollinearity_Variance_Inflation_Factor():
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    s = start_date() 
    e = end_date()
    sym = input_symbol()
    df = yf.download(sym, s, e)
    df['Increase_Decrease'] = np.where(df['Volume'].shift(-1) > df['Volume'],1,0)
    df['Buy_Sell_on_Open'] = np.where(df['Open'].shift(-1) > df['Open'],1,0)
    df['Buy_Sell'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'],1,0)
    df['Returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    features = df
    def calculate_vif(features):
        vif = pd.DataFrame()
        vif["Features"] = features.columns
        vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]    
        return(vif)
    vif = calculate_vif(features)
    while vif['VIF'][vif['VIF'] > 10].any():
        remove = vif.sort_values('VIF',ascending=0)['Features'][:1]
        features.drop(remove,axis=1,inplace=True)
        vif = calculate_vif(features)
    print(vif)
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
    ans = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
    print(""" 
              
                           MENU
                       FEATURE SELECTION      
                  ---------------------------
                  1. Univariate Selection
                  2. Recursive Feature Elimination
                  3. Principal Component Analysis
                  4. Feature Importance
                  5. Random Forest Classifier
                  6. Chi Square on Features
                  7. L1 Feature Selection
                  8. Multicollinearity Variance Inflation factor
                  9. Beginning Menu
                  10. Exit the Program
                  """)
    user_input = input("Command (1-10): ") 
    while user_input not in ans:
        print("Error: Please enter a valid option 1-10")
        user_input=input("Command: ")             
    if user_input == '1':
        Univariate_Selection()
    elif user_input == '2':
        Recursive_Feature_Elimination()
    elif user_input == '3':
        Principal_Component_Analysis()
    elif user_input == '4':
        Feature_Importance()
    elif user_input == '5':
        Random_Forest_Classifier()
    elif user_input == '6':
        Chi_Square_on_Features()
    elif user_input == '7':
        L1_Feature_Selection()
    elif user_input == '8':
        Multicollinearity_Variance_Inflation_Factor()
    elif user_input == "9":  
        beginning()
    elif user_input == "10":
        exit() 
        
        
#***********************************************************************************************************************#    
#*************************************************** Start of Program **************************************************# 
#***********************************************************************************************************************#  
def beginning():
    print()
    print("----------Welcome to Feature Selection for Machine Learning--------")
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