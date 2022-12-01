#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

# Importing the libraries

## This version made 9/12 implements a feature to choose which accuracy metric is most desirable in order
## to save the best found model and load it back in for use after searching all training options.

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#import pandas_datareader as web #grab data from online
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler #replaced
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Bidirectional
import math
from sklearn.metrics import mean_squared_error
import sys
import csv
from datetime import datetime

start = datetime.now()


#grab the system arg
ticker_File = sys.argv[1] #"F"



print(ticker_File)


##This is var will be used to make a decision of which GRU module to save.
# options are either directional error "DirErr" or RMSE "RMSE"
useMetric = "RMSE"
curr_BestResult = 0
attrib_used_metric = []  # this will save which attributes were used that got the best metric

curr_BestResultRMSE = 1000
curr_BestResultDirErr = 0

model_SaveLocation = "saved_GRU_" + ticker_File.strip('.csv') + "_Model.h5" # This string needs to include the directory where the model will be saved.

training_Data_forCSV = [] #should be stock name, attributes, DirErr, RMSE
CSV_Header = ['Ticker', 'Attributes', 'DirErr', 'RMSE', 'Training_Time']
csv_file_name = "Training_data" + (ticker_File.strip('.csv'))


#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************
#*********************************************************************************************************************************************************

#Some functions to help out
def makeModelDataCSVFile(csv_file, startTime, endTime):
    try:
        with open(csv_file + '.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_Header)
            writer.writeheader()
            for data in training_Data_forCSV:
                writer.writerow(data)
            row0 = writer.next()
            row0.append('Training_Time')

    except Exception as e:
        print("GRU_Model data file Error\n",e)


#This is just for printing to the terminal in green for better visibility
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def prRed(skk): print("\033[93m {}\033[00m" .format(skk))



#The Scaler will take in the selected columns and scale them between 0-1 using the data normalization formula
def Scaler(arr, ADJMAX):
    minimum = 0  #All stocks can level off at worthless ($0.00)
    maximum = ADJMAX  #This will be changed so that an imperical estimation of growth max can be made.
    dummy = 0
    scaled_list = []
    for x in range(len(arr)):
        dummy = (arr[x] - minimum)/(maximum-minimum)
        scaled_list.append(dummy)
    #print(scaled_list)
    return scaled_list


#The Decaler will take in the selected columns and descale them out of 0-1 using the data normalization formula inverse
def Descaler(arr, ADJMAX):
    minimum = 0
    maximum = ADJMAX
    dummy = 0
    descaled_list = []
    for x in range(len(arr)):
        dummy = (arr[x]*(maximum-minimum)) + minimum
        descaled_list.append(dummy)
    return descaled_list

# PercIncr represents the day over day percent increase/decrease of stock price. This is a commonly used metric to
# attest a stocks performace both over days, to months, to years
def percIncr(data): #this will measure the percentage increase/decrease day to day
    percents = []
    for z in range(1,len(data)):
        per = (data[z]-data[z-1])/data[z-1]   # per[%] = (Day2 - Day1) / Day1
        percents.append(per)
    return percents


#This function will plot the data and predictions to a graph style similar to those that are used for stocks "Fivethiryeight"
# It takes as input the test and predicted data, and the lookback days to use a padding.
# The padding allows for visual verification why an initial prediction was made.
def plot_predictions(test,predicted,padding):
    plt.plot(test, color='red',label='Real Stock Price')  #plot the test data in red
    #plt.plot(range(padding, padding + len(predicted)), predicted, color='blue',label='Predicted Stock Price') #plot predicted
    plt.plot(predicted, color='blue',label='Predicted Stock Price') #plot predicted
    #Label parts of the plot
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


#This function will take in the test and predicted data and produce the rsme of the entire set it is given.
def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted)) #use rsme from the math library to calculate rsme across the whole set
    print("The root mean squared error is {}.".format(rmse))
    return rmse


#*************************************************************************************************
# This function will produce the directional accuracy in percent of the test vs prediction
def Directional_error(test, predicted):
    correct = 0
    wrong = 0
    for x in range(len(test)): #ignore first step as there is nothing to compare it to yet.
        if(x==0):
            continue
        elif((test[x]>test[x-1]) and (predicted[x]>predicted[x-1])): #case in which prediction and test went up
            correct +=1

        elif((test[x]<test[x-1]) and (predicted[x]<predicted[x-1])): #case in which prediction and test went down
            correct +=1

        elif((test[x]==test[x-1]) and (predicted[x]==predicted[x-1])): #case in which no change happened
            correct +=1

        elif(test[x] == predicted[x]):  #case in which both are the same
            correct+=1

        else:
            wrong+=1

    accuracy = round((correct/(correct+wrong))*100 , 2)
    print("The directional accuracy is {}%".format(accuracy))
    return accuracy


#*********************************************************************************************************

#The GRU Model will be built via a function call so that it may be altered to handle different array deminsions.
# Droput is used here to attempt to mitigate overfitting a model so that it can be better generalized.
#Learning rate has also been a previously altered variable that will be further tested later,
# it produced largely worse results upon trying to fine tune with LR


def GRUModelBuild(arrX, arrY, arrDem, daysback):
    # The GRU architecture
    ModelGRU = Sequential()

    # First GRU layer with Dropout regularisation
    ModelGRU.add(GRU(units=50, input_shape=(daysback,arrDem), return_sequences=True, activation='tanh'))
    ModelGRU.add(Dropout(0.1))   # activation Tanh is used as that is native to the GRU model, returning seq will keep
                                # the tensors shapes consistent between inputs

    # Second GRU layer
    ModelGRU.add(GRU(units=80, return_sequences=True, activation='tanh' ))
    ModelGRU.add(Dropout(0.2))

    # Third GRU layer
    ModelGRU.add(GRU(units=80, return_sequences=True, activation='tanh' ))
    ModelGRU.add(Dropout(0.1))

    #ModelGRU.add(GRU(units=100, return_sequences=True, activation='tanh' ))
    #ModelGRU.add(Dropout(0.3))

    ModelGRU.add(GRU(units=80, activation='tanh'))
    ModelGRU.add(Dropout(0.1))

    # The output layer
    ModelGRU.add(Dense(units=1)) # The final output is a 1D array of the predicted price point.
                                #A relu activation has been tested here before and proves to give better occasionally better directional

    #opt = keras.optimizers.Adam(learning_rate=0.001)
    ModelGRU.compile(optimizer='adam' , loss='mean_squared_error') #opt
    #Loss of MSE is best in this scenario as it is a measure of how far off the system is from the real value,
        # it acts similar to an averaged standard deviation. Closer to zero is better
    return ModelGRU

#**********************************************************************************************

#This function will take the training set and build an array of each data point that is looking back the
# number of specified days, it will take in the number of chars being used so it can shape properly
# the structure will be as  (size , #lookbackdays, chars )
# each lookback set has a corresponding Y value, this is in order to handle the offest of looking back
def LookbackTrain(arrX, arrY , chars, NumLookback):
    X_train_lookback = []
    y_train = []
    for i in range(NumLookback,len(arrX)): ## *************double check this len statement
        X_train_lookback.append(arrX[i-NumLookback:i])
        y_train.append(arrY[i,0])

    X_train_lookback, y_train = np.array(X_train_lookback), np.array(y_train)

    # Reshaping X_train for efficient modeling
    X_train_lookback = np.reshape(X_train_lookback, (X_train_lookback.shape[0],X_train_lookback.shape[1],chars))

    return X_train_lookback, y_train


#********************************************************************************************************************************
#This function will take the test set and build an array of each data point that is looking back the
# number of specified days, it will take in the number of chars being used so it can shape properly
# the structure will be as  (size , #lookbackdays, chars )
def LookbackTest(arrTest, arrTestY, arrTrain, chars, NumLookback):
    X_test_lookback_funct = []
    Y_dataTemp = []
    for i in range(len(arrTest)+1):  ## *************double check this len statement
        temp_list = []
        if(i <= NumLookback):
            temp_list = list(arrTrain[(len(arrTrain)-NumLookback)+i:]) + list(arrTest[:i])
            #if i is less than the numlookback then grab the data out of the training set for the lookback in the test
            X_test_lookback_funct.append(temp_list)
            Y_dataTemp.append(arrTestY[i,0])
        elif(i == len(arrTest)):
            X_test_lookback_funct.append(arrTest[i-NumLookback:i])
        else:
            X_test_lookback_funct.append(arrTest[i-NumLookback:i])
            Y_dataTemp.append(arrTestY[i,0])

    X_test_lookback_funct = np.array(X_test_lookback_funct)
    #print(X_test_lookback)
    X_test_lookback_funct = np.reshape(X_test_lookback_funct, (X_test_lookback_funct.shape[0],X_test_lookback_funct.shape[1],chars))

    #print(X_test_lookback.shape)
    #print(X_test_lookback)
    return X_test_lookback_funct, Y_dataTemp

#************************************************************************************************************************************

def scaleGT(arr, MaxGT):
    minimum = 0  #zero is no google trends searches
    maximum = MaxGT #int(np.nanmax(arr)*1.25)  #This will be changed so that an estimation max searches made.
    dummy = 0
    #print(arr)
    scaled_GTlist = []
    for x in range(len(arr)):
        if(pd.isna(arr[x])):
            #print(pd.isna(arr[x]))
            #math.isnan(arr[x])
            scaled_GTlist.append(np.nan)
        else:
            dummy = (int(arr[x]) - minimum)/(maximum-minimum)
            #print(dummy)
            scaled_GTlist.append(dummy)
    #print(scaled_GTlist)
    return scaled_GTlist

def scaleVolume(arr, MaxVol, MinVol): #currently unused
    minimum = MinVol #int((arr.min())*0.90)  #zero is no google trends searches
    maximum = MaxVol #int((arr.max())*1.15)  #This will be changed so that an estimation max searches made.
    dummy = 0
    #print(arr)
    scaled_Volumelist = []
    for x in range(len(arr)):
        if(pd.isna(arr[x])):
            #print(pd.isna(arr[x]))
            #math.isnan(arr[x])
            scaled_Volumelist.append(np.nan)
        else:
            dummy = (int(arr[x]) - minimum)/(maximum-minimum)
            #print(dummy)
            scaled_Volumelist.append(dummy)
    #print(scaled_Volumelist)
    return scaled_Volumelist



#************************************************************************************************************************************
#************************************************************************************************************************************
#************************************************************************************************************************************
#************************************************************************************************************************************

file_name_used = ticker_File + '.csv'
print(file_name_used + ": this file was called")
dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])

attribs_Inside_of_dataset = dataset.columns.tolist()


if(len(attribs_Inside_of_dataset) < 9):
    None
    ##RUN the data scrapper again because alot of the attributes where missing



dataset.index = pd.to_datetime(dataset.index) #grab the date column so values can be taken out of it.
dataset['Day'] = dataset.index.dayofyear  # list days as their day of year

#dataset = dataset.dropna() # remove all NaNs, due to these being incomplete data sets

if 'IR' in attribs_Inside_of_dataset:
    #print("IR was not in the dataset")
    dataset['IR'] = dataset['IR'].str.rstrip('%').astype('float') / 100.0  #strip the % from the data and then convert to decimal
else:
    prRed("IR was not in the dataset")


dataset = dataset.drop(columns = ['Adj Close']) # Remove adjusted close, may consider as attribute in later version
#dataset = dataset.drop(columns = ['Volume'])
print(dataset)

datapercents = percIncr(dataset['Close'])
#print(datapercents)
dataset.drop(dataset.head(1).index,inplace=True) # drop last last row
dataset['PercentChange'] = datapercents
#print(datapercents)

day_sin_signal = np.sin(dataset['Day']*(2*np.pi/365.24)) #convert the days into a proper signal
#plt.plot(day_sin_signal) #verify sinusoidal day list
#plt.show()

dataset = dataset.drop(columns = ['Day'])  #Drop the original days column and replace it with new signal version.
dataset['Day'] = day_sin_signal


MaxData = dataset[:'2021']['High'].max() #grba the max value out of the high
#print(MaxData)
MinData = dataset[:'2021']['High'].min() #grab the min value out of the high

#this conditional was to fix a case in which the scale factor would become drastically large
#Look into a another potential solution for this in the future
# solution could be to use the percentage of increase of the stock in the past ten years, this could also be bad for instances in which the
# the stock has been decreasing in value though

if(MinData < 1):
    MinData = dataset['2010':'2021']['High'].min()
    MaxData = dataset['2010':'2021']['High'].max()
    InflateFactor = (dataset['2022':'2022']['IR'].max()) + 1
    #print("curr IR:", InflateFactor)
    increase_max = (((MaxData-MinData)/MaxData)*InflateFactor) + 100
else:
    increase_max = (MaxData/MinData)/10  # This aids to find a "max" value the stock can exist at, this is needed for scaling

MaxData = dataset[:'2021']['High'].max()
#this is calculation is the high growth potential from min to max over and arbitrary 9 years.
Factor = increase_max*MaxData #save this calucation to use for the scalers

#print("Volume Factor:", VolumeFactor)
#print("Increase factor: ", Factor)

## minmaxscaler was not functioning how intended so i made my own scaling function.
dataset['High'] = Scaler(dataset['High'].values, Factor)
dataset['Low']  = Scaler(dataset['Low'].values, Factor)
dataset['Open'] = Scaler(dataset['Open'].values, Factor)
dataset['Close']= Scaler(dataset['Close'].values, Factor)

#Now scale the google trends data and the volume

#ScaleGT(Data, Max)
if "GT" in attribs_Inside_of_dataset:
    dataset['GT'] = scaleGT(dataset['GT'].values, int(np.nanmax(dataset[:'2021']['GT'])*1.25))
else: prRed("GT was not in the given dataset")


if "Volume" in attribs_Inside_of_dataset:
    dataset['Volume'] = scaleVolume(dataset['Volume'].values, int((dataset[:'2021']['Volume'].max())*1.15), int((dataset[:'2021']['Volume'].min())*0.90))
else: prRed("volume was not in the given dataset")


#print(dataset.keys())
#dataset.head()
#print(dataset)



#*****************************************************************************************************************************
# Here is a set of different combinations of attributes that will be used for training models
# Volume has been excluded for the time being. Using a permutation library could be good for this...
characteristics = {}


characteristics[" 1"] = ['High', 'Low', 'Open', 'EPS', 'Day']
#characteristics[" 2"] = ['High', 'Low', 'Open', 'Day']
characteristics[" 3"] = ['High', 'Low', 'Day']
characteristics[" 4"] = ['High', 'Low', 'Open']
characteristics[" 13"] = ['High', 'Low', 'IR']
characteristics[" 23"] = ['High', 'Low', 'IR', 'PercentChange']
characteristics[" 50"] = ['High','GT', 'Volume']
characteristics[" 82"] = ['High', 'Low', 'Open', 'Day', 'IR','PercentChange', 'Volume']

characteristics[" 8"] = ['High', 'Low']
characteristics[" 9"] = ['High', 'Low', 'EPS', 'Day']
characteristics[" 14"] = ['High', 'Low', 'Day', 'IR']
characteristics[" 17"] = ['High', 'Low', 'Day','PercentChange']
characteristics[" 15"] = ['High', 'Low', 'Open', 'EPS', 'Day','PercentChange']


'''
characteristics[" 8"] = ['High', 'Low']
characteristics[" 9"] = ['High', 'Low', 'EPS', 'Day']
characteristics[" 10"] = ['High']

characteristics[" 11"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR']
characteristics[" 91"] = ['High', 'Low', 'Open', 'Day', 'IR']
characteristics[" 12"] = ['High', 'Low', 'EPS', 'Day', 'IR']

characteristics[" 14"] = ['High', 'Low', 'Day', 'IR']

characteristics[" 15"] = ['High', 'Low', 'Open', 'EPS', 'Day','PercentChange']
characteristics[" 16"] = ['High', 'Low', 'Open', 'Day','PercentChange']
characteristics[" 17"] = ['High', 'Low', 'Day','PercentChange']
characteristics[" 18"] = ['High', 'Low', 'Open','PercentChange']
characteristics[" 19"] = ['High', 'Low', 'Day', 'IR', 'PercentChange']

characteristics[" 20"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR','PercentChange']
characteristics[" 21"] = ['High', 'Low', 'Open', 'Day', 'IR','PercentChange']
characteristics[" 22"] = ['High', 'Low', 'EPS', 'Day', 'IR','PercentChange']



##**********************************************************************

characteristics[" 24"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'GT']
characteristics[" 25"] = ['High', 'Low', 'Open', 'Day', 'GT']
characteristics[" 26"] = ['High', 'Low', 'Day', 'GT']
characteristics[" 27"] = ['High', 'Low', 'Open', 'GT']

characteristics[" 28"] = ['High', 'Low', 'GT']
characteristics[" 29"] = ['High', 'Low', 'EPS', 'Day', 'GT']
characteristics[" 30"] = ['High','GT']

characteristics[" 31"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR', 'GT']
characteristics[" 32"] = ['High', 'Low', 'Open', 'Day', 'IR', 'GT']
characteristics[" 33"] = ['High', 'Low', 'EPS', 'Day', 'IR', 'GT']
characteristics[" 34"] = ['High', 'Low', 'IR', 'GT']
characteristics[" 35"] = ['High', 'Low', 'Day', 'IR', 'GT']

characteristics[" 36"] = ['High', 'Low', 'Open', 'EPS', 'Day','PercentChange', 'GT']
characteristics[" 37"] = ['High', 'Low', 'Open', 'Day','PercentChange', 'GT']
characteristics[" 38"] = ['High', 'Low', 'EPS', 'Day', 'IR','PercentChange', 'GT']
characteristics[" 39"] = ['High', 'Low', 'IR', 'PercentChange', 'GT']
characteristics[" 40"] = ['High', 'Low', 'Open', 'IR', 'GT']


#********************************************************************************************

characteristics[" 44"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'GT', 'Volume']
characteristics[" 45"] = ['High', 'Low', 'Open', 'Day', 'GT', 'Volume']
characteristics[" 46"] = ['High', 'Low', 'Day', 'GT', 'Volume']
characteristics[" 47"] = ['High', 'Low', 'Open', 'GT', 'Volume']

characteristics[" 48"] = ['High', 'Low', 'GT', 'Volume']
characteristics[" 49"] = ['High', 'Low', 'EPS', 'Day', 'GT', 'Volume']


characteristics[" 51"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR', 'GT', 'Volume']
characteristics[" 52"] = ['High', 'Low', 'Open', 'Day', 'IR', 'GT', 'Volume']
characteristics[" 53"] = ['High', 'Low', 'EPS', 'Day', 'IR', 'GT', 'Volume']
characteristics[" 54"] = ['High', 'Low', 'IR', 'GT', 'Volume']
characteristics[" 35"] = ['High', 'Low', 'Day', 'IR', 'GT', 'Volume']

characteristics[" 56"] = ['High', 'Low', 'Open', 'EPS', 'Day','PercentChange', 'GT', 'Volume']
characteristics[" 57"] = ['High', 'Low', 'Open', 'Day','PercentChange', 'GT', 'Volume']
characteristics[" 58"] = ['High', 'Low', 'EPS', 'Day', 'IR','PercentChange', 'GT', 'Volume']
characteristics[" 59"] = ['High', 'Low', 'IR', 'PercentChange', 'GT', 'Volume']
characteristics[" 60"] = ['High', 'Low', 'Open', 'IR', 'GT', 'Volume']

characteristics[" 61"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'Volume']
characteristics[" 62"] = ['High', 'Low', 'Open', 'Day', 'Volume']
characteristics[" 63"] = ['High', 'Low', 'Day', 'Volume']
characteristics[" 64"] = ['High', 'Low', 'Open', 'Volume']

characteristics[" 68"] = ['High', 'Low', 'Volume']
characteristics[" 69"] = ['High', 'Low', 'EPS', 'Day', 'Volume']


characteristics[" 71"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR', 'Volume']
characteristics[" 72"] = ['High', 'Low', 'Open', 'Day', 'IR', 'Volume']
characteristics[" 73"] = ['High', 'Low', 'EPS', 'Day', 'IR', 'Volume']
characteristics[" 74"] = ['High', 'Low', 'IR', 'Volume']
characteristics[" 75"] = ['High', 'Low', 'Day', 'IR', 'Volume']

characteristics[" 76"] = ['High', 'Low', 'Open', 'EPS', 'Day','PercentChange', 'Volume']
characteristics[" 77"] = ['High', 'Low', 'Open', 'Day','PercentChange', 'Volume']
characteristics[" 78"] = ['High', 'Low', 'Day','PercentChange', 'Volume']
characteristics[" 79"] = ['High', 'Low', 'Open','PercentChange', 'Volume']
characteristics[" 80"] = ['High', 'Low', 'Day', 'IR', 'PercentChange', 'Volume']

characteristics[" 81"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR','PercentChange', 'Volume']

characteristics[" 83"] = ['High', 'Low', 'EPS', 'Day', 'IR','PercentChange', 'Volume']
characteristics[" 84"] = ['High', 'Low', 'IR', 'PercentChange', 'Volume']
'''

#****************************************************************************************************************************


## User input region and or section to handle the varying of arrays
## this may need to be added above.


lookback_days = 20 # number of days to lookback on the data sets

options = list(characteristics.keys()) # make a list out of the keys so they can be better tracked and iterated through.
#print(options[0])

output_var = pd.DataFrame(dataset['Close'])  ## Select the close value as the output data set, this will be used for "Y"


## implement a loop here********************************************
keras.backend.clear_session()#delete the GRU and clear all weight data, this is essential to train a new model after

#dataset = dataset.drop(columns = ['GT'])

bestModelAttributes = []

for j in range(len(options)):
    #numChars = len(characteristics[options[j]]) #save num of how many characteristics are being used for building model
    #I put "close" in all the attributes but this was a bad idea cause it changes both the len number and the chars used later on
    #What if you make a temp of

    for AttributeInOptions in characteristics[options[j]]:  ##This nest loop will quickly make sure that the dictionary attribute combo all exist in the dataset
        if AttributeInOptions not in (dataset.columns.tolist()):  ##if not in the dataset skip this training combo
            prRed("There is an expected Attribute missing: ", AttributeInOptions)
            break
    else:
        numChars = len(characteristics[options[j]]) #save num of how many characteristics are being used for building model
        templistofChars = characteristics[options[j]]
        templistofChars.append('Close')


        #print("This is the temp list of attributes: ",templistofChars)

        TempDataFrame = pd.DataFrame(dataset[templistofChars])
        #print(TempDataFrame)

        TempDataFrame = TempDataFrame.dropna() # remove all NaNs, due to these being incomplete data sets

        output_var = pd.DataFrame(TempDataFrame['Close'])  ## Select the close value as the output data set, this will be used for "Y"

        #print("TempDataFrame:")
        #print(TempDataFrame['2022':'2022'])
        #print("\n","This is the output_var, aka the y values")
        #print(output_var)
        TempDataFrame = TempDataFrame.drop("Close", axis=1)
        Y_train = output_var[:'2021'].values ## grab training data from outputvar
        Y_test = output_var['2022':'2022'].values #grab test data from outputvar

        #print(i)
        #print(options[j])
        #print(X_Values)

        X_Values = TempDataFrame #pd.DataFrame(TempDataFrame[characteristics[options[j]]])
        #*****************
        #change code here

        #print("This is the dataframe of Xvalues")
        #print(X_Values)

        X_train = X_Values[:'2021'].values
        X_test = X_Values['2022':'2022'].values  ## grouping day of year in and scaling...

        #print(X_test)
        '''
        Y_train = output_var[:'2021'].values ## grab training data from outputvar
        Y_test = output_var['2022':'2022'].values #grab test data from outputvar
    
        #print(i)
        #print(options[j])
        #print(X_Values)
    
        X_Values = pd.DataFrame( dataset[characteristics[options[j]]])
        print(X_Values)
        X_train = X_Values[:'2021'].values
        X_test = X_Values['2022':'2022'].values  ## grouping day of year in and scaling...
        #print(X_test)
    
        #*************************************************************************************************************
        #This section will take the Test set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back
        X_test_lookback = []
        Y_test_ready = []
        '''

        X_test_lookback, Y_test_ready = LookbackTest(X_test, Y_test, X_train, numChars, lookback_days)


        #print("Temp data frame",TempDataFrame['2022':'2022'])
        #print("X test lookback", X_test_lookback)
        #print("Y test ready", Y_test_ready)


        #*************************************************************************************************************
        #This section will take the Training set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back

        #X_train_lookback, Y_train_ready = LookbackTrain(X_train, Y_train, numChars , lookback_days)

        X_train_lookback = []
        Y_train_ready = []
        #print(training_set_scaled[54-54:54, 1])

        for i in range(lookback_days,len(X_train)): ## *************double check this len statement
            X_train_lookback.append(X_train[i-lookback_days:i])
            Y_train_ready.append(Y_train[i,0])

        X_train_lookback, Y_train_ready = np.array(X_train_lookback), np.array(Y_train_ready)
        #print(X_train_lookback)
        # Reshaping X_train for efficient modeling
        X_train_lookback = np.reshape(X_train_lookback, (X_train_lookback.shape[0],X_train_lookback.shape[1],numChars))

        #print("Xtrain lookback", X_train_lookback)
        #print("YTrain ready", Y_train_ready)
    #*************************************************************************************************************


        #print(X_train_lookback)
        #print(y_train)
        #print(numChars)   #verification step to prove the array dimensions
        #print("xtrain shape", X_train_lookback.shape) #verification step to prove the array dimensions
        #print("Y_train_ready shape", Y_train_ready.shape)

        #**************************************************************************************************************
        #This section handles building the model and then testing it out.


        #break

        Model1 = GRUModelBuild(X_train_lookback, Y_train_ready, numChars, lookback_days) #call the builder function above
        #Model1.summary()  ## This will print the structure of the GRU that has been built


        Model1.fit(X_train_lookback, Y_train_ready, validation_split=0.1, epochs = 10) # ,batch_size=1


        #train the model using the datasets, validation split will take a random 10% of the data out of training and use it
            # to test the model for validation, the result will show if genrealization is occuring properly
            # train over 15 iteraitions (epochs)

        GRU_predicted_stock_price = Model1.predict(X_test_lookback)
        #Predict the prices using the test set with lookback applied to it

        # Visualizing the results for GRU
        GRU_predicted_stock_price = Descaler(GRU_predicted_stock_price, Factor)
        Y_test_ready = Descaler(Y_test_ready, Factor)
        ## Descale the predictions, Y_test_ready was used for validation of descaling.

        Y_test = Descaler(Y_test, Factor)

        #plot_predictions(Y_test, GRU_predicted_stock_price, lookback_days)
        #Plot the predictions, the final attribute of this function allows for shifting the plots to see what occured before
            # the displayed prediction


        # Evaluating GRU
        tempHold_RMSE = return_rmse(Y_test_ready, GRU_predicted_stock_price[:len(GRU_predicted_stock_price)-1]) #call rmse to give error on data set
        tempHold_DirErr = Directional_error(Y_test_ready, GRU_predicted_stock_price[:len(GRU_predicted_stock_price)-1])

        if(useMetric == "RMSE"):
            if((tempHold_RMSE < curr_BestResultRMSE) or ((tempHold_DirErr > curr_BestResultDirErr) and (tempHold_RMSE == curr_BestResultRMSE) ) ):
                curr_BestResultRMSE = tempHold_RMSE
                curr_BestResultDirErr = tempHold_DirErr
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
                bestModelAttributes = characteristics[options[j]]
            else:
                print("This model was discarded")

        if(useMetric == "DirErr"):
            if((tempHold_DirErr > curr_BestResultDirErr) or ((tempHold_DirErr == curr_BestResultDirErr) and (tempHold_RMSE < curr_BestResultRMSE) ) ):
                curr_BestResultDirErr = tempHold_DirErr
                curr_BestResultRMSE = tempHold_RMSE
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
                bestModelAttributes = characteristics[options[j]]
            else:
                print("This model was discarded")


        #CSV_Header = ['Ticker', 'Attributes', 'DirErr', 'RMSE']
        training_Data_forCSV.append({'Ticker': ticker_File.strip('.csv'), 'Attributes': characteristics[options[j]], 'DirErr': tempHold_DirErr, 'RMSE': tempHold_RMSE})
        print("Attributes:", characteristics[options[j]]) #print what characteristics were used

        #print(Model1.summary()) #this will print out the structure of the GRU

        keras.backend.clear_session()#delete the GRU and clear all weight data, this is essential to train a new model after
        print("Model Cleared")




#This section will then load in the best found model found and use it to make predictions for the user******************************************************
#As well as make a call to create the csv file with all the stock data

end = datetime.now()

training_Data_forCSV.append({'Training_Time': start})
training_Data_forCSV.append({'Training_Time': end})


makeModelDataCSVFile(csv_file_name, start, end)


with open(ticker_File+"_Attributes.csv", 'w') as myfile:
    if "Close" in bestModelAttributes:
        bestModelAttributes.remove("Close")
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(bestModelAttributes)

reconstruct_Model = keras.models.load_model(model_SaveLocation)

#**********************************************************************************************************************************************************
