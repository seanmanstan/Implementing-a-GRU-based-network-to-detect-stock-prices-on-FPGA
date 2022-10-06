#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing dependencies
import datetime as dt  # for working with dates
import pandas as pd  # for working with large lists and large csv files
from pandas_datareader import data as pdr  # for main stock data
import csv  # for working with csv files
import requests  # for html parsing
import bs4  # for html parsing
from bs4 import BeautifulSoup  # for html parsing
import pytrends  # for google trends
from pytrends.request import TrendReq  # for google trends
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('fivethirtyeight')
import pandas_datareader as web  # grab data from online
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # replaced
from tensorflow import keras
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
# from keras. utils.vis_utils import plot_model
from keras.layers import Dense, Dropout, GRU, Bidirectional
# from keras.optimizers import sgd
import math
from sklearn.metrics import mean_squared_error
import sys
import csv

# enter start date, limited by whatever stock has the "least oldest" stock data available
startDate = input("Enter START date in format 'YYYY-MM-DD': ")  # get start date from user
startDate = dt.datetime.strptime(startDate, '%Y-%m-%d')  # make start date a "datetime" object in order to work with it
# enter end date
endDate = dt.datetime.now()  # set end date to today ("datetime" object)

# prompt user for stock tickers
userInput = input("Enter stock tickers of interest, one at a time, followed by the 'Enter' key (enter q to stop): ")
userInput = userInput.upper()  # make stock ticker all capital letters

stockTickerArray = []  # empty list of stock tickers
while (userInput != 'Q'):
    try:
        currentData = pdr.DataReader(userInput, 'yahoo', startDate,
                                     endDate)  # try to get main stock data from yahoo finance
    except:
        print("Stock ticker invalid please enter another or press 'q' to stop: ")
    else:
        stockTickerArray.append(userInput)  # add valid stock ticker to list of valid stick tickers
        currentFileName = userInput + ".csv"  # create base file
        with open(currentFileName, 'w',
                  encoding='UTF8') as f:  # this writes the current stock tickers data to a csv file
            currentData.to_csv(
                currentFileName)  # save data retriebed from yahoo finance (dataframe object) to the base csv

    userInput = input()  # get next stock ticker from user
    userInput = userInput.upper()

for x in stockTickerArray:  # iterate over every stock ticker in array
    stock_ticker = x
    print("Getting", stock_ticker, "stock data...")

    fileName = stock_ticker + ".csv"  # current file name is based on the current stock ticker

    ###########################################################################################################################
    ################################################################EPS########################################################

    # Macrotrends for past quarter eps data

    URL = "https://www.macrotrends.net/stocks/charts/" + stock_ticker + "/alphabet/eps-earnings-per-share-diluted"
    # get url where eps data is

    page = requests.get(URL)  # gets HTML from from site using its URL
    soup = BeautifulSoup(page.content, "html.parser")  # this line ensures we use the right parser for HTML
    chart = soup.find_all("td")  # find all objects in html with attribute "<td>"

    i = 0
    while i < len(chart):  # i indexes in range from 0 - (# of elements -1)
        if chart[i].text.startswith(('$', '2')):  # POTENTIALLY desireable data (dates and prices)
            if (chart[i].text.isnumeric()):  # undesirable data (ex: "2012" and its avg (yearly EPS price))
                del chart[i + 1]  # HAVE TO DELETE THE ONE WITH HIGHER INDEX FIRST
                del chart[(i)]
            else:  # desireable data (i.e., quarter dates and quarter prices)
                i += 1
        else:  # once this condition is reached only undesireable data is afterwards due to format of the sites HTMl
            while i != len(chart):
                del chart[i]
    # chart has data of eps quarter dates on even indexes (incl. "0") and EPS values on odd values (incl. "1")

    epsDataList = []
    for x in chart:
        if (x.text[0] == "$"):
            epsDataList.append(x.text[1:])
        else:
            epsDataList.append(x.text)
    # epsDataList has eps quarter dates on even indexes (incl. "0") and EPS values on odd values (incl. "1")
    # with most recent data at the beginning

    # find earliest and latest (most current) date
    earliestEpsDate = dt.datetime.now()  # set to random date that is forward in time of earliest EPS date
    latestEpsDate = dt.datetime.strptime("1900-01-01",
                                         '%Y-%m-%d')  # set to random date that is further back in time of latest EPS date

    # actually find correct earliest and latest eps dates
    i = 0
    while i < (len(epsDataList) - 1):
        tempDate = dt.datetime.strptime(epsDataList[i], '%Y-%m-%d')
        if (tempDate < earliestEpsDate):
            earliestEpsDate = tempDate
        if tempDate > latestEpsDate:
            latestEpsDate = tempDate
        i += 2

    # create column to be appended to csv after main data is created
    epsColumn = []
    i = 0
    with open(fileName, 'r', encoding='utf-8') as f:  # open file
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:  # look at one row at a time
            date = row[0]
            if (date != "Date"):  # ignore the header column
                fileDate = dt.datetime.strptime(date, '%Y-%m-%d')  # convert date in file to datetime object
                if (fileDate < (earliestEpsDate - dt.timedelta(
                        weeks=13))):  # check if date in the file is before the earliest quarter we have data for
                    epsColumn.append("NaN")
                elif (
                        fileDate > latestEpsDate):  # check if date in the file is a date that is in the current fiscal quarter
                    epsColumn.append("NaN")
                else:  # we have data for these dates
                    while i < (len(epsDataList) - 2):
                        if (fileDate < earliestEpsDate):
                            epsColumn.append(epsDataList[-1])  # end of the list is where the earliest data is stored
                            break
                        # format of list is "Date (\n) Price (\n) Date (\n)....(\n) Price"
                        tempDateRecent = dt.datetime.strptime(epsDataList[i], '%Y-%m-%d')
                        tempDatePast = dt.datetime.strptime(epsDataList[i + 2], '%Y-%m-%d')
                        if (tempDateRecent >= fileDate >= tempDatePast):
                            epsColumn.append(epsDataList[i + 1])
                            if (i != 0):
                                i = i - 2
                            break
                        else:
                            i += 2

    ###########################################################################################################################
    ################################################################INSIDER####################################################

    URL = "http://openinsider.com/screener?s=" + stock_ticker + "&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&" + "fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=" + "9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
    page = requests.get(URL)  # gets HTML from from site using its URL
    soup = BeautifulSoup(page.content, "html.parser")  # this line ensures we use the right parser for HTML
    table = soup.find(class_="tinytable")

    amt_of_buys = []
    amt_of_sells = []
    amt_traded = []
    try:
        data = table.find_all("td")
    except:
        with open(fileName, 'r', encoding='utf-8') as f:  # open file
            csv_reader = csv.reader(f, delimiter=",")
            for row in csv_reader:  # look at one row at a time
                date = row[0]
                if (date != "Date"):  # ignore the header column
                    amt_of_buys.append("NaN")
                    amt_of_sells.append("NaN")
                    amt_traded.append("NaN")

    else:
        trade_dates = []  # (Trade date) index 2
        titles = []  # (Title) index 5
        trade_type = []  # (Trade Type) index 5
        transaction = []  # (Qty) index 8
        amtOwned = []  # (Owned) index 9

        i = 0
        while i < len(data):
            trade_dates.append(data[i + 2].text)
            titles.append(data[i + 5].text)
            trade_type.append(data[i + 6].text)
            transaction.append(data[i + 8].text)
            amtOwned.append(data[i + 9].text)
            i += 16

        # Make dates in trade date array into date time objects
        i = 0
        while i < len(trade_dates):
            trade_dates[i] = dt.datetime.strptime(trade_dates[i], '%Y-%m-%d')
            i += 1

        # remove comma from numbers
        i = 0
        while i < len(transaction):
            transaction[i] = transaction[i].replace(',', '')
            i += 1

        with open(fileName, 'r', encoding='utf-8') as f:  # open file
            csv_reader = csv.reader(f, delimiter=",")
            for row in csv_reader:  # look at one row at a time
                date = row[0]
                if (date != "Date"):  # ignore the header column
                    fileDate = dt.datetime.strptime(date, '%Y-%m-%d')  # convert date in file to datetime object
                    if (fileDate < trade_dates[-1]):  # if date in file is less than earliest date
                        amt_of_buys.append("NaN")
                        amt_of_sells.append("NaN")
                        amt_traded.append("NaN")
                    elif (fileDate > trade_dates[0]):  # if date in file is more recent than latest date
                        amt_of_buys.append("NaN")
                        amt_of_sells.append("NaN")
                        amt_traded.append("NaN")
                    else:
                        try:
                            index = trade_dates.index(
                                fileDate)  # find at what index is the file date in the trade dates list
                        except ValueError:  # no reported trades on this day
                            amt_of_buys.append(0)
                            amt_of_sells.append(0)
                            amt_traded.append(0)
                        else:
                            # If the date from csv is found in trade_dates list
                            transactionCounter = 0
                            buyCounter = 0
                            sellCounter = 0
                            # count all buys, sells, and total amount traded
                            while index < len(trade_dates):
                                if (trade_dates[index] == fileDate):
                                    transactionCounter += int(transaction[index])
                                    if (trade_type[index] == "P - Purchase"):
                                        sellCounter += 1
                                    else:
                                        buyCounter += 1
                                else:
                                    break
                                index += 1
                            amt_of_buys.append(buyCounter)
                            amt_of_sells.append(sellCounter)
                            amt_traded.append(transactionCounter)

    ###########################################################################################################################
    ############################################################INFLATION######################################################

    # base year = str(startDate.year)
    # $1 in (base_year) is roughly equivalent (in purchasing power) to (inflated amount) in (whatever year you are looking at)

    url = "https://www.officialdata.org/us/inflation/" + str(startDate.year) + "?amount=1#buying-power"
    page = requests.get(url)  # gets HTML from from site using its URL
    soup = BeautifulSoup(page.content, "html.parser")  # this line ensures we use the right parser for HTML\
    table = soup.find(class_="regular-data table-striped")  # find table with desired data (html class = ...)
    data = table.find_all("tr")  # find all elements wrapped in <tr> tag in table

    # for x in data:
    #    print(x)
    #    print(x.text[0])
    #    print(x.text.splitlines())
    #    print(x.text.splitlines()[0][0])

    inflationDataList = []

    # get data from chart
    for x in data:
        if x.text[0] != 'Y':
            year = int(x.text[0] + x.text[1] + x.text[2] + x.text[3])
            dollar_value = (x.text[5] + x.text[6] + x.text[7] + x.text[8])
            inflation_rate = (x.text[9] + x.text[10] + x.text[11] + x.text[12] + x.text[13])
            inflationDataList.extend((year, dollar_value, inflation_rate))

    # get data from chart
    # for x in data:
    #    if x.text.splitlines()[1] != "Year":
    #        year = int(x.text.splitlines()[1])
    #        dollar_value = x.text.splitlines()[2]
    #        inflation_rate = x.text.splitlines()[3]
    #        inflationDataList.extend((year, dollar_value, inflation_rate))

    inflationColumn = []
    i = 0
    with open(fileName, 'r', encoding='utf-8') as f:  # open file
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:  # look at one row at a time
            date = row[0]
            if (date != "Date"):  # ignore the header column
                fileDate = dt.datetime.strptime(date, '%Y-%m-%d')  # convert date in file to datetime object
                while i * 3 < (len(inflationDataList) - 1):
                    if int(fileDate.year) == inflationDataList[3 * i]:
                        inflationColumn.append(inflationDataList[(3 * i) + 2])
                        break
                    else:
                        i += 1

    ###########################################################################################################################
    ##################################################GOOGLE TREND#############################################################

    pytrends = TrendReq(hl='en-US', tz=360)  # connect to google
    kw_list = [stock_ticker]  # list of keywords (doing one at a time) (may  generate 409 error if too many tickers)

    stock_df = pd.read_csv(fileName)
    trend_start_date = stock_df.iloc[0, 0]  # get earliest stock data date
    trend_end_date = stock_df.iloc[-1, 0]  # get most recent stock data date
    trend_time_frame = "2004-01-01" + " " + trend_end_date

    pytrends.build_payload(kw_list, cat=0, timeframe=trend_time_frame, geo='', gprop='')  # build payload
    data = pytrends.interest_over_time()  # put data in df object

    i = 0
    gtrendData = []
    with open(fileName, 'r', encoding='utf-8') as f:  # open file
        csv_reader = csv.reader(f, delimiter=",")
        for row in csv_reader:  # look at one row at a time
            date = row[0]
            if (date != "Date"):  # ignore the header column
                fileDate = dt.datetime.strptime(date, '%Y-%m-%d')  # convert date in file to datetime object
                if (fileDate < data.index[0]):  # check if date in the file is before the earliest date we have data for
                    gtrendData.append("NaN")
                elif (fileDate > data.index[-1]):  # check if date in the file is a date that is too recent
                    gtrendData.append("NaN")
                    # print("file date = ", fileDate, "trend date = ", data.index[-1])
                else:  # we have data for these dates
                    while i < (len(data)):
                        if (data.index[i].month == fileDate.month):
                            gtrendData.append(data.iloc[i, 0])
                            break
                        else:
                            i += 1

    # print(gtrendData)

    ###########################################################################################################################
    #################################################APPEND EVERYTHING#########################################################

    df = pd.read_csv(fileName)
    #buys_column = pd.DataFrame({'Buys': amt_of_buys})
    #sells_column = pd.DataFrame({'Sells': amt_of_sells})
    #traded_column = pd.DataFrame({'Traded': amt_traded})
    eps_column = pd.DataFrame({'EPS': epsColumn})
    inflation_column = pd.DataFrame({'IR': inflationColumn})
    gt_column = pd.DataFrame({'GT': gtrendData})
    #df = df.merge(buys_column, left_index=True, right_index=True)
    #df = df.merge(sells_column, left_index=True, right_index=True)
    #df = df.merge(traded_column, left_index=True, right_index=True)
    df = df.merge(eps_column, left_index=True, right_index=True)
    df = df.merge(inflation_column, left_index=True, right_index=True)
    df = df.merge(gt_column, left_index=True, right_index=True)
    df.to_csv(fileName, index=False)
    print(stock_ticker, "stock data gathering complete! :^) ")

############################################################################################################################
############################################################################################################################
############################################################################################################################
#GRU
print("Script complete! :D ")
for m in stockTickerArray:
    # !/usr/bin/env python
    # coding: utf-8

    # !/usr/bin/env python
    # coding: utf-8

    # Importing the libraries

    ## This version made 9/12 implements a feature to choose which accuracy metric is most desirable in order
    ## to save the best found model and load it back in for use after searching all training options.



    # grab the system arg
    ticker_File = m
    start= datetime.now()
    # sys.argv[1]
    print(ticker_File)

    ##This is var will be used to make a decision of which GRU module to save.
    # options are either directional error "DirErr" or RMSE "RMSE"
    useMetric = "RMSE"
    curr_BestResult = 0
    attrib_used_metric = []  # this will save which attributes were used that got the best metric

    curr_BestResultRMSE = 1000
    curr_BestResultDirErr = 0

    model_SaveLocation = "saved_GRU_" + ticker_File.strip(
        '.csv') + "_Model.h5"  # This string needs to include the directory where the model will be saved.

    training_Data_forCSV = []  # should be stock name, attributes, DirErr, RMSE
    CSV_Header = ['Ticker', 'Attributes', 'DirErr', 'RMSE']
    csv_file_name = "Training_data" + ticker_File


    # Some functions to help out

    def makeModelDataCSVFile(csv_file):
        try:
            with open(csv_file + '.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_Header)
                writer.writeheader()
                for data in training_Data_forCSV:
                    writer.writerow(data)
        except:
            print("GRU_Model data file Error")


    # This is just for printing to the terminal in green for better visibility
    def prGreen(skk):
        print("\033[92m {}\033[00m".format(skk))


    # The Scaler will take in the selected columns and scale them between 0-1 using the data normalization formula
    def Scaler(arr, ADJMAX):
        minimum = 0  # All stocks can level off at worthless ($0.00)
        maximum = ADJMAX  # This will be changed so that an imperical estimation of growth max can be made.
        dummy = 0
        scaled_list = []
        for x in range(len(arr)):
            dummy = (arr[x] - minimum) / (maximum - minimum)
            scaled_list.append(dummy)
        # print(scaled_list)
        return scaled_list


    # The Decaler will take in the selected columns and descale them out of 0-1 using the data normalization formula inverse
    def Descaler(arr, ADJMAX):
        minimum = 0
        maximum = ADJMAX
        dummy = 0
        descaled_list = []
        for x in range(len(arr)):
            dummy = (arr[x] * (maximum - minimum)) + minimum
            descaled_list.append(dummy)
        return descaled_list


    # PercIncr represents the day over day percent increase/decrease of stock price. This is a commonly used metric to
    # attest a stocks performace both over days, to months, to years
    def percIncr(data):  # this will measure the percentage increase/decrease day to day
        percents = []
        for z in range(1, len(data)):
            per = (data[z] - data[z - 1]) / data[z - 1]  # per[%] = (Day2 - Day1) / Day1
            percents.append(per)
        return percents


    # This function will plot the data and predictions to a graph style similar to those that are used for stocks "Fivethiryeight"
    # It takes as input the test and predicted data, and the lookback days to use a padding.
    # The padding allows for visual verification why an initial prediction was made.
    def plot_predictions(test, predicted, padding):
        plt.plot(test, color='red', label='Real Stock Price')  # plot the test data in red
        # plt.plot(range(padding, padding + len(predicted)), predicted, color='blue',label='Predicted Stock Price') #plot predicted
        plt.plot(predicted, color='blue', label='Predicted Stock Price')  # plot predicted
        # Label parts of the plot
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()


    # This function will take in the test and predicted data and produce the rsme of the entire set it is given.
    def return_rmse(test, predicted):
        rmse = math.sqrt(mean_squared_error(test,
                                            predicted))  # use rsme from the math library to calculate rsme across the whole set
        print("The root mean squared error is {}.".format(rmse))
        return rmse


    # *************************************************************************************************
    # This function will produce the directional accuracy in percent of the test vs prediction
    def Directional_error(test, predicted):
        correct = 0
        wrong = 0

        for x in range(len(test)):  # ignore first step as there is nothing to compare it to yet.
            if (x == 0):
                continue
            elif ((test[x] > test[x - 1]) and (
                    predicted[x] > predicted[x - 1])):  # case in which prediction and test went up
                correct += 1

            elif ((test[x] < test[x - 1]) and (
                    predicted[x] < predicted[x - 1])):  # case in which prediction and test went down
                correct += 1

            elif ((test[x] == test[x - 1]) and (predicted[x] == predicted[x - 1])):  # case in which no change happened
                correct += 1

            elif (test[x] == predicted[x]):  # case in which both are the same
                correct += 1

            else:
                wrong += 1

        accuracy = round((correct / (correct + wrong)) * 100, 2)
        print("The directional accuracy is {}%".format(accuracy))
        return accuracy


    # *********************************************************************************************************

    # The GRU Model will be built via a function call so that it may be altered to handle different array deminsions.
    # Droput is used here to attempt to mitigate overfitting a model so that it can be better generalized.
    # Learning rate has also been a previously altered variable that will be further tested later,
    # it produced largely worse results upon trying to fine tune with LR

    def GRUModelBuild(arrX, arrY, arrDem, daysback):
        # The GRU architecture
        ModelGRU = Sequential()

        # First GRU layer with Dropout regularisation
        ModelGRU.add(GRU(units=50, input_shape=(daysback, arrDem), return_sequences=True, activation='tanh'))
        ModelGRU.add(
            Dropout(0.1))  # activation Tanh is used as that is native to the GRU model, returning seq will keep
        # the tensors shapes consistent between inputs

        # Second GRU layer
        ModelGRU.add(GRU(units=80, return_sequences=True, activation='tanh'))
        ModelGRU.add(Dropout(0.2))

        # Third GRU layer
        ModelGRU.add(GRU(units=80, return_sequences=True, activation='tanh'))
        ModelGRU.add(Dropout(0.2))

        ModelGRU.add(GRU(units=100, return_sequences=True, activation='tanh'))
        ModelGRU.add(Dropout(0.3))

        ModelGRU.add(GRU(units=80, activation='tanh'))
        ModelGRU.add(Dropout(0.1))

        # The output layer
        ModelGRU.add(Dense(units=1))  # The final output is a 1D array of the predicted price point.
        # A relu activation has been tested here before and proves to give better occasionally better directional

        # opt = keras.optimizers.Adam(learning_rate=0.001)
        ModelGRU.compile(optimizer='adam', loss='mean_squared_error')  # opt
        # Loss of MSE is best in this scenario as it is a measure of how far off the system is from the real value,
        # it acts similar to an averaged standard deviation. Closer to zero is better
        return ModelGRU


    # **********************************************************************************************

    # This function will take the training set and build an array of each data point that is looking back the
    # number of specified days, it will take in the number of chars being used so it can shape properly
    # the structure will be as  (size , #lookbackdays, chars )
    # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back
    def LookbackTrain(arrX, arrY, chars, NumLookback):
        X_train_lookback = []
        y_train = []
        for i in range(NumLookback, len(arrX)):  ## *************double check this len statement
            X_train_lookback.append(arrX[i - NumLookback:i])
            y_train.append(arrY[i, 0])

        X_train_lookback, y_train = np.array(X_train_lookback), np.array(y_train)

        # Reshaping X_train for efficient modeling
        X_train_lookback = np.reshape(X_train_lookback, (X_train_lookback.shape[0], X_train_lookback.shape[1], chars))

        return X_train_lookback, y_train


    # ********************************************************************************************************************************
    # This function will take the test set and build an array of each data point that is looking back the
    # number of specified days, it will take in the number of chars being used so it can shape properly
    # the structure will be as  (size , #lookbackdays, chars )
    def LookbackTest(arrTest, arrTestY, arrTrain, chars, NumLookback):
        X_test_lookback_funct = []
        Y_dataTemp = []
        for i in range(len(arrTest)):  ## *************double check this len statement
            temp_list = []
            if (i <= NumLookback):
                temp_list = list(arrTrain[(len(arrTrain) - NumLookback) + i:]) + list(arrTest[:i])
                # if i is less than the numlookback then grab the data out of the training set for the lookback in the test
                X_test_lookback_funct.append(temp_list)
                Y_dataTemp.append(arrTestY[i, 0])
            else:
                X_test_lookback_funct.append(arrTest[i - NumLookback:i])
                Y_dataTemp.append(arrTestY[i, 0])

        X_test_lookback_funct = np.array(X_test_lookback_funct)
        # print(X_test_lookback)
        X_test_lookback_funct = np.reshape(X_test_lookback_funct,
                                           (X_test_lookback_funct.shape[0], X_test_lookback_funct.shape[1], chars))

        # print(X_test_lookback.shape)
        print(X_test_lookback)
        return X_test_lookback_funct, Y_dataTemp


    # ************************************************************************************************************************************

    def EvaluationAndPlot():  # currently unused
        None
        return


    file_name_used = ticker_File + '.csv'
    print(file_name_used + ": this file was called")
    dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])

    dataset.index = pd.to_datetime(dataset.index)  # grab the date column so values can be taken out of it.
    dataset['Day'] = dataset.index.dayofyear  # list days as their day of year
    dataset = dataset.dropna()  # remove all NaNs, due to these being incomplete data sets
    dataset['IR'] = dataset['IR'].str.rstrip('%').astype(
        'float') / 100.0  # strip the % from the data and then convert to decimal

    dataset = dataset.drop(columns=['Adj Close'])  # Remove adjusted close, may consider as attribute in later version
    dataset = dataset.drop(columns=['Volume'])
    print(dataset)
    datapercents = percIncr(dataset['Close'])
    # print(datapercents)
    dataset.drop(dataset.head(1).index, inplace=True)  # drop last last row becasue
    dataset['PercentChange'] = datapercents
    # print(datapercents)

    day_sin_signal = np.sin(dataset['Day'] * (2 * np.pi / 365.24))  # convert the days into a proper signal
    # plt.plot(day_sin_signal) #verify sinusoidal day list
    # plt.show()

    dataset = dataset.drop(columns=['Day'])  ## Drop the original days column and replace it with new signal version.
    dataset['Day'] = day_sin_signal

    MaxData = dataset[:'2021']['High'].max()  # grba the max value out of the high
    # print(MaxData)
    MinData = dataset[:'2021']['High'].min()  # grab the min value out of the high
    increase_max = (
                               MaxData / MinData) / 9  # This aids to find a "max" value the stock can exist at, this is needed for scaling
    # this is calculation is the high growth potential from min to max over and arbitrary 9 years.

    Factor = increase_max * MaxData  # save this calucation to use for the sclaers

    ## minmaxscaler was garbage so i made my own simple scaling function.
    dataset['High'] = Scaler(dataset['High'].values, Factor)
    dataset['Low'] = Scaler(dataset['Low'].values, Factor)
    dataset['Open'] = Scaler(dataset['Open'].values, Factor)
    dataset['Close'] = Scaler(dataset['Close'].values, Factor)

    print(dataset.keys())
    # dataset.head()
    print(dataset)

    # In[20]:

    # *****************************************************************************************************************************
    # Here is a set of different combinations of attributes that will be used for training models
    # Volume has been exluded for the time being. Using a permutation library could be good for this...
    characteristics = {}
    # characteristics["arrayset1"] = ['High', 'Low', 'Open', 'EPS', 'Day']
    characteristics["arrayset2"] = ['High', 'Low', 'Open', 'Day']
    characteristics["arrayset3"] = ['High', 'Low', 'Day']
    characteristics["arrayset4"] = ['High', 'Low', 'Open']

    characteristics["arrayset8"] = ['High', 'Low']
    characteristics["arrayset9"] = ['High', 'Low', 'EPS', 'Day']
    characteristics["arrayset10"] = ['High']

    characteristics["arrayset11"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR']
    characteristics["arrayset11"] = ['High', 'Low', 'Open', 'Day', 'IR']
    characteristics["arrayset12"] = ['High', 'Low', 'EPS', 'Day', 'IR']
    characteristics["arrayset13"] = ['High', 'Low', 'IR']
    characteristics["arrayset14"] = ['High', 'Low', 'Day', 'IR']

    characteristics["arrayset15"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'PercentChange']
    characteristics["arrayset16"] = ['High', 'Low', 'Open', 'Day', 'PercentChange']
    characteristics["arrayset17"] = ['High', 'Low', 'Day', 'PercentChange']
    characteristics["arrayset18"] = ['High', 'Low', 'Open', 'PercentChange']
    characteristics["arrayset19"] = ['High', 'Low', 'Day', 'IR', 'PercentChange']

    characteristics["arrayset20"] = ['High', 'Low', 'Open', 'EPS', 'Day', 'IR', 'PercentChange']
    characteristics["arrayset21"] = ['High', 'Low', 'Open', 'Day', 'IR', 'PercentChange']
    characteristics["arrayset22"] = ['High', 'Low', 'EPS', 'Day', 'IR', 'PercentChange']
    characteristics["arrayset23"] = ['High', 'Low', 'IR', 'PercentChange']

    # ****************************************************************************************************************************

    ## User input region and or section to handle the varying of arrays
    ## this may need to be added above.

    lookback_days = 30  # number of days to lookback on the data sets

    options = list(
        characteristics.keys())  # make a list out of the keys so they can be better tracked and iterated through.
    # print(options[0])

    output_var = pd.DataFrame(
        dataset['Close'])  ## Select the close value as the output data set, this will be used for "Y"

    ## implement a loop here********************************************
    keras.backend.clear_session()  # delete the GRU and clear all weight data, this is essential to train a new model after

    for j in range(len(options)):
        numChars = len(
            characteristics[options[j]])  # save num of how many characteristics are being used for building model

        Y_train = output_var[:'2021'].values  ## grab training data from outputvar
        Y_test = output_var['2022':'2022'].values  # grab test data from outputvar

        # print(i)
        # print(options[j])
        # print(X_Values)

        X_Values = pd.DataFrame(dataset[characteristics[options[j]]])
        print(X_Values)
        X_train = X_Values[:'2021'].values
        X_test = X_Values['2022':'2022'].values  ## grouping day of year in and scaling...
        # print(X_test)

        # *************************************************************************************************************
        # This section will take the Test set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back
        X_test_lookback = []
        Y_test_ready = []

        X_test_lookback, Y_test_ready = LookbackTest(X_test, Y_test, X_train, numChars, lookback_days)

        # *************************************************************************************************************
        # This section will take the Training set and build an array of each data point that is looking back the
        # number of specified days, it will take in the number of chars being used so it can shape properly
        # the structure will be as  (size , #lookbackdays, chars )
        # each lookback set has a corresponding Y value, this is in order to handle the offest of looking back

        # X_train_lookback, Y_train_ready = LookbackTrain(X_train, Y_train, numChars , lookback_days)

        X_train_lookback = []
        Y_train_ready = []
        # print(training_set_scaled[54-54:54, 1])

        for i in range(lookback_days, len(X_train)):  ## *************double check this len statement
            X_train_lookback.append(X_train[i - lookback_days:i])
            Y_train_ready.append(Y_train[i, 0])

        X_train_lookback, Y_train_ready = np.array(X_train_lookback), np.array(Y_train_ready)
        # print(X_train_lookback)
        # Reshaping X_train for efficient modeling
        X_train_lookback = np.reshape(X_train_lookback,
                                      (X_train_lookback.shape[0], X_train_lookback.shape[1], numChars))

        # *************************************************************************************************************

        # print(X_train_lookback)
        # print(y_train)
        # print(numChars)   #verification step to prove the array dimensions
        print("xtrain shape", X_train_lookback.shape)  # verification step to prove the array dimensions
        print("Y_train_ready shape", Y_train_ready.shape)

        # **************************************************************************************************************
        # This section handles building the model and then testing it out.

        Model1 = GRUModelBuild(X_train_lookback, Y_train_ready, numChars,
                               lookback_days)  # call the builder function above
        # Model1.summary()  ## This will print the structure of the GRU that has been built

        Model1.fit(X_train_lookback, Y_train_ready, validation_split=0.1, epochs=30)  # ,batch_size=1
        # train the model using the datasets, validation split will take a random 10% of the data out of training and use it
        # to test the model for validation, the result will show if genrealization is occuring properly
        # train over 15 iteraitions (epochs)

        GRU_predicted_stock_price = Model1.predict(X_test_lookback)
        # Predict the prices using the test set with lookback applied to it

        # Visualizing the results for GRU
        GRU_predicted_stock_price = Descaler(GRU_predicted_stock_price, Factor)
        Y_test_ready = Descaler(Y_test_ready, Factor)
        ## Descale the predictions, Y_test_ready was used for validation of descaling.

        Y_test = Descaler(Y_test, Factor)

        plot_predictions(Y_test, GRU_predicted_stock_price, lookback_days)
        # Plot the predictions, the final attribute of this function allows for shifting the plots to see what occured before
        # the displayed prediction

        # Evaluating GRU
        tempHold_RMSE = return_rmse(Y_test_ready, GRU_predicted_stock_price)  # call rmse to give error on data set
        tempHold_DirErr = Directional_error(Y_test_ready, GRU_predicted_stock_price)

        if (useMetric == "RMSE"):
            if ((tempHold_RMSE < curr_BestResultRMSE) or (
                    (tempHold_DirErr > curr_BestResultDirErr) and (tempHold_RMSE == curr_BestResultRMSE))):
                curr_BestResultRMSE = tempHold_RMSE
                curr_BestResultDirErr = tempHold_DirErr
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
            else:
                print("This model was discarded")

        if (useMetric == "DirErr"):
            if ((tempHold_DirErr > curr_BestResultDirErr) or (
                    (tempHold_DirErr == curr_BestResultDirErr) and (tempHold_RMSE < curr_BestResultRMSE))):
                curr_BestResultDirErr = tempHold_DirErr
                curr_BestResultRMSE = tempHold_RMSE
                attrib_used_metric = characteristics[options[j]]
                prGreen("This Model has been saved")
                Model1.save(model_SaveLocation)
            else:
                print("This model was discarded")

        # CSV_Header = ['Ticker', 'Attributes', 'DirErr', 'RMSE']
        training_Data_forCSV.append(
            {'Ticker': ticker_File.strip('.csv'), 'Attributes': characteristics[options[j]], 'DirErr': tempHold_DirErr,
             'RMSE': tempHold_RMSE})
        print("Attributes:", characteristics[options[j]])  # print what characteristics were used

        # print(Model1.summary()) #this will print out the structure of the GRU

        keras.backend.clear_session()  # delete the GRU and clear all weight data, this is essential to train a new model after
        print("Model Cleared")

    # This section will then load in the best found model found and use it to make predictions for the user******************************************************
    # As well as make a call to create the csv file with all the stock data
    
    makeModelDataCSVFile(csv_file_name)
    end = datetime.now()
    filed="L.csv"
    b=open(filed,'w')
    writer=csv.writer(b)
    writer.writerow(start)
    writer.writerow(end)
    b.close()
    reconstruct_Model = keras.models.load_model(model_SaveLocation)

# In[ ]:


