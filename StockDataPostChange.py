#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing dependencies
import datetime as dt #for working with dates
import pandas as pd #for working with large lists and large csv files
from pandas_datareader import data as pdr #for main stock data
import csv #for working with csv files
import requests #for html parsing
import bs4 #for html parsing
from bs4 import BeautifulSoup  #for html parsing
import pytrends #for google trends
from pytrends.request import TrendReq #for google trends
import time

#tweet scraper libs
import snscrape.modules.twitter as sntwitter
from dateutil.relativedelta import relativedelta
from datetime import date
from statistics import mean
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string

import os

#enter start date, limited by whatever stock has the "least oldest" stock data available

#startDate = input("Enter START date in format 'YYYY-MM-DD': ") #get start date from user
#instead of asking for start date set start date to 2001-01-01
startDate = dt.datetime.strptime("2000-01-01", '%Y-%m-%d') #make start date a "datetime" object in order to work with it

#set end date to today 
endDate = dt.datetime.now() #set end date to today ("datetime" object)
#prompt user for number of monthly tweets to collect.
num_of_tweets = "uninitialized"
print("How many tweets do you want to pull per month? (number must be an integer between 1 and 500 inclusive):") #note that it takes about 15 total minutes per stock at 100 tweets per month
while (num_of_tweets == "uninitialized"):
    try:
        num_of_tweets = int(input()) #raises error if float is input because it's seen as a string first
        if (num_of_tweets < 1 or num_of_tweets > 500):
            raise ValueError("input not within range")
    except Exception as ex:
        #print(ex)
        print("Number invalid, please enter an integer between 1 and 500:")

#prompt user for stock tickers
userInput = input("Enter stock ticker of interest (or enter q to quit): ")
userInput = userInput.upper() #make stock ticker all capital letters

stockTickerArray = [] #empty list of stock tickers
searchWordArray = [] #empty list of stock tickers

while (userInput != 'Q'):
    try:
        currentData = pdr.DataReader(userInput, 'yahoo', startDate, endDate) #try to get main stock data from yahoo finance
    except:
        print ("Stock ticker invalid please enter another or press 'q' to quit: ")
    else:
        stockTickerArray.append(userInput) # add valid stock ticker to list of valid stick tickers
        searchWord = input("Enter corresponding search word: ")
        searchWordArray.append(searchWord) # add valid search word to list

        currentFileName = userInput + ".csv" #create base file
        with open(currentFileName, 'w', encoding= 'UTF8' ) as f: #this writes the current stock tickers data to a csv file
            currentData.to_csv(currentFileName) #save data retrieved from yahoo finance (dataframe object) to the base csv
            
    userInput = input("Enter stock tickers of interest (or enter q to stop): ") #get next stock ticker from user
    userInput = userInput.upper()

#start_time = time.time()
j = 0
for x in stockTickerArray: #iterate over every stock ticker in array
    stock_ticker = x
    print("Getting" ,stock_ticker, "stock data..." )

    fileName = stock_ticker + ".csv" #current file name is based on the current stock ticker
    
    with open(fileName, 'r', encoding= 'UTF8' ) as f:
            #print(sum(1 for line in f))
            
            
###########################################################################################################################
################################################################EPS########################################################

    #Macrotrends for past quarter eps data
    epsDataList = []
    epsColumn = []
    try:
        URL = "https://www.macrotrends.net/stocks/charts/" + stock_ticker + "/alphabet/eps-earnings-per-share-diluted"
        #get url where eps data is

        page = requests.get(URL) #gets HTML from from site using its URL
        soup = BeautifulSoup(page.content, "html.parser") # this line ensures we use the right parser for HTML
        chart = soup.find_all("td") #find all objects in html with attribute "<td>"


        i = 0
        while i < len(chart): # i indexes in range from 0 - (# of elements -1)
            if chart[i].text.startswith(('$','2')): #POTENTIALLY desireable data (dates and prices)
                if (chart[i].text.isnumeric()): #undesirable data (ex: "2012" and its avg (yearly EPS price))
                    del chart[i+1] #HAVE TO DELETE THE ONE WITH HIGHER INDEX FIRST
                    del chart[(i)]
                else: #desireable data (i.e., quarter dates and quarter prices)
                    i+=1
            else: #once this condition is reached only undesireable data is afterwards due to format of the sites HTMl 
                while i != len(chart):
                    del chart[i]
        #chart has data of eps quarter dates on even indexes (incl. "0") and EPS values on odd values (incl. "1")

        for x in chart:
            if (x.text[0] == "$"):
                epsDataList.append(x.text[1:])
            else:
                epsDataList.append(x.text)
        #epsDataList has eps quarter dates on even indexes (incl. "0") and EPS values on odd values (incl. "1")
        #with most recent data at the beginning

        #find earliest and latest (most current) date
        earliestEpsDate = dt.datetime.now() #set to random date that is forward in time of earliest EPS date
        latestEpsDate = dt.datetime.strptime("1900-01-01", '%Y-%m-%d') #set to random date that is further back in time of latest EPS date

        #actually find correct earliest and latest eps dates
        i = 0
        while i < (len(epsDataList) - 1 ):
            tempDate = dt.datetime.strptime(epsDataList[i], '%Y-%m-%d')
            if (tempDate < earliestEpsDate):
                earliestEpsDate = tempDate
            if tempDate > latestEpsDate:
                latestEpsDate = tempDate        
            i += 2

        #create column to be appended to csv after main data is created
        i = 0
        with open(fileName, 'r', encoding = 'utf-8') as f: #open file
            csv_reader = csv.reader(f, delimiter =",")
            for row in csv_reader: #look at one row at a time
                date = row[0]
                if (date != "Date"): #ignore the header column
                    fileDate = dt.datetime.strptime(date, '%Y-%m-%d') #convert date in file to datetime object       
                    if (fileDate < (earliestEpsDate)): #check if date in the file is before the earliest quarter we have data for
                        epsColumn.append("NaN") 
                    elif (fileDate > latestEpsDate): #check if date in the file is a date that is in the current fiscal quarter
                        epsColumn.append(epsDataList[1])
                    else: #we have data for these dates
                        #WE ARE HERE
                        
                        
                        
                        
                        
                        while i < (len(epsDataList) - 2 ):
                            #end of the list is where the earliest data is stored
                            #format of list is "Date (\n) Price (\n) Date (\n)....(\n) Price"
                            tempDateRecent = dt.datetime.strptime(epsDataList[i], '%Y-%m-%d')
                            tempDatePast = dt.datetime.strptime(epsDataList[i+2], '%Y-%m-%d')
                            if (tempDateRecent >= fileDate >= tempDatePast):
                                epsColumn.append(epsDataList[i+3])
                                if (i != 0):
                                    i = i-2
                                break
                            else:
                                i += 2
    except Exception as ex:
        print("There was an exception (", ex, ") in the EPS module, this attribute will be ignored.")        
        pass
                            
###########################################################################################################################
################################################################INSIDER####################################################
    amt_of_buys = []
    amt_of_sells = []
    amt_traded = []
    
    try:
        URL = "http://openinsider.com/screener?s=" +  stock_ticker +   "&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&" + "fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=" + "9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        page = requests.get(URL) #gets HTML from from site using its URL
        soup = BeautifulSoup(page.content, "html.parser") # this line ensures we use the right parser for HTML
        table = soup.find(class_="tinytable")


        try:
            data = table.find_all("td")
        except:
            with open(fileName, 'r', encoding = 'utf-8') as f: #open file
                csv_reader = csv.reader(f, delimiter =",")
                for row in csv_reader: #look at one row at a time
                    date = row[0]
                    if (date != "Date"): #ignore the header column
                        amt_of_buys.append("NaN")
                        amt_of_sells.append("NaN")
                        amt_traded.append("NaN")

        else:
            trade_dates= [] #(Trade date) index 2
            titles = [] #(Title) index 5
            trade_type = [] #(Trade Type) index 5
            transaction = [] #(Qty) index 8
            amtOwned= [] #(Owned) index 9

            i = 0
            while i < len(data):
                trade_dates.append(data[i+2].text) 
                titles.append(data[i+5].text) 
                trade_type.append(data[i+6].text)
                transaction.append(data[i+8].text) 
                amtOwned.append(data[i+9].text)
                i += 16

            #Make dates in trade date array into date time objects
            i = 0
            while i <len(trade_dates):
                trade_dates[i] = dt.datetime.strptime(trade_dates[i], '%Y-%m-%d')
                i += 1

            #remove comma from numbers
            i = 0
            while i <len(transaction):
                transaction[i] = transaction[i].replace(',','')
                i += 1

            with open(fileName, 'r', encoding = 'utf-8') as f: #open file
                csv_reader = csv.reader(f, delimiter =",")
                for row in csv_reader: #look at one row at a time
                    date = row[0]
                    if (date != "Date"): #ignore the header column
                        fileDate = dt.datetime.strptime(date, '%Y-%m-%d')#convert date in file to datetime object
                        if (fileDate < trade_dates[-1]): #if date in file is less than earliest date
                            amt_of_buys.append("NaN")
                            amt_of_sells.append("NaN")
                            amt_traded.append("NaN")
                        elif (fileDate > trade_dates[0]): #if date in file is more recent than latest date
                            amt_of_buys.append(0)
                            amt_of_sells.append(0)
                            amt_traded.append(0)
                        else:
                            try:
                                index = trade_dates.index(fileDate) #find at what index is the file date in the trade dates list
                            except ValueError: #no reported trades on this day
                                amt_of_buys.append(0)
                                amt_of_sells.append(0)
                                amt_traded.append(0)
                            else:
                                #If the date from csv is found in trade_dates list
                                transactionCounter = 0
                                buyCounter = 0
                                sellCounter = 0
                                #count all buys, sells, and total amount traded
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

    except Exception as ex:
        print("There was an exception (", ex, ") in the IT module, these attributes will be ignored.")        
        pass

    
###########################################################################################################################
############################################################INFLATION######################################################
    
    # base year = str(startDate.year)
    # $1 in (base_year) is roughly equivalent (in purchasing power) to (inflated amount) in (whatever year you are looking at)

    inflationDataList = []
    inflationColumn = []
    
    try: 
        url = "https://www.officialdata.org/us/inflation/" + str(startDate.year) + "?amount=1#buying-power"
        page = requests.get(url) #gets HTML from from site using its URL
        soup = BeautifulSoup(page.content, "html.parser") # this line ensures we use the right parser for HTML\
        table = soup.find(class_ = "regular-data table-striped") #find table with desired data (html class = ...)
        data = table.find_all("tr") #find all elements wrapped in <tr> tag in table



        #get data from chart
        for x in data:
            if x.text[0] != 'Y':
                year = int(x.text[0] + x.text[1] + x.text[2] + x.text[3])
                dollar_value = (x.text[5] + x.text[6] + x.text[7] + x.text[8])
                inflation_rate = (x.text[9] + x.text[10] + x.text[11] + x.text[12] + x.text[13])
                inflationDataList.extend((year, dollar_value, inflation_rate))


        #get data from chart
        #for x in data:
        #    if x.text.splitlines()[1] != "Year":
        #        year = int(x.text.splitlines()[1])
        #        dollar_value = x.text.splitlines()[2]
        #        inflation_rate = x.text.splitlines()[3]
        #        inflationDataList.extend((year, dollar_value, inflation_rate))

          
        i=0
        with open(fileName, 'r', encoding = 'utf-8') as f: #open file
            csv_reader = csv.reader(f, delimiter =",")
            for row in csv_reader: #look at one row at a time
                date = row[0]        
                if (date != "Date"): #ignore the header column
                    fileDate = dt.datetime.strptime(date, '%Y-%m-%d')#convert date in file to datetime object
                    while i *3 < (len(inflationDataList) - 1):
                        if int(fileDate.year) == inflationDataList[3*i]:
                            inflationColumn.append(inflationDataList[(3*i) + 2])
                            break
                        else:
                            i += 1
        
    except Exception as ex:
        print("There was an exception (",ex, ") in the IR module, this attribute will be ignored.")        
        pass
                        
###########################################################################################################################
##################################################GOOGLE TREND#############################################################
    
    gtrendData = []
    try:
        pytrends = TrendReq(hl='en-US', tz=360) #connect to google
        kw_list = [stock_ticker] #list of keywords (doing one at a time) (may  generate 409 error if too many tickers)

        stock_df = pd.read_csv(fileName) 
        trend_start_date = stock_df.iloc[0,0] #get earliest stock data date 
        trend_end_date =stock_df.iloc[-1,0] #get most recent stock data date 
        trend_time_frame = "2004-01-01" + " " + trend_end_date

        pytrends.build_payload(kw_list, cat=0, timeframe= trend_time_frame, geo='', gprop='') #build payload
        data = pytrends.interest_over_time() #put data in df object
        #print(data)

        i = 0
        with open(fileName, 'r', encoding = 'utf-8') as f: #open file
            csv_reader = csv.reader(f, delimiter =",")
            for row in csv_reader: #look at one row at a time
                date = row[0]
                if (date != "Date"): #ignore the header column
                    fileDate = dt.datetime.strptime(date, '%Y-%m-%d') #convert date in file to datetime object       
                    if (fileDate < data.index[0]): #check if date in the file is before the earliest date we have data for
                        gtrendData.append("NaN")
                        #print(fileDate, " < ", data.index[0])
                        
                        
                        
                    elif (fileDate >= data.index[-1]): #check if date in the file is a date that is too recent
                        gtrendData.append(data.iloc[-1,0])
                        #print(fileDate, " > ", data.index[-1])
                        #print("file date = ", fileDate, "trend date = ", data.index[-1])
                    else: #we have data for these dates
                        while i < (len(data)-1):
                            if (data.index[i].month == fileDate.month):
                                #print(fileDate, " = ", data.index[i])
                                #print("data.index[i].month = ", data.index[i].month,"data.iloc[i,0]= ",data.iloc[i,0] )
                                gtrendData.append(data.iloc[i+1,0])
                                break
                            else:
                                i += 1


        #print(gtrendData)
    except Exception as ex:
        print("There was an exception (", ex, ") in the GT module, this attribute will be ignored.")        
        pass
    
###########################################################################################################################
##################################################Sentiment Analysis#######################################################
    sentimentColumn = []
    try:
        exclusion_list = ['listen', 'add', 'swear'] # create a blacklist of words
        analyzer = SentimentIntensityAnalyzer()

        kw = searchWordArray[j]

        #commented out and hardcoded for availability purposes 
        start_date = "2022-01-01" #FIX ME set to 2010-01-01

        #commented out and hardcoded for testing purposes
        #end_date = "2022-10-01" #FIX ME set to todays date
        
        end_date = endDate.strftime("%Y-%m-%d")

        search_settings = (kw +' since:' + start_date +' until:' + end_date)
        #print(search_settings)
        #search_settings = (kw +' since:2010-07-05 until:2011-07-06')


        #start_time = time.time()
        #print("--- %s seconds ---" % (time.time() - start_time))

        attributes_container = [] # Creating list to append tweet data to

        sent_dict= {} # create dictionary with sentiment data

        next_month = str((dt.datetime.strptime(start_date, '%Y-%m-%d') + relativedelta(months=+1)).date())

        while (dt.datetime.strptime(next_month, '%Y-%m-%d') <= dt.datetime.now()):
            #print(start_date)
            search_settings = (kw +' since:' + start_date +' until:' + next_month)
            i = 0 #initialize index variable
            for a,tweet in enumerate(sntwitter.TwitterSearchScraper(search_settings).get_items()):
                if i > num_of_tweets: #gets 100 different datapoints
                    break
                compound_score = analyzer.polarity_scores(tweet.content)['compound'] # -1(most extreme negative) and +1 (most extreme positive)
                words = tweet.content.lower() # lower case all words
                words = words.translate(str.maketrans('', '', string.punctuation)) #remove punctuation from string
                words = words.split() #split sentence by spaces
                if (any([x in words for x in exclusion_list]) or compound_score == 0): #check if any words in the blacklist is in the sentence/tweet
                    #print("found")
                    i -= 1
                else: #no non-permitted words in tweet so we will use them
                    attributes_container.append([tweet.date, tweet.content, compound_score])


                i += 1

            # if we are here then i > 100 so we have collected 100 tweets

            # Creating a dataframe to load the list
            tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Tweet", "Compound Score"])
            #finding avg compound score of tweets gathered from previous month
            for (column_name, column_data) in tweets_df.iteritems():
                if column_name == "Compound Score" :
                    avg = mean(column_data.values)
                    #print(round(avg,3)) #round to three decimal places
                    sent_dict[start_date[:7]] = round(avg,3) #set sentiment for each month searched
            #increasing search parameters by one month
            start_date = str((dt.datetime.strptime(start_date, '%Y-%m-%d') + relativedelta(months=+1)).date())
            next_month = str((dt.datetime.strptime(next_month, '%Y-%m-%d') + relativedelta(months=+1)).date())
            i = 0

        #print(sent_dict)

        #variable name with stock ticker needs to be implemented
        with open(fileName, 'r', encoding= 'UTF8' ) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                date = row[0]
                if (date != "Date"):
                    date_in_file = row[0][:7]
                    #print(row[0][:7])
                    if (date_in_file in sent_dict):
                        #latest_month_in_file = date_in_file # not needed?
                        sentimentColumn.append(sent_dict.get(date_in_file))
                        #print("Date in file found in dictionary!")
                    else:
                        sentimentColumn.append("NaN")
                else:
                    pass #do nothing

    except Exception as ex:
        print("There was an exception (", ex, ") in the SA module, this attribute will be ignored.")        
        pass
    
###########################################################################################################################
#################################################APPEND EVERYTHING#########################################################
    
    df = pd.read_csv(fileName)
    
    if len(amt_of_buys) != 0:
        #print(len(amt_of_buys))
        buys_column = pd.DataFrame({'Buys': amt_of_buys})
        df = df.merge(buys_column, left_index = True, right_index = True)
    
    if len(amt_of_sells) != 0:
        #print(len(amt_of_sells))
        sells_column = pd.DataFrame({'Sells': amt_of_sells})
        df = df.merge(sells_column, left_index = True, right_index = True)

    if len(amt_traded) != 0:
        #print(len(amt_traded))
        traded_column = pd.DataFrame({'Traded': amt_traded})
        df = df.merge(traded_column, left_index = True, right_index = True)

    if len(epsColumn) != 0:
        #print(len(epsColumn))
        eps_column = pd.DataFrame({'EPS': epsColumn})
        df = df.merge(eps_column, left_index = True, right_index = True)

    if len(inflationColumn) != 0:
        #print(len(inflationColumn))
        inflation_column = pd.DataFrame({'IR': inflationColumn})
        df = df.merge(inflation_column, left_index = True, right_index = True)
   
    if len(gtrendData) != 0:
        #print(len(gtrendData))
        gt_column = pd.DataFrame({'GT': gtrendData})
        df = df.merge(gt_column, left_index = True, right_index = True)

    if len(sentimentColumn) != 0:
        #print(len(sentimentColumn))
        sentiment_Column = pd.DataFrame({'SA': sentimentColumn})
        df = df.merge(sentiment_Column, left_index = True, right_index = True)
    
    df.to_csv(fileName, index = False)
    print(stock_ticker, "stock data gathering complete! :^) ")
    
    j += 1

############################################################################################################################
print("Script complete! :D ")
#print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:



