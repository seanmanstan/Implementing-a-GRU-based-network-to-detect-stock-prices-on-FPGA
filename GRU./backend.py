
from time import sleep

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
import os


def Data_Scraper(userInput, startDate, endDate):

    try:
        #userInput = input("Enter stock tickers of interest, one at a time, followed by the 'Enter' key (enter q to stop): ")
        userInput = userInput.upper() #make stock ticker all capital letters
        #enter start date, limited by whatever stock has the "least oldest" stock data available
        #startDate = input("Enter START date in format 'YYYY-MM-DD': ") #get start date from user
        #startDate = dt.datetime.strptime(startDate, '%Y-%m-%d') #make start date a "datetime" object in order to work with it
        #enter end date
        #endDate = dt.datetime.now() #set end date to today ("datetime" object)

        #prompt user for stock tickers

        #***********************************************************************************************
        #this is where it looks for the ticker on yahoo and tells if it is invalid.
        #***********************************************************************************************

        stockTickerArray = [] #empty list of stock tickers
        #while (userInput != 'Q'):
        try:
            currentData = pdr.DataReader(userInput, 'yahoo', startDate, endDate) #try to get main stock data from yahoo finance
        except:
            print ("Stock ticker invalid please enter another or press 'q' to stop: ")
        else:
            stockTickerArray.append(userInput) # add valid stock ticker to list of valid stick tickers
            currentFileName = userInput + ".csv" #create base file
            with open(currentFileName, 'w', encoding= 'UTF8' ) as f: #this writes the current stock tickers data to a csv file
                currentData.to_csv(currentFileName) #save data retriebed from yahoo finance (dataframe object) to the base csv

        #userInput = input() #get next stock ticker from user
        userInput = userInput.upper()
        #***********************************************************************************************



        for x in stockTickerArray: #iterate over every stock ticker in array
            stock_ticker = x
            print("Getting" ,stock_ticker, "stock data..." )

            fileName = stock_ticker + ".csv" #current file name is based on the current stock ticker

        ###########################################################################################################################
        ################################################################EPS########################################################

            #Macrotrends for past quarter eps data


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

            epsDataList = []
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
            epsColumn = []
            i = 0
            with open(fileName, 'r', encoding = 'utf-8') as f: #open file
                csv_reader = csv.reader(f, delimiter =",")
                for row in csv_reader: #look at one row at a time
                    date = row[0]
                    if (date != "Date"): #ignore the header column
                        fileDate = dt.datetime.strptime(date, '%Y-%m-%d') #convert date in file to datetime object
                        if (fileDate < (earliestEpsDate - dt.timedelta(weeks=13))): #check if date in the file is before the earliest quarter we have data for
                            epsColumn.append("NaN")
                        elif (fileDate > latestEpsDate): #check if date in the file is a date that is in the current fiscal quarter
                            epsColumn.append("NaN")
                        else: #we have data for these dates
                            while i < (len(epsDataList) - 2 ):
                                if (fileDate < earliestEpsDate):
                                    epsColumn.append(epsDataList[-1]) #end of the list is where the earliest data is stored
                                    break
                                #format of list is "Date (\n) Price (\n) Date (\n)....(\n) Price"
                                tempDateRecent = dt.datetime.strptime(epsDataList[i], '%Y-%m-%d')
                                tempDatePast = dt.datetime.strptime(epsDataList[i+2], '%Y-%m-%d')
                                if (tempDateRecent >= fileDate >= tempDatePast):
                                    epsColumn.append(epsDataList[i+1])
                                    if (i != 0):
                                        i = i-2
                                    break
                                else:
                                    i += 2

        ###########################################################################################################################
        ################################################################INSIDER####################################################

            URL = "http://openinsider.com/screener?s=" +            stock_ticker +            "&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&" +                        "fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=" +            "9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
            page = requests.get(URL) #gets HTML from from site using its URL
            soup = BeautifulSoup(page.content, "html.parser") # this line ensures we use the right parser for HTML
            table = soup.find(class_="tinytable")

            amt_of_buys = []
            amt_of_sells = []
            amt_traded = []
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
                                amt_of_buys.append("NaN")
                                amt_of_sells.append("NaN")
                                amt_traded.append("NaN")
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

        ###########################################################################################################################
        ############################################################INFLATION######################################################

            # base year = str(startDate.year)
            # $1 in (base_year) is roughly equivalent (in purchasing power) to (inflated amount) in (whatever year you are looking at)

            url = "https://www.officialdata.org/us/inflation/" + str(startDate.year) + "?amount=1#buying-power"
            page = requests.get(url) #gets HTML from from site using its URL
            soup = BeautifulSoup(page.content, "html.parser") # this line ensures we use the right parser for HTML\
            table = soup.find(class_ = "regular-data table-striped") #find table with desired data (html class = ...)
            data = table.find_all("tr") #find all elements wrapped in <tr> tag in table

            #for x in data:
            #    print(x)
            #    print(x.text[0])
            #    print(x.text.splitlines())
            #    print(x.text.splitlines()[0][0])

            inflationDataList = []


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

            inflationColumn = []
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

        ###########################################################################################################################
        ##################################################GOOGLE TREND#############################################################

            pytrends = TrendReq(hl='en-US', tz=360) #connect to google
            kw_list = [stock_ticker] #list of keywords (doing one at a time) (may  generate 409 error if too many tickers)

            stock_df = pd.read_csv(fileName)
            trend_start_date = stock_df.iloc[0,0] #get earliest stock data date
            trend_end_date =stock_df.iloc[-1,0] #get most recent stock data date
            trend_time_frame = "2004-01-01" + " " + trend_end_date

            pytrends.build_payload(kw_list, cat=0, timeframe= trend_time_frame, geo='', gprop='') #build payload
            data = pytrends.interest_over_time() #put data in df object

            i = 0
            gtrendData = []
            with open(fileName, 'r', encoding = 'utf-8') as f: #open file
                csv_reader = csv.reader(f, delimiter =",")
                for row in csv_reader: #look at one row at a time
                    date = row[0]
                    if (date != "Date"): #ignore the header column
                        fileDate = dt.datetime.strptime(date, '%Y-%m-%d') #convert date in file to datetime object
                        if (fileDate < data.index[0]): #check if date in the file is before the earliest date we have data for
                            gtrendData.append("NaN")
                        elif (fileDate > data.index[-1]): #check if date in the file is a date that is too recent
                            gtrendData.append("NaN")
                            #print("file date = ", fileDate, "trend date = ", data.index[-1])
                        else: #we have data for these dates
                            while i < (len(data)):
                                if (data.index[i].month == fileDate.month):
                                    gtrendData.append(data.iloc[i,0])
                                    break
                                else:
                                    i += 1



            #print(gtrendData)

        ###########################################################################################################################
        #################################################APPEND EVERYTHING#########################################################

            df = pd.read_csv(fileName)
            buys_column = pd.DataFrame({'Buys': amt_of_buys})
            sells_column = pd.DataFrame({'Sells': amt_of_sells})
            traded_column = pd.DataFrame({'Traded': amt_traded})
            eps_column = pd.DataFrame({'EPS': epsColumn})
            inflation_column = pd.DataFrame({'IR': inflationColumn})
            gt_column = pd.DataFrame({'GT': gtrendData})
            df = df.merge(buys_column, left_index = True, right_index = True)
            df = df.merge(sells_column, left_index = True, right_index = True)
            df = df.merge(traded_column, left_index = True, right_index = True)
            df = df.merge(eps_column, left_index = True, right_index = True)
            df = df.merge(inflation_column, left_index = True, right_index = True)
            df = df.merge(gt_column, left_index = True, right_index = True)
            df.to_csv(fileName, index = False)
            print(stock_ticker, "stock data gathering complete! :^) ")

        ############################################################################################################################
        print("Script complete! :D ")
        return
    except:
        print("Something Failed inside the datascraper script")
        return
    #os.system('python version_2.18.3.py ' + stock_ticker)





