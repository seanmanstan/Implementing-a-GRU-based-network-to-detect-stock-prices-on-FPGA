#800000  Hex for maroon
import math
from tkinter import *
import os
import numpy as np
import threading
from time import sleep
#import csv
from PIL import ImageTk, Image
#from GUI_Functs import makeDataYearPlot, packPlotImage
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

plt.style.use('fivethirtyeight')
import pandas as pd
from tkinter import messagebox
import datetime as dt

from backend import *
#from backend import *

## change this to backend for the known working version

from PrepdictionDataPrep import *


#*************************************************************
#Definitions section
VolumeDef ="Volume: The volume of trade refers to the total number of shares or contracts exchanged between buyers and sellers of a security during trading hours on a given day. The volume of trade is a measure of the market\'s activity \nand liquidity during a set period of time. Higher trading volumes are considered more positive than lower trading volumes because they mean more liquidity and better order execution."
InsiderTradeDef = "Insider Trading: The trading of a public company's stock or other securities from employees within the company based on material, that is nonpublic information about the company.\n" \
                  "Our data shows legally reported insider trades through the SEC."
EPSDef = "Earnings per share (EPS): is a company's net profit divided by the number of common shares it has outstanding. EPS indicates how much money a company makes for each share of its stock and is a widely \nused metric for estimating corporate value."

#print(Volume)

stockName = ""
working_Frame = 0
optionsWindowOpen = 0
top = 0
loadText = "Collecting your stock data"
loadText1 = "Collecting your stock data."
loadText2 = "Collecting your stock data.."
loadText3 = "Collecting your stock data..."
loadTextDone = "Complete"
RecentinsiderTrade = ""


Price1, Price2 = 0, 0
dodPercIncr = ""

EPS1, EPS2 = 0, 0
EPSStatement = ""

permanentStartDate = '2000-01-01'
permanentStartDate = dt.datetime.strptime(permanentStartDate, '%Y-%m-%d') #make start date a "datetime" object in order to work with it
predictedValuesList = []

ExceptionsList = [None]*1

#************************************************************

class GUI(Tk):

    def __init__(self):
        super().__init__()

        self.title("GBSPS Window")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        #root.configure(background="darkgrey")
        self.geometry("900x525")
        self.resizable(width=False, height=False)

        self.background_image = ImageTk.PhotoImage(Image.open("stolen_stockImage.jpg"))
        self.background_label = Label(self, image=self.background_image)
        #background_image = ImageTk.PhotoImage(file="stolen_stockImage.jpg")
        #self.background_label = Label(self, image=background_image)

        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.greeting_Frame = Label(master = self,text="Howdy Trader!\nEnter A Stock Ticker", font=('calibre',14,'bold'), fg="white", bg = "#800000")

        self.greeting_Frame.pack(fill=BOTH)

        self.ticker = StringVar()

        self.StockentryFrame = Label(master = self, bg="darkgrey")
        self.StockentryFrame.pack()

        self.name_entry = Entry(master = self.StockentryFrame, textvariable = self.ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")

        self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")



    def MakePredictionPlot(self):
        global predictedValuesList, top, stockName

        if(not (os.path.exists("saved_GRU_"+stockName+"_Model.h5"))):  ## Shows up in this format saved_GRU_F_Model.h5
            print("No trained model exists in the directory")
            self.RunTheGRU()
            #RunGRUmodelTraining()
            return

        predictedValuesList = []
        predictedValuesList = MakePrediction(stockName)
        # threading.Thread(target=lambda:MakePrediction(stockName)).start()
        #sleep(5)

        file_name_used = stockName + '.csv'

        dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])
        dataset = dataset['2022':'2022']
        dataset = dataset.tail(5)

        dataset['Day'] = dataset.index.dayofyear
        dataset['DateUse'] = pd.to_datetime(dataset['Day'], format='%j').dt.strftime('%m-%d')

        figure, (ax1, ax2) = plt.subplots(2, 1)
        #figure.autolayout :True

        days = dataset['Day'].tolist()

        for i in range(0, len(days)):
            days[i] = int(days[i])
        days.append(days[-1]+1)
        #print(days)

        #ax1.plot(dataset['Day'], dataset['Close'], linewidth=2.0)
        ax1.plot(days[:5], dataset['Close'], linewidth=2.0)

        ax1.plot(days[1:6], predictedValuesList[-5:], linewidth=2.0)

        ax1.set_title(stockName + " Stock Values")
        #ax1.set_xlabel('Day of year')
        ax1.set_ylabel('Value at Close USD:$')

        ax2.bar(dataset['Day'], dataset['Volume'])
        ax2.set_xlabel('Day of year')
        ax2.set_ylabel('Volume Traded')

        #ax1.xticks(np.arange(0, len(x)+1, 5))
        dif_ofMaxMin = dataset['Close'].max() - dataset['Close'].min()
        if(dif_ofMaxMin < 5):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 0.5))
        elif((dif_ofMaxMin > 5) and (dif_ofMaxMin < 7)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 1))
        elif((dif_ofMaxMin > 7) and (dif_ofMaxMin < 10)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 1.55))
        elif((dif_ofMaxMin > 10) and (dif_ofMaxMin < 15)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 2.5))
        elif((dif_ofMaxMin > 15) and (dif_ofMaxMin < 20)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 5))
        else:
            None

        plt.tight_layout(h_pad=2)
        plt.savefig(stockName + 'prediction.png') #, bbox_inches="tight"
        plt.cla()
        plt.close()
        top = Toplevel(root)
        top.geometry("300x100")
        top.title("Price Prediction")

        Label(top, text="The prediction for "+stockName+ " is $"+str(round(predictedValuesList[-1][0],2)), font=('Arial 12')).place(x=10, y=10)

##Bug inside this KillTheGUI function that causes it not to show the countdown... but still functions
##Bug inside this KillTheGUI function that causes it not to show the countdown... but still functions
    def KillTheGUI(self):
        global BranchtoGRU
        BranchtoGRU = 1
        #self.greeting_Frame.config(text="The GUI will be terminated in: 3")

        self.GRUOptionsFrame.grid_slaves(row=1, column=0)[0].destroy()
        self.GRUOptionsFrame.grid_slaves(row=1, column=1)[0].destroy()
        self.GRUOptionsFrame.pack_forget()
        self.GRUOptionsFrame.destroy()

        self.greeting_Frame.config(text="The GUI will be terminated in: 3")
        sleep(1)
        self.greeting_Frame.config(text="The GUI will be terminated in: 2")
        sleep(1)
        self.greeting_Frame.config(text="The GUI will be terminated in: 1")
        sleep(1)
        root.destroy()


    def RunTheGRU(self):
        global stockName
        self.greeting_Frame.config(text="Alert: No Trained Model Found")

        self.OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
        self.OptionFrame.grid_slaves(row=1, column=0)[0].destroy()
        self.OptionFrame.grid_slaves(row=0, column=1)[0].destroy()
        self.OptionFrame.pack_forget()
        self.OptionFrame.destroy()

        self.GRUOptionsFrame = Label(master=root, bg="darkgrey")
        self.GRUOptionsFrame.pack()

        self.GRUNotificationFrame = Label(master=self.GRUOptionsFrame, bg='darkgrey', text = "There is no trained model in the directory for "+stockName+".\nIn " +
        "order to make a prediction a model must be trained. To save device resources the GUI will be closed for model training.\n"+
        "Once the GUI is closed this action cannot be undone. Training can take up to 8 hours to complete, the GUI will reopen when completed.\n"+
        "Would you like to proceed with training a model?"
        , font=('calibre',12,'normal')).grid(row=0,column=0,columnspan=2)


        self.NoGRU_btn = Button(master = self.GRUOptionsFrame, text = 'No', font=('calibre',12,'normal'), width = 15, command=self.BacktoOptsFromRunTheGRU).grid(row=1,column=1, sticky= "NS")
        self.YesGRU_btn = Button(master = self.GRUOptionsFrame, text='Yes',font=('calibre',12,'normal'), width = 15, command=self.KillTheGUI).grid(row=1,column=0, sticky= "NS")
        #root.destroy()

    def BacktoOptsFromRunTheGRU(self):
        self.GRUOptionsFrame.grid_slaves(row=1, column=0)[0].destroy()
        self.GRUOptionsFrame.grid_slaves(row=1, column=1)[0].destroy()
        self.GRUOptionsFrame.pack_forget()
        self.GRUOptionsFrame.destroy()
        self.build_optionsMenu()


    def returnToOpts(self):
        global optionsWindowOpen, top
        self.OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
        self.OptionFrame.grid_slaves(row=2, column=0)[0].destroy() #destory year button
        self.OptionFrame.grid_slaves(row=2, column=1)[0].destroy() #destroy week button

        self.OptionFrame.grid_slaves(row=0, column=2)[0].destroy() #destroy the insidertrading frame
        self.OptionFrame.grid_slaves(row=1, column=2)[0].destroy() #destroy the EPS frame
        #self.OptionFrame.grid_slaves(row=2, column=2)[0].destroy() #destroy the DOD frame


        self.OptionFrame.config(image="")
        self.OptionFrame.pack_forget()
        self.labelimage.pack_forget()
        if optionsWindowOpen:
            optionsWindowOpen = 0
            top.destroy()
        #Tk.update(root)
        self.build_optionsMenu()



    def HelpExplainData(self):
        global VolumeDef, InsiderTradeDef, optionsWindowOpen, top
        if(optionsWindowOpen):
            return
        optionsWindowOpen = 1
        top = Toplevel(root)
        top.geometry("750x400")
        top.title("Help Window")

        #Text(top, text= VolumeDef, font=('Arial 12')).place(x=10, y=10)

        #.grid(row=1,column=0, sticky= "NS")
        Label(top, text= VolumeDef, font=('Arial 12'), anchor="w", pady = 10).grid(row=0,column=0, sticky= "NW") #.place(x=10, y=10)
        Label(top, text= InsiderTradeDef, font=('Arial 12'), anchor="w", pady = 10).grid(row=2,column=0, sticky= "NW") #.place(x=10, y=110)
        Label(top, text= EPSDef, font=('Arial 12'), anchor="w", pady = 10).grid(row=4,column=0, sticky= "NW") #.place(x=10, y=210)

        #GrabOtherData()
        #Label(top, text= RecentinsiderTrade, font=('Arial 12')).place(x=10, y=200)


    def packPlotImage(self, recalled):
        #global stockName, OptionFrame, labelimage, greeting_Frame
        if(recalled == 0):
            self.greeting_Frame.config(text="Stock Data for ticker: "+stockName)
            self.OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
            self.OptionFrame.grid_slaves(row=0, column=1)[0].destroy()
            self.OptionFrame.grid_slaves(row=1, column=0)[0].destroy()
            self.GrabOtherData()
        else:
            self.OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
            self.OptionFrame.grid_slaves(row=2, column=0)[0].destroy()  #destroy year button
            self.OptionFrame.grid_slaves(row=2, column=1)[0].destroy()  #destory week button

            self.OptionFrame.grid_slaves(row=0, column=2)[0].destroy()
            self.OptionFrame.grid_slaves(row=1, column=2)[0].destroy()
            self.labelimage.pack_forget()

        percentage = 0.8
        self.image = Image.open(stockName+".png")
        width, height = self.image.size
        resized_dimensions = (int(width * percentage), int(height * percentage))
        resized = self.image.resize(resized_dimensions)
        #resize_image = image.resize((480, 375))
        photo = ImageTk.PhotoImage(resized)

        self.PlotYear_btn = Button(self.OptionFrame, text='Year', command=lambda: self.makeDataYearPlot(1), height = 1, width = 15).grid(row=2,column=0, sticky= "NSE",columnspan=1)
        self.PlotWeek_btn = Button(self.OptionFrame, text='Week', command=lambda: self.makeDataWeekPlot(1), height = 1, width = 15).grid(row=2,column=1, sticky= "NSW",columnspan=1)

        self.returnToOptions = Button(self.OptionFrame, text='Return to Options', command=self.returnToOpts, width = 15).grid(row=3,column=2, sticky= "NSE")
        self.HelpExplain = Button(self.OptionFrame, text='Help/Definitions', command=self.HelpExplainData, width = 15).grid(row=3,column=3, sticky= "NSW")

        self.insiderData = Label(self.OptionFrame, text=RecentinsiderTrade,fg= 'white', bg ='#800000', font='Calibri 12', wraplength=300).grid(row=0,column=2, sticky= "NWE",pady=2, columnspan=2)
        self.EPSDataLabel = Label(self.OptionFrame, text=EPSStatement + "\n\n"+ dodPercIncr, bg ='#800000', fg= 'white', font='Calibri 12', wraplength=300).grid(row=1,column=2, sticky= "NWE", columnspan=2)

        #self.DODDataLabel = Label(self.OptionFrame, text=dodPercIncr, bg ='#800000', fg= 'white', font='Calibri 12', wraplength=300).grid(row=2,column=2, sticky= "NWE", columnspan=2)

        self.labelimage = Label(master=self.OptionFrame, image = photo)
        self.labelimage.image = photo
        self.labelimage.grid(row=0, column=0, columnspan=2, rowspan=2, pady=2, padx=2)
        #Label(top, text= "Hello World!", font=('Mistral 18 bold')).place(x=150,y=80)


    def GrabOtherData(self):
        global stockName, RecentinsiderTrade, EPSStatement, dodPercIncr
        file_name_used = stockName + '.csv'
        dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])
        dataset = dataset['2021':'2022']
        dataset.index = pd.to_datetime(dataset.index)
        TemplengthOfdata = len(dataset['Traded'])
        for x in range(TemplengthOfdata):
            temp = dataset['Traded'][TemplengthOfdata-x-1]
            if((temp != 0) and (not(pd.isna(temp)))):
                if(temp < 0):
                    RecentinsiderTrade = "The most recent insider trade was a sale of:\n" + str(temp) + " shares on " + str(dataset.index[TemplengthOfdata-x-1])[:10]
                else:
                    RecentinsiderTrade = "The most recent insider trade was a buy of:\n" + str(temp)+ " shares on " + str(dataset.index[TemplengthOfdata-x-1])[:10]
                break
            RecentinsiderTrade = "No Recent Insider Trades"

        TemplengthOfdata = len(dataset['EPS'])
        Temp2InstancesFound = 0
        EPS1, EPS2 = 0, 0
        for x in range(TemplengthOfdata):
            temp = dataset['EPS'][TemplengthOfdata-x-1]
            if(not(pd.isna(temp))):
                if(Temp2InstancesFound == 0):
                    EPS1 = dataset['EPS'][TemplengthOfdata-x-1]
                    Temp2InstancesFound +=1
                elif((Temp2InstancesFound == 1) and (dataset['EPS'][TemplengthOfdata-x-1]!=EPS1) ):
                    EPS2 = dataset['EPS'][TemplengthOfdata-x-1]
                    break
                else:
                    continue
        temp = round(((EPS1 - EPS2) / EPS1),2)
        if(temp>0):
            EPSStatement = "There was an increase of " + str(temp)+"% \nin the past two EPS reports."
        else:
            EPSStatement = "There was an decrease of " + str(temp)+"% \nin the past two EPS reports."

        TemplengthOfdata = len(dataset['Close'])
        Price1, Price2 = 0, 0

        Price1 = dataset['Close'][TemplengthOfdata-1]
        Price2 = dataset['Close'][TemplengthOfdata-2]
        temp = round(((Price1 - Price2)/Price2)*100, 2)

        if(temp>0):
            dodPercIncr = "There was a DOD increase of " + str(temp)+"% \nin the past two trading days."
        else:
            dodPercIncr = "There was a DOD decrease of " + str(temp)+"% \nin the past two trading days."






    def makeDataWeekPlot(self,PlotRecall):

        global stockName
        #ticker_File = "AAPL"
        file_name_used = stockName + '.csv'
        #print(file_name_used + ": this file was called")

        dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])
        dataset = dataset['2022':'2022']
        dataset = dataset.tail(5)

        dataset['Day'] = dataset.index.dayofyear
        dataset['DateUse'] = pd.to_datetime(dataset['Day'], format='%j').dt.strftime('%m-%d')

        figure, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(dataset['Day'], dataset['Close'], linewidth=2.0)
        ax1.set_title(stockName + " Stock Values")
        ax1.set_ylabel('Value at Close USD:$')
        ax2.bar(dataset['Day'], dataset['Volume'])
        ax2.set_xlabel('Day of year')
        ax2.set_ylabel('Volume Traded')

        #ax1.xticks(np.arange(0, len(x)+1, 5))
        dif_ofMaxMin = dataset['Close'].max() - dataset['Close'].min()

        if(dif_ofMaxMin < 2):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 0.25))
        if((dif_ofMaxMin >= 2) and (dif_ofMaxMin < 3)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 0.35))
        elif((dif_ofMaxMin >= 3) and (dif_ofMaxMin < 5)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 0.50))
        elif((dif_ofMaxMin >= 5) and (dif_ofMaxMin < 7)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 1))
        elif((dif_ofMaxMin >= 7) and (dif_ofMaxMin < 10)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 1.55))
        elif((dif_ofMaxMin >= 10) and (dif_ofMaxMin < 15)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 2.5))
        elif((dif_ofMaxMin >= 15) and (dif_ofMaxMin < 20)):
            ax1.yaxis.set_ticks(np.arange(math.floor(dataset['Close'].min()), math.ceil(dataset['Close'].max()), 5))
        else:
            None

        plt.tight_layout(h_pad=2)
        plt.savefig(stockName + '.png') #, bbox_inches="tight"
        plt.cla()
        plt.close()
        if(PlotRecall):
            self.packPlotImage(1)



    def makeDataYearPlot(self, PlotRecall):
        global stockName
        #ticker_File = "AAPL"
        file_name_used = stockName + '.csv'
        #print(file_name_used + ": this file was called")

        dataset = pd.read_csv(file_name_used, index_col='Date', parse_dates=['Date'])
        dataset = dataset['2022':'2022']

        dataset['Day'] = dataset.index.dayofyear
        dataset['DateUse'] = pd.to_datetime(dataset['Day'], format='%j').dt.strftime('%m-%d')

        figure, (ax1, ax2) = plt.subplots(2, 1)
        #figure.autolayout :True
        ax1.plot(dataset['Day'], dataset['Close'], linewidth=2.0)
        ax1.set_title(stockName + " Stock Values")
        ax1.set_ylabel('Value at Close USD:$')

        ax2.bar(dataset['Day'], dataset['Volume'])
        ax2.set_xlabel('Day of year')
        ax2.set_ylabel('Volume Traded')

        plt.tight_layout(h_pad=2)
        plt.savefig(stockName + '.png') #, bbox_inches="tight"
        plt.cla()
        plt.close()
        if(PlotRecall):
            self.packPlotImage(1)
        else:
            self.packPlotImage(0)


    def rebuildHome(self):
        #global StockentryFrame
        #global OptionFrame
        self.greeting_Frame.config(text="Enter A Stock Ticker")
        self.OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
        self.OptionFrame.grid_slaves(row=1, column=0)[0].destroy()
        self.OptionFrame.grid_slaves(row=0, column=1)[0].destroy()
        self.OptionFrame.pack_forget()
        self.StockentryFrame = Label(master=root, bg="darkgrey")
        self.StockentryFrame.pack()
        self.name_entry = Entry(master = self.StockentryFrame, textvariable = ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
        self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")


    def build_optionsMenu(self):
        #global OptionFrame
        self.greeting_Frame.config(text="Would you like to view the stock information or train a prediction model?")
        self.OptionFrame = Label(master = root, fg="white", bg = "#800000")
        self.OptionFrame.pack()
        self.view_stockInfo_btn = Button(self.OptionFrame ,font='Bold', text='View Information', command=lambda: self.makeDataYearPlot(0),height = 2, width = 15).grid(row=0,column=0, sticky= "NS")
        self.train_model_btn = Button(self.OptionFrame, font='Bold' , text ='Make Prediction', command=self.MakePredictionPlot, height = 2, width = 15).grid(row=0,column=1, sticky= "NS")
        self.ReturnHome_btn = Button(self.OptionFrame, text='Return Home', command=self.rebuildHome).grid(row=1,column=0, sticky= "NSWE", columnspan=2, pady=2)
        return


    ###**********************************************************************************************************************************************
    ###**********************************************************************************************************************************************
    def loadingData(self):
        global loadText1, loadText3, loadText2, loadText, loadTextDone
        global permanentStartDate
        global stockName, ExceptionsList

        Scraper_was_called=0

        #sleep(0.5)
        self.working_Frame.config(text = loadText1)
        #while(1):

        sleep(0.5)
        self.working_Frame.config(text = loadText2)
        sleep(0.5)
        self.working_Frame.config(text = loadText3)
        sleep(0.5)
        self.working_Frame.config(text = loadText)
        try:
            if(os.path.exists(stockName+".csv")):
                try:
                    dataset = pd.read_csv(stockName + ".csv", index_col='Date', parse_dates=['Date'])
                except:
                    print("Failed to open the csv inside the callFile function")
                    return
                #dataset = pd.read_csv(stockName + ".csv", index_col='Date', parse_dates=['Date'])
                todaysDate = str(dt.datetime.now())[0:10]
                dataset = dataset[todaysDate[0:4] : todaysDate[0:4]]
                finalDatasetDate = str(dataset.index[-1])[0:10]
                endDate = dt.datetime.now() #set end date to today ("datetime" object)

                print("The data end dates were the same:",(todaysDate == finalDatasetDate))

                if(todaysDate != finalDatasetDate):
                    try:
                        self.working_Frame.config(text = "Updating Data")
                        sleep(0.1)
                        Scraper_was_called = 1
                        threading.Thread(target=lambda:Data_Scraper(stockName, permanentStartDate, endDate, ExceptionsList)).start()
                    except:
                        print("Calling the datascraper script from the loadingData funct failed")
                else:
                    None ## date was the same so dont call scraper script.
                #sleep(3)
                while(1):
                    self.working_Frame.config(text = "Updating Data.")
                    sleep(0.75)
                    self.working_Frame.config(text = "Updating Data..")
                    sleep(0.75)
                    self.working_Frame.config(text = "Updating Data...")
                    sleep(0.75)
                    dataset = pd.read_csv(stockName + ".csv", index_col='Date', parse_dates=['Date'])
                    attributes = dataset.columns.tolist()
                    finalDatasetDate = str(dataset.index[-1])[0:10]
                    if(("EPS" in attributes) and ("Traded" in attributes)):
                        if(todaysDate == finalDatasetDate):
                            break
                        if( (dt.datetime.today().weekday() == 6) or (dt.datetime.today().weekday() == 5)):
                            break
                        #if((dt.datetime.today().weekday() == 0)):
                        if(datetime.now().time().hour < 9):
                            print("Updating overnight so dates will be different")
                            break
                    if(("EPS" not in attributes) and ("Traded" not in attributes) and (not Scraper_was_called) and (todaysDate == finalDatasetDate)):
                        print("data was missing, calling scraper")
                        try:
                            threading.Thread(target=lambda:Data_Scraper(stockName, permanentStartDate, endDate, ExceptionsList)).start()
                            Scraper_was_called = 1
                        except Exception as e:
                            self.working_Frame.config(text = "Something Went Wrong :( \nReturning Home")
                            print(e)
                            print("Calling the datascraper script from the loadingData funct failed within the no existing csv conditional")
                            sleep(3)
                            self.working_Frame.pack_forget()
                            self.working_Frame.destroy()
                            #global StockentryFrame
                            #global OptionFrame
                            self.greeting_Frame.config(text="Enter A Stock Ticker")
                            self.StockentryFrame = Label(master = root, bg="darkgrey")
                            self.StockentryFrame.pack()
                            self.name_entry = Entry(master = self.StockentryFrame, textvariable = self.ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
                            self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")
                            return
                    if "ScraperFail" in ExceptionsList:
                        self.working_Frame.config(text = "Something Went Wrong :( \nReturning Home")
                        print("The datascraper script failed")
                        sleep(3)
                        self.working_Frame.pack_forget()
                        self.working_Frame.destroy()
                        #global StockentryFrame
                        #global OptionFrame
                        self.greeting_Frame.config(text="Enter A Stock Ticker")
                        self.StockentryFrame = Label(master = root, bg="darkgrey")
                        self.StockentryFrame.pack()
                        self.name_entry = Entry(master = self.StockentryFrame, textvariable = self.ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
                        self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")
                        return

                self.working_Frame.config(text = loadTextDone)
                self.working_Frame.pack_forget()
                self.working_Frame.destroy()
                #StockentryFrame.grid_slaves(row=2, column=0)[0].destroy()
                self.build_optionsMenu()
                #break
            else:
                try:
                    endDate = dt.datetime.now() #set end date to today ("datetime" object)
                    threading.Thread(target=lambda:Data_Scraper(stockName, permanentStartDate, endDate)).start()
                except Exception as e:
                    self.working_Frame.config(text = "Something Went Wrong :( \nReturning Home")
                    print(e)
                    print("Calling the datascraper script from the loadingData funct failed within the no existing csv conditional")
                    sleep(3)
                    self.working_Frame.pack_forget()
                    self.working_Frame.destroy()
                    #global StockentryFrame
                    #global OptionFrame
                    self.greeting_Frame.config(text="Enter A Stock Ticker")
                    self.StockentryFrame = Label(master = root, bg="darkgrey")
                    self.StockentryFrame.pack()
                    self.name_entry = Entry(master = self.StockentryFrame, textvariable = self.ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
                    self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")
                    return

                while(1):
                    sleep(1)
                    self.working_Frame.config(text = loadText1)
                    sleep(1)
                    self.working_Frame.config(text = loadText2)
                    sleep(1)
                    self.working_Frame.config(text = loadText3)
                    sleep(1)
                    self.working_Frame.config(text = loadText)

                    if("ScraperFail" in ExceptionsList):
                        self.working_Frame.config(text = "Something Went Wrong :( \nReturning Home")
                        print("The datascraper script crashed")
                        sleep(3)
                        self.working_Frame.pack_forget()
                        self.working_Frame.destroy()
                        #global StockentryFrame
                        #global OptionFrame
                        self.greeting_Frame.config(text="Enter A Stock Ticker")
                        self.StockentryFrame = Label(master = root, bg="darkgrey")
                        self.StockentryFrame.pack()
                        self.name_entry = Entry(master = self.StockentryFrame, textvariable = self.ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
                        self.sub_btn = Button(self.StockentryFrame, text='Submit', command=self.submit).grid(row=1,column=1, sticky= "NS")
                        return

                    if(os.path.exists(stockName+".csv")):
                        dataset = pd.read_csv(stockName + ".csv", index_col='Date', parse_dates=['Date'])
                        attributes = dataset.columns.tolist()
                        if(("EPS" in attributes) and ("Traded" in attributes)):
                            sleep(4)
                            self.working_Frame.config(text = loadTextDone)
                            sleep(1)
                            self.working_Frame.pack_forget()
                            self.working_Frame.destroy()
                            #StockentryFrame.grid_slaves(row=2, column=0)[0].destroy()
                            self.build_optionsMenu()
                            break
                        else:
                            continue

        except Exception as e:
            print("OS file exist failed within the loadingData funct")
            print(e)
                #print("Failed to open the csv inside the callFile function")
                #return
    ###**********************************************************************************************************************************************
    ###**********************************************************************************************************************************************


    def submit(self):
        global stockName, ticker
        #global StockentryFrame
        ticker = self.ticker
        stockName = self.ticker.get()

        if ticker.get() == '':
            messagebox.showerror('Error message box',
                                 'Please enter a stock ticker in the blank field', parent=root)
            self.StockentryFrame.focus_force()
            return
        stockName = stockName.upper()
        print("The stock is : " + stockName)
        ticker.set("")

        try:
            pdr.DataReader(stockName, 'yahoo', '2021-01-01', '2022-01-01') #try to get main stock data from yahoo finance
        except:
            messagebox.showerror('Error message box',
                                 'The ticker you entered was not valid or real.\nPlease enter a valid stock ticker in the blank field', parent=root)
            self.StockentryFrame.focus_force()
            return

        #del currentData
        self.StockentryFrame.grid_slaves(row=0, column=1)[0].destroy()
        self.StockentryFrame.grid_slaves(row=1, column=1)[0].destroy()
        self.StockentryFrame.forget()
        self.StockentryFrame.destroy()

        #global working_Frame
        self.working_Frame = Label(master=root,text=loadText, font=('calibre',14,'normal'), fg="White", bg = "darkgrey")
        #working_Frame.grid(row=0,column=0,sticky=W)
        self.working_Frame.pack()

        #threading.Thread(target=callFile).start()
        threading.Thread(target=self.loadingData).start()

    def on_closing(self):
        global setWindowClosed
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            setWindowClosed = 1


#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************
#*******************************************************************************************************************************


setWindowClosed = 0
BranchtoGRU = 0

if __name__=="__main__":

    while(1):
        if not setWindowClosed:
            root = GUI()
            root.mainloop()
            print("We are now back inside of the main line")
            if BranchtoGRU:
                del root
                print("Entering GRU call")
                os.system('python GRU.py ' + stockName)
                sleep(1)
                BranchtoGRU = 0
        else:
            break


