
#800000  Hex for maroon

from tkinter import *
import os
import threading
from time import sleep
import csv
from PIL import ImageTk, Image
#from GUI_Functs import makeDataPlot, packPlotImage
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from tkinter import messagebox


#******************************************************
#Definitions section
Volume ="Volume: "+ "The volume of trade refers to the total number of shares or contracts exchanged between buyers and sellers of a security during trading hours on a given day.\nThe volume of trade is a measure of the market\'s activity and liquidity during a set period of time.\n Higher trading volumes are considered more positive than lower trading volumes because they mean more liquidity and better order execution."
#print(Volume)

stockName = ""
working_Frame = 0
loadText = "Collecting your stock data"
loadText1 = "Collecting your stock data."
loadText2 = "Collecting your stock data.."
loadText3 = "Collecting your stock data..."
loadTextDone = "Complete"


def returnToOpts():
    #OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
    OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
    OptionFrame.grid_slaves(row=1, column=0)[0].destroy()

    OptionFrame.config(image="")
    OptionFrame.pack_forget()
    labelimage.pack_forget()

    #Tk.update(root)
    build_optionsMenu()

def HelpExplainData():
    global Volume
    top= Toplevel(root)
    top.geometry("750x400")
    top.title("Help Window")
    Label(top, text= Volume, font=('Arial 12')).place(x=10)


def packPlotImage():
    global stockName
    global OptionFrame
    global labelimage
    greeting_Frame.config(text="Stock Data for ticker: "+stockName)
    OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
    OptionFrame.grid_slaves(row=0, column=1)[0].destroy()
    OptionFrame.grid_slaves(row=1, column=0)[0].destroy()


    percentage = 0.8
    image = Image.open(stockName+".png")
    width, height = image.size
    resized_dimensions = (int(width * percentage), int(height * percentage))
    resized = image.resize(resized_dimensions)
    #resize_image = image.resize((480, 375))
    photo = ImageTk.PhotoImage(resized)


    returnToOptions = Button(OptionFrame, text='Return', command=returnToOpts).grid(row=1,column=0, sticky= "NS")
    HelpExplain = Button(OptionFrame, text='Help', command=HelpExplainData).grid(row=1,column=1, sticky= "NS")

    #top= Toplevel(root)
    #top.geometry("750x800")
    #top.title("Child Window")
    labelimage = Label(master = OptionFrame, image = photo)
    labelimage.image = photo
    labelimage.grid(row=0, column=0, columnspan=2)
    #Label(top, text= "Hello World!", font=('Mistral 18 bold')).place(x=150,y=80)




def makeDataPlot():
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
    #ax1.set_xlabel('Day of year')
    ax1.set_ylabel('Value at Close USD:$')

    ax2.bar(dataset['Day'], dataset['Volume'])
    ax2.set_xlabel('Day of year')
    ax2.set_ylabel('Volume Traded')

    plt.tight_layout(h_pad=2)
    plt.savefig(stockName + '.png') #, bbox_inches="tight"
    packPlotImage()


def rebuildHome():
    global StockentryFrame
    global OptionFrame
    greeting_Frame.config(text="Enter A Stock Ticker")
    OptionFrame.grid_slaves(row=0, column=0)[0].destroy()
    OptionFrame.grid_slaves(row=1, column=0)[0].destroy()
    OptionFrame.grid_slaves(row=0, column=1)[0].destroy()
    OptionFrame.pack_forget()
    StockentryFrame = Label(master = root, bg="darkgrey")
    StockentryFrame.pack()
    name_entry = Entry(master = StockentryFrame, textvariable = ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")
    sub_btn = Button(StockentryFrame, text='Submit', command=submit).grid(row=1,column=1, sticky= "NS")




def build_optionsMenu():
    global OptionFrame
    greeting_Frame.config(text="Would you like to view the stock information or train a prediction model?")
    OptionFrame = Label(master = root, fg="white", bg = "#800000")
    OptionFrame.pack()
    view_stockInfo_btn = Button(OptionFrame ,font='Bold', text='View Information', command=makeDataPlot,height = 2, width = 15).grid(row=0,column=0, sticky= "NS")
    train_model_btn = Button(OptionFrame, font='Bold' , text ='Train Predictor', command=submit, height = 2, width = 15).grid(row=0,column=1, sticky= "NS")
    ReturnHome_btn = Button(OptionFrame, text='Return Home', command=rebuildHome).grid(row=1,column=0, sticky= "NSWE", columnspan=2, pady=2)
    return



def loadingData():
    global loadText1, loadText3, loadText2, loadText, loadTextDone
    if(os.path.exists(stockName+".csv")):
            working_Frame.pack_forget()
            working_Frame.destroy()
            build_optionsMenu()
            return
    else:
        threading.Thread(target=callFile).start()
    while(1):
        sleep(1)
        working_Frame.config(text = loadText1)
        sleep(1)
        working_Frame.config(text = loadText2)
        sleep(1)
        working_Frame.config(text = loadText3)
        sleep(1)
        working_Frame.config(text = loadText)
        try:
            with open(stockName+".csv", 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                working_Frame.config(text = loadTextDone)
                sleep(1)
                working_Frame.pack_forget()
                working_Frame.destroy()
                #StockentryFrame.grid_slaves(row=2, column=0)[0].destroy()
                build_optionsMenu()
                break
        except:
            None


def callFile():
    os.system('Python backend.py')



def submit():
    global stockName
    global StockentryFrame
    stockName=ticker.get()
    if ticker.get() == '':
        messagebox.showerror('Error message box',
                             'Please enter a stock ticker in the blank field', parent=root)
        StockentryFrame.focus_force()
        return

    print("The stock is : " + stockName)
    ticker.set("")
    StockentryFrame.grid_slaves(row=0, column=1)[0].destroy()
    StockentryFrame.grid_slaves(row=1, column=1)[0].destroy()
    StockentryFrame.forget()
    StockentryFrame.destroy()

    global working_Frame
    working_Frame = Label(master=root,text=loadText, font=('calibre',14,'normal'), fg="White", bg = "darkgrey")
    #working_Frame.grid(row=0,column=0,sticky=W)
    working_Frame.pack()

    #threading.Thread(target=callFile).start()
    threading.Thread(target=loadingData).start()


#*******************************************************************************************************************************

root = Tk()
root.title("GBSPS Window")
#root.configure(background="darkgrey")
root.geometry("900x525")
root.resizable(width=False, height=False)

background_image= ImageTk.PhotoImage(Image.open("stolen_stockImage.jpg"))
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


#background_Frame = Frame(master = root, background="#800000")
#

greeting_Frame = Label(master = root,text="Howdy Trader!\nEnter A Stock Ticker", font=('calibre',14,'bold'), fg="white", bg = "#800000")
    #.grid(row=0,column=0,sticky=W)

greeting_Frame.pack(fill=BOTH)

ticker = StringVar()

StockentryFrame = Label(master = root, bg="darkgrey")
StockentryFrame.pack()
#StockentryFrame.columnconfigure(index, weight)
#StockentryFrame.rowconfigure(index, weight)
name_entry = Entry(master = StockentryFrame, textvariable = ticker, font=('calibre',12,'normal')).grid(row=0,column=1, sticky= "NS")

sub_btn = Button(StockentryFrame, text='Submit', command=submit).grid(row=1,column=1, sticky= "NS")


#Packing frames
#background_Frame.pack()
#greeting_Frame.pack()
#name_entry.pack()
#sub_btn.pack()


root.mainloop()

