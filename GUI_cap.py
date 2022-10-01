# This is a sample Python script.

#800000  Hex for maroon

from tkinter import Tk, Label, Button, X, BOTH, Frame, Text, END, Entry, StringVar
import os
import sys
import threading
from time import sleep
import csv

stockName = ""
working_Frame = 0
loadText = "Collecting your stock data"
loadText1 = "Collecting your stock data."
loadText2 = "Collecting your stock data.."
loadText3 = "Collecting your stock data..."
loadTextDone = "Complete"
def loadingData():
    global loadText1, loadText3, loadText2, loadText, loadTextDone
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
                break
        except:
            None

def callFile():
    os.system('Python backend.py')


def submit():
    global stockName
    stockName=ticker.get()
    print("The stock is : " + stockName)
    ticker.set("")
    name_entry.pack_forget()
    name_entry.destroy()
    sub_btn.pack_forget()
    sub_btn.destroy()
    global working_Frame
    working_Frame = Label(master=root,text=loadText, font=('calibre',14,'normal'), fg="black")
    working_Frame.pack()

    threading.Thread(target=callFile).start()
    threading.Thread(target=loadingData).start()


root = Tk()
root.title("Tkinter Window")
root.geometry("500x300")

#background_Frame = Frame(master = root, background="#800000")
#

greeting_Frame = Label(master = root,text="Howdy, Trader", font=('calibre',12,'normal'), fg="white", bg = "#800000")
greeting_Frame.place(x=0, y=0)
greeting_Frame.pack(fill=BOTH)

ticker = StringVar()
name_entry = Entry(root, textvariable = ticker, font=('calibre',12,'normal'))

sub_btn= Button(root,text = 'Submit', command=submit)

'''
button = Button(
    text="Click me!",
    bg="blue",
    fg="yellow",
)
'''
#Packing frames
#background_Frame.pack()
greeting_Frame.pack()
name_entry.pack()
sub_btn.pack()


root.mainloop()

