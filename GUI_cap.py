# This is a sample Python script.

#800000  Hex for maroon

from tkinter import Tk, Label, Button, X, BOTH, Frame, Text, END, Entry, StringVar

def submit():

    stockName=ticker.get()
    print("The stock is : " + stockName)
    ticker.set("")


root = Tk()
root.title("Tkinter Window")
root.geometry("500x300")

background_Frame = Frame(background="#800000", width=300, height = 300)
background_Frame.pack(fill=BOTH)

greeting_Frame = Label(master = background_Frame,text="Hello, Trader", font=('calibre',10,'normal'), fg="white", bg = "#800000")
greeting_Frame.place(x=0, y=0)

ticker = StringVar()
name_entry = Entry(root, textvariable = ticker, font=('calibre',10,'normal'))

sub_btn= Button(root,text = 'Submit', command = submit)

C:\Users\hsipp\PycharmProjects\GUIProto\GUI_cap.py
'''
button = Button(
    text="Click me!",
    bg="blue",
    fg="yellow",
)
'''
#Packing frames
background_Frame.pack()
sub_btn.pack()
greeting_Frame.pack()
name_entry.pack()
root.mainloop()

