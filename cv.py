import datetime as dt  # for working with dates
import pandas as pd  # for working with large lists and large csv files
from pandas_datareader import data as pdr  # for main stock data
import csv  # for working with csv files
from datetime import datetime
start = str(datetime.now())
end =str(datetime.now())
tims=pdr.DataReader(start,end)
filed="L.csv"
b=open(filed,w)
writer=csv.writer(b)
writer.writerow(start)
writer.writerow(end)
b.close()
