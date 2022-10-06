import datetime as dt  # for working with dates
import pandas as pd  # for working with large lists and large csv files
from pandas_datareader import data as pdr  # for main stock data
import csv  # for working with csv files
from datetime import datetime
start = dt.now()
end =dt.now()
tims=pdr.DataReader(start,end)
filed="L.csv"
with open(filed, 'w',encoding='UTF8') as f:  # this writes the current stock tickers data to a csv file
  tims.to_csv(filed)
