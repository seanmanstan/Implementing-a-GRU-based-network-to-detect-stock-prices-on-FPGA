import csv
from datetime import datetime
start = datetime.now()
end =datetime.now()
tims=pdr.DataReader(start,end)
filed="L.csv"
with open(filed, 'w',encoding='UTF8') as f:  # this writes the current stock tickers data to a csv file
  tims.to_csv(filed)
