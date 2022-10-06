import csv
from datetime import datetime
start = datetime.now()
end =datetime.now()
with open('oo.csv', 'w', newline='') as csvfile:
  writer = csv.DictWriter(csvfile)
  writer.writerow(start)
  writer.writerow(end)
  
