import os
import csv
import pandas as pd

if 'time.csv' in os.listdir():
    os.remove('./time.csv')

for i in os.listdir():
    if '.csv' in i and not i == 'time.csv':
        data = pd.read_csv(i, header=None)
        tmp = data.mean() * 1000
        with open('./time.csv', 'a', encoding='utf8') as f:
            wr = csv.writer(f)
            wr.writerow([i[0:-4], *tmp])