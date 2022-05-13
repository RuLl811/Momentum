import sqlalchemy
import pandas as pd
import yfinance as yf
import bs4 as bs
import requests
from pandas_datareader import data as reader
import datetime as dt
from dateutil.relativedelta import relativedelta

# Utilizo web scraping para obtener la lista de los componentes del S&P500 desde Wikipedia
html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(html.text)

tickers = []

table = soup.find('table', {'class': 'wikitable sortable'})
rows = table.findAll('tr')[1:]

for row in rows:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker[:-1])
# Transformo la lista a un DataFrame
df = pd.DataFrame(tickers, columns=['Symbols'])
print(df)

# Obtengo los precios de todos los componentes del S&P

start = dt.datetime(2020, 1, 1)
end = dt.datetime(2022, 4, 29)

df_2 = reader.get_data_yahoo(tickers, start, end)

df_2.to_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\SP500_2.csv')
