import datetime as dt

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from pandas_datareader import data as reader

# Importo la lista de componentes del S&P500
df = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\S&P500_Seasonality.csv', index_col='Date', parse_dates=True)
df = df.dropna(axis=True)
df.index = pd.to_datetime(df.index)

# Acumulo los retornos diarios en sampleo mensual

monthly_ret = df.pct_change().resample('M').agg(lambda x: (x+1).prod() - 1)
past_9 = (monthly_ret + 1).rolling(9).apply(np.prod)-1

# Formacion de portafolio
formation = dt.datetime(2017, 1, 1)

## Backtesting
def momentum(formation):
    end_measurement = formation - MonthEnd(1)
    ret_9 = past_9.loc[end_measurement]
    ret_9 = ret_9.reset_index()
    ret_9['deciles'] = pd.qcut(ret_9.iloc[:, 1], 10, labels=False, duplicates='drop')
    ganadores = ret_9[ret_9.deciles == 9]
    perdedores = ret_9[ret_9.deciles == 0]
    ganadores_ret = monthly_ret.loc[formation + MonthEnd(1), monthly_ret.columns.isin(ganadores['index'])]
    perdedores_ret = monthly_ret.loc[formation + MonthEnd(1), monthly_ret.columns.isin(perdedores['index'])]
    MomentumProfits = ganadores_ret.mean() - perdedores_ret.mean()
    return MomentumProfits

#Itero para obtener retornos de diferentes dias de formacion

profits = []
dates = []

for i in range(60):
    profits.append(momentum(formation + MonthEnd(i)))
    dates.append(formation + MonthEnd(i))
print(dates)

# Obtengo los datos del S&P500

SP500 = reader.DataReader('^GSPC', data_source='yahoo', start='2017-1-31', end='2021-12-31')['Adj Close']
SP500_monthly = SP500.pct_change().resample('M').agg(lambda x: (x+1).prod() - 1)

#Comparo los valores entre los retornos de la estrategia y el S&P500

frame = pd.DataFrame(profits).rename(columns={0: 'Profits'})
frame['S&P500'] = SP500_monthly.values
frame['Diferencia'] = frame.iloc[:, 0] - frame.iloc[:, 1]
frame['outperformed'] = ['SI' if i > 0 else 'NO' for i in frame.Diferencia]
cant_outperformed = frame[frame.outperformed == 'SI'].shape

print(f"Cantidad de veces que la estrategia supero al S&P500 en los ultimos 5 años: {cant_outperformed[0]} veces. "
      f"Un {round((cant_outperformed[0]/len(frame)) *100, 2)} % de las veces")
