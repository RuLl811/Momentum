import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as reader


# Import los retornos acumulados de las estrategias con duration de portafolio igual a 3 meses
df_1 = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\Estrategias\Retornos acumulados de las estrategias/3,3.csv', index_col='Date', parse_dates=True)
df_2 = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\Estrategias\Retornos acumulados de las estrategias/6,3.csv', index_col='Date', parse_dates=True)
df_3 = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\Estrategias\Retornos acumulados de las estrategias/9,3.csv', index_col='Date', parse_dates=True)

df_1.columns = ['Portfolio_Acc_Ret']
df_2.columns = ['Portfolio_Acc_Ret']
df_3.columns = ['Portfolio_Acc_Ret']

# Obtengo los datos del S&P500
SP500 = reader.DataReader('^GSPC', data_source='yahoo', start='2021-3-31', end='2021-6-30')['Adj Close']
SP500 = SP500.pct_change(1).dropna()
SP500_cum = ((1 + SP500).cumprod() - 1) * 100
SP500_cum = SP500_cum.to_frame()
SP500_cum = SP500_cum.reset_index()
SP500_cum.columns = ['Date', 'SP500_Acc_Ret']
SP500_cum.index = df_1.index

# Grafico
fig, ax = plt.subplots(figsize=(10, 3), dpi=110)
df_1['Portfolio_Acc_Ret'].plot(ax=ax, label='3,1')
df_2['Portfolio_Acc_Ret'].plot(ax=ax,label='6,1')
df_3['Portfolio_Acc_Ret'].plot(ax=ax,label='9,1')
SP500_cum['SP500_Acc_Ret'].plot(ax=ax, label='S&P500', dashes=[2, 2, 10, 2])
plt.ylabel("Daily Returns")
plt.legend()
plt.show()
