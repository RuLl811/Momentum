import datetime as dt
import statsmodels.api as sm
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as reader

##################################################################################################################
########### Fama-French Factor Model: Retornos rezagados a 9 meses; 6 meses "Holding Periods" ##########
##################################################################################################################

df = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\S&P500.csv',index_col='Date', parse_dates=True)
df = df.dropna(axis=True)

#Calculo de los retornos mensuales acumulados
ret_acc = (((1 + df.pct_change()).cumprod() - 1) * 100).dropna()
ret_acc = ret_acc.resample('M').last()

# Fechas
start_formation = dt.datetime(2021, 1, 31)  # Dia de compra del portafolio
end_formation = start_formation + relativedelta(months=6) - relativedelta(days=1)  # Dia de venta del portafolio
start_mesurement = start_formation - relativedelta(months=10)
end_mesurement = start_formation - relativedelta(months=1) # Ultimo dia de calculos de rendimientos.

print(f" Cantidad de dias del portafolio: {end_formation - start_formation}")
print(f" Cantidad de dias de backward looking: {end_mesurement - start_mesurement}")

# Calcular la suma de los retornos promedios durante los ultimos 9 meses
past_9 = (ret_acc + 1).rolling(9).sum()

# Performance de los ultimos 9 meses hasta el ultimo dia de medicion
ret_9_meses = past_9.loc[end_mesurement]
ret_9_meses = ret_9_meses.reset_index()

# Deciles
ret_9_meses['deciles'] = pd.qcut(ret_9_meses.iloc[:, 1], 10, labels=False)

# Top decil = ganadores
ganadores = ret_9_meses[ret_9_meses.deciles == 9]
ganadores.columns = ['Symbol', 'Price', 'Deciles']

# Bottom decil = perdedores
perdedores = ret_9_meses[ret_9_meses.deciles == 0]
perdedores.columns = ['Symbol', 'Price', 'Deciles']

# Retornos portafolio " ganadores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_ganadores = df.loc[start_formation:end_formation, df.columns.isin(ganadores.Symbol)]
# Armo los ponderadores
# Equal Weight
N_G = len(df_ganadores.columns)
equal_weights_ganadores = N_G * [1/N_G]
df_ganadores_ret = df_ganadores.pct_change(1).dropna()
equal_weighted_returns_ganadores = np.dot(equal_weights_ganadores, df_ganadores_ret.transpose())

## Retornos portafolio " perdedores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_perdedores = df.loc[start_formation:end_formation, df.columns.isin(perdedores.Symbol)]
# Armo los ponderadores
# Equal Weight
N_P = len(df_perdedores.columns)
equal_weights_perdedores = N_P * [1/N_P]
df_perdedores_ret = df_perdedores.pct_change(1).dropna()
equal_weighted_returns_perdedores = np.dot(equal_weights_perdedores, df_perdedores_ret.transpose())

#Retornos del portafolio
portafolio_ret = equal_weighted_returns_ganadores - equal_weighted_returns_perdedores
portafolio_ret = pd.Series(portafolio_ret, index=df_perdedores_ret.index)
portafolio_ret = portafolio_ret.resample('M').last()

# Factores de FAMA-FRENCH 3 FACTOR MODEL
factors = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start_formation+relativedelta(months=1), end_formation)[0]

# Transformo a un DataFrame e igualo los indices
portafolio_ret = portafolio_ret.to_frame()
portafolio_ret.columns = ['Portfolio_Ret']
portafolio_ret.index = factors.index
# Concateno los Dataframe
df_2 = pd.merge(portafolio_ret, factors, on='Date')
#Igualo la unidad de medida
df_2[['Mkt-RF', 'SMB', 'HML']] = df_2[['Mkt-RF', 'SMB', 'HML']] / 100

# Creo la variable dependientes e independientes

df_2['Portfolio_Ret-RF'] = df_2.Portfolio_Ret
y = df_2['Portfolio_Ret-RF']
x = df_2[['Mkt-RF', 'SMB', 'HML']]
x_sm = sm.add_constant(x)

#Calculo el modelo
modelo = sm.OLS(y, x_sm)
results = modelo.fit()
print(results.summary())