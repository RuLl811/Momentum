import numpy as np
from pandas_datareader import data as reader
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt

##################################################################################################################
########### Calculo de Beta y Alpha para los deciles de la estrategia 6,6 ##########
##################################################################################################################

df = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\S&P500.csv',index_col='Date', parse_dates=True)
df = df.dropna(axis=True)

#Calculo de los retornos mensuales acumulados
ret_acc = (((1 + df.pct_change()).cumprod() - 1) * 100).dropna()
ret_acc = ret_acc.resample('M').last()
# Fechas
start_formation = dt.datetime(2021, 1, 31)
end_formation = start_formation + relativedelta(months=6) - relativedelta(days=1)
start_mesurement = start_formation - relativedelta(months=7)
end_mesurement = start_formation - relativedelta(months=1)
# Calcular la suma de los retornos promedios durante los ultimos 6 meses
past_6 = (ret_acc + 1).rolling(6).sum()
# Performance de los ultimos 6 meses hasta el ultimo dia de medicion
ret_6_meses = past_6.loc[end_mesurement]
ret_6_meses = ret_6_meses.reset_index()
# Deciles
ret_6_meses['deciles'] = pd.qcut(ret_6_meses.iloc[:, 1], 10, labels=False)
# Top decil = ganadores
ganadores = ret_6_meses[ret_6_meses.deciles == 9]
ganadores.columns = ['Symbol', 'Price', 'Deciles']
#D8
d_8 = ret_6_meses[ret_6_meses.deciles == 8]
d_8.columns = ['Symbol', 'Price', 'Deciles']
#D7
d_7 = ret_6_meses[ret_6_meses.deciles == 7]
d_7.columns = ['Symbol', 'Price', 'Deciles']
#D6
d_6 = ret_6_meses[ret_6_meses.deciles == 6]
d_6.columns = ['Symbol', 'Price', 'Deciles']

#D5
d_5 = ret_6_meses[ret_6_meses.deciles == 5]
d_5.columns = ['Symbol', 'Price', 'Deciles']
#D4
d_4 = ret_6_meses[ret_6_meses.deciles == 4]
d_4.columns = ['Symbol', 'Price', 'Deciles']
#D3
d_3 = ret_6_meses[ret_6_meses.deciles == 3]
d_3.columns = ['Symbol', 'Price', 'Deciles']
#D2
d_2 = ret_6_meses[ret_6_meses.deciles == 2]
d_2.columns = ['Symbol', 'Price', 'Deciles']
#D8
d_1 = ret_6_meses[ret_6_meses.deciles == 1]
d_1.columns = ['Symbol', 'Price', 'Deciles']
# Bottom decil = perdedores
perdedores = ret_6_meses[ret_6_meses.deciles == 0]
perdedores.columns = ['Symbol', 'Price', 'Deciles']

# Retornos portafolio " ganadores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_ganadores = df.loc[start_formation:end_formation, df.columns.isin(ganadores.Symbol)]
# Armo los ponderadores
# Equal Weight
N = len(df_ganadores.columns)
equal_weights = N * [1/N]
df_ganadores_ret = df_ganadores.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_ganadores = np.dot(equal_weights, df_ganadores_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_ganadores = ((1 + equal_weighted_returns_ganadores).cumprod() - 1) * 100
cewrp_ganadores = pd.Series(cum_equal_weighted_returns_ganadores, index=df_ganadores_ret.index)

# Retornos portafolio " D8 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d8 = df.loc[start_formation:end_formation, df.columns.isin(d_8.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d8 = len(df_d8.columns)
equal_weights_d8 = N_d8 * [1/N_d8]
df_d8_ret = df_d8.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d8 = np.dot(equal_weights_d8, df_d8_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d8 = ((1 + equal_weighted_returns_d8).cumprod() - 1) * 100
cewrp_d8 = pd.Series(cum_equal_weighted_returns_d8, index=df_d8_ret.index)

# Retornos portafolio " D7 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d7 = df.loc[start_formation:end_formation, df.columns.isin(d_7.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d7 = len(df_d7.columns)
equal_weights_d7 = N_d7 * [1/N_d7]
df_d7_ret = df_d7.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d7 = np.dot(equal_weights_d7, df_d7_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d7 = ((1 + equal_weighted_returns_d7).cumprod() - 1) * 100
cewrp_d7 = pd.Series(cum_equal_weighted_returns_d7, index=df_d7_ret.index)

# Retornos portafolio " D6 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d6 = df.loc[start_formation:end_formation, df.columns.isin(d_6.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d6 = len(df_d6.columns)
equal_weights_d6 = N_d6 * [1/N_d6]
df_d6_ret = df_d6.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d6 = np.dot(equal_weights_d6, df_d6_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d6 = ((1 + equal_weighted_returns_d6).cumprod() - 1) * 100
cewrp_d6 = pd.Series(cum_equal_weighted_returns_d6, index=df_d6_ret.index)

# Retornos portafolio " D5 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d5 = df.loc[start_formation:end_formation, df.columns.isin(d_5.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d5 = len(df_d5.columns)
equal_weights_d5 = N_d5 * [1/N_d5]
df_d5_ret = df_d5.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d5 = np.dot(equal_weights_d5, df_d5_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d5 = ((1 + equal_weighted_returns_d5).cumprod() - 1) * 100
cewrp_d5 = pd.Series(cum_equal_weighted_returns_d5, index=df_d5_ret.index)

# Retornos portafolio " D4 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d4 = df.loc[start_formation:end_formation, df.columns.isin(d_4.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d4 = len(df_d4.columns)
equal_weights_d4 = N_d4 * [1/N_d4]
df_d4_ret = df_d4.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d4 = np.dot(equal_weights_d4, df_d4_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d4 = ((1 + equal_weighted_returns_d4).cumprod() - 1) * 100
cewrp_d4 = pd.Series(cum_equal_weighted_returns_d4, index=df_d4_ret.index)

# Retornos portafolio " D3 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d3 = df.loc[start_formation:end_formation, df.columns.isin(d_3.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d3 = len(df_d3.columns)
equal_weights_d3 = N_d3 * [1/N_d3]
df_d3_ret = df_d3.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d3 = np.dot(equal_weights_d3, df_d3_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d3 = ((1 + equal_weighted_returns_d3).cumprod() - 1) * 100
cewrp_d3 = pd.Series(cum_equal_weighted_returns_d3, index=df_d3_ret.index)

# Retornos portafolio " D2 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d2 = df.loc[start_formation:end_formation, df.columns.isin(d_2.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d2 = len(df_d2.columns)
equal_weights_d2 = N_d2 * [1/N_d2]
df_d2_ret = df_d2.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d2 = np.dot(equal_weights_d2, df_d2_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d2 = ((1 + equal_weighted_returns_d2).cumprod() - 1) * 100
cewrp_d2 = pd.Series(cum_equal_weighted_returns_d2, index=df_d2_ret.index)

# Retornos portafolio " D1 "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_d1 = df.loc[start_formation:end_formation, df.columns.isin(d_1.Symbol)]
# Armo los ponderadores
# Equal Weight
N_d1 = len(df_d1.columns)
equal_weights_d1 = N_d1 * [1/N_d1]
df_d1_ret = df_d1.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_d1 = np.dot(equal_weights_d1, df_d1_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_d1 = ((1 + equal_weighted_returns_d1).cumprod() - 1) * 100
cewrp_d1 = pd.Series(cum_equal_weighted_returns_d1, index=df_d1_ret.index)

## Retornos portafolio " perdedores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_perdedores = df.loc[start_formation:end_formation, df.columns.isin(perdedores.Symbol)]
# Armo los ponderadores
# Equal Weight
equal_weights_perdedores = N * [1/N]
df_perdedores_ret = df_perdedores.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_perdedores = np.dot(equal_weights_perdedores, df_perdedores_ret.transpose())
#Retornso acumulados
cum_equal_weighted_returns_perdedores = ((1 + equal_weighted_returns_perdedores).cumprod() - 1) * 100
cewrp_perdedores = pd.Series(cum_equal_weighted_returns_perdedores, index=df_perdedores_ret.index)

####### Calculo de Beta & Alpha de los portafolios equal-weighted #######

# Obtengo los datos del S&P500
SP500 = reader.DataReader('^GSPC', data_source='yahoo', start='2021-1-31', end='2021-7-30')['Adj Close']
SP500 = SP500.pct_change(1).dropna()

# Beta & Alpha del portafolio 'Ganadores'
ewr_d9 = pd.Series(equal_weighted_returns_ganadores, index=df_ganadores_ret.index)
beta_d9, alpha_d9, _, _, _ = linregress(SP500, ewr_d9)  # Solo miro beta y alpha
ewr_d8 = pd.Series(equal_weighted_returns_d8, index=df_d8_ret.index)
beta_d8, alpha_d8, _, _, _ = linregress(SP500, ewr_d8)  # Solo miro beta y alpha

ewr_d7 = pd.Series(equal_weighted_returns_d7, index=df_d7_ret.index)
beta_d7, alpha_d7, _, _, _ = linregress(SP500, ewr_d7)  # Solo miro beta y alpha

ewr_d6 = pd.Series(equal_weighted_returns_d6, index=df_d6_ret.index)
beta_d6, alpha_d6, _, _, _ = linregress(SP500, ewr_d6)  # Solo miro beta y alpha

ewr_d5 = pd.Series(equal_weighted_returns_d5, index=df_d5_ret.index)
beta_d5, alpha_d5, _, _, _ = linregress(SP500, ewr_d5)  # Solo miro beta y alpha

ewr_d4 = pd.Series(equal_weighted_returns_d4, index=df_d4_ret.index)
beta_d4, alpha_d4, _, _, _ = linregress(SP500, ewr_d4)  # Solo miro beta y alpha

ewr_d3 = pd.Series(equal_weighted_returns_d3, index=df_d3_ret.index)
beta_d3, alpha_d3, _, _, _ = linregress(SP500, ewr_d3)  # Solo miro beta y alpha

ewr_d2 = pd.Series(equal_weighted_returns_d2, index=df_d2_ret.index)
beta_d2, alpha_d2, _, _, _ = linregress(SP500, ewr_d2)  # Solo miro beta y alpha

ewr_d1 = pd.Series(equal_weighted_returns_d1, index=df_d1_ret.index)
beta_d1, alpha_d1, _, _, _ = linregress(SP500, ewr_d1)  # Solo miro beta y alpha

ewr_d0 = pd.Series(equal_weighted_returns_perdedores, index=df_perdedores_ret.index)
beta_d0, alpha_d0, _, _, _ = linregress(SP500, ewr_d0)  # Solo miro beta y alpha

# Calculo de Beta y Alpha de los diferentes deciles utilizando la estrategia 6,6

print(f"Beta del portafolio equal-weighted conformado por el decil 9: {round(beta_d9, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 9: {round(alpha_d9,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 8: {round(beta_d8, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 8: {round(alpha_d8,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 7: {round(beta_d7, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 7: {round(alpha_d7,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 6: {round(beta_d6, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 6: {round(alpha_d6,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 5: {round(beta_d5, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 5: {round(alpha_d5,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 4: {round(beta_d4, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 4: {round(alpha_d4,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 3: {round(beta_d3, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 3: {round(alpha_d3,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 2: {round(beta_d2, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 2: {round(alpha_d2,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 1: {round(beta_d1, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 1: {round(alpha_d1,4)}")
print('\n')
print(f"Beta del portafolio equal-weighted conformado por el decil 0: {round(beta_d0, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 0: {round(alpha_d0,4)}")
print('\n')

# Calculo de los retornos mensuales promedios para cada uno de los deciles de la estrategia 6,6
print(f"El retorno mensual promedio del decil 9 es : {equal_weighted_returns_ganadores.mean()*100} %")
print(f"El retorno mensual promedio del decil 8 es: {equal_weighted_returns_d8.mean()*100} %")
print(f"El retorno mensual promedio del decil 7 es: {equal_weighted_returns_d7.mean()*100} %")
print(f"El retorno mensual promedio del decil 6 es: {equal_weighted_returns_d6.mean()*100} %")
print(f"El retorno mensual promedio del decil 5 es: {equal_weighted_returns_d5.mean()*100} %")
print(f"El retorno mensual promedio del decil 4 es: {equal_weighted_returns_d4.mean()*100} %")
print(f"El retorno mensual promedio del decil 3 es: {equal_weighted_returns_d3.mean()*100} %")
print(f"El retorno mensual promedio del decil 2 es: {equal_weighted_returns_d2.mean()*100} %")
print(f"El retorno mensual promedio del decil 1 es: {equal_weighted_returns_d1.mean()*100} %")
print(f"El retorno mensual promedio del decil 0 es: {equal_weighted_returns_perdedores.mean()*100} %")
print('\n')

# Calculo de los retornos acumulados mensuales para cada uno de los deciles de la estrategia 6,6
print(f"El retorno mensual acumulado del decil 9 es : {cewrp_ganadores[end_formation]} %")
print(f"El retorno mensual acumulado del decil 8 es: {cewrp_d8[end_formation]} %")
print(f"El retorno mensual acumulado del decil 7 es: {cewrp_d7[end_formation]} %")
print(f"El retorno mensual acumulado del decil 6 es: {cewrp_d6[end_formation]} %")
print(f"El retorno mensual acumulado del decil 5 es: {cewrp_d5[end_formation]} %")
print(f"El retorno mensual acumulado del decil 4 es: {cewrp_d4[end_formation]} %")
print(f"El retorno mensual acumulado del decil 3 es: {cewrp_d3[end_formation]} %")
print(f"El retorno mensual acumulado del decil 2 es: {cewrp_d2[end_formation]} %")
print(f"El retorno mensual acumulado del decil 1 es: {cewrp_d2[end_formation]} %")
print(f"El retorno mensual acumulado del decil 0 es: {cewrp_perdedores[end_formation]} %")

##### End #####