import datetime as dt

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as reader
from scipy import stats
from scipy.stats import linregress

##################################################################################################################
########### Cross-Sectional Momentum Strategy: Retornos rezagados a 6 meses; 1 meses "Holding Periods" ##########
##################################################################################################################

df = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\S&P500.csv', index_col='Date', parse_dates=True)
df = df.dropna(axis=True)

#Calculo de los retornos mensuales acumulados
ret_acc = (((1 + df.pct_change()).cumprod() - 1) * 100).dropna()
ret_acc = ret_acc.resample('M').last()

# Fechas
start_formation = dt.datetime(2021, 3, 31)  # Dia de compra del portafolio
end_formation = start_formation + relativedelta(months=1) # Dia de venta del portafolio
start_mesurement = start_formation - relativedelta(months=7) # Primer dia de calculo de rendimientos
end_mesurement = start_formation - relativedelta(months=1) # Ultimo dia de calculo de rendimientos.

print(f" Cantidad de dias del portafolio: {end_formation - start_formation}")
print(f" Cantidad de dias de backward looking: {end_mesurement - start_mesurement}")

# Calcular la suma de los retornos promedios durante los ultimos 6 meses
past_6 = (ret_acc + 1).rolling(6).sum()

# Performance de los ultimos 6 meses hasta el ultimo dia de medicion
ret_6_meses = past_6.loc[end_mesurement]
ret_6_meses = ret_6_meses.reset_index()

# Clasifico la performance de cada activo por deciles
ret_6_meses['deciles'] = pd.qcut(ret_6_meses.iloc[:, 1], 10, labels=False)

# Top decil = ganadores
ganadores = ret_6_meses[ret_6_meses.deciles == 9]
ganadores.columns = ['Symbol', 'Price', 'Deciles']
print(f"Los activos con los mayores retornos son:")
print('\n')
print(ganadores)
print('\n')

# Bottom decil = perdedores
perdedores = ret_6_meses[ret_6_meses.deciles == 0]
perdedores.columns = ['Symbol', 'Price', 'Deciles']
print(f"Los activos con los peores retornos son:")
print('\n')
print(perdedores)
print('\n')

# Retornos portafolio " ganadores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_ganadores = df.loc[start_formation:end_formation, df.columns.isin(ganadores.Symbol)]
# Armo los ponderadores
# Equal Weight
N_G = len(df_ganadores.columns)
equal_weights_ganadores = N_G * [1/N_G]
df_ganadores_ret = df_ganadores.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_ganadores = np.dot(equal_weights_ganadores, df_ganadores_ret.transpose())
#Retornos acumulados del portafolio
cum_equal_weighted_returns_ganadores = ((1 + equal_weighted_returns_ganadores).cumprod() - 1) * 100
cewrp_ganadores = pd.Series(cum_equal_weighted_returns_ganadores, index=df_ganadores_ret.index)
retornos_impulso_compra = cewrp_ganadores[end_formation]

## Retornos portafolio " perdedores "
    # Filtro los activos segun la fecha de comienzo y final del portafolio
df_perdedores = df.loc[start_formation:end_formation, df.columns.isin(perdedores.Symbol)]
# Armo los ponderadores
# Equal Weight
N_P = len(df_perdedores.columns)
equal_weights_perdedores = N_P * [1/N_P]
df_perdedores_ret = df_perdedores.pct_change(1).dropna()
# Retornos del portafolio
equal_weighted_returns_perdedores = np.dot(equal_weights_perdedores, df_perdedores_ret.transpose())
#Retornos acumulados del portafolio
cum_equal_weighted_returns_perdedores = ((1 + equal_weighted_returns_perdedores).cumprod() - 1) * 100
cewrp_perdedores = pd.Series(cum_equal_weighted_returns_perdedores, index=df_perdedores_ret.index)
retornos_impulso_venta = cewrp_perdedores[end_formation]

retornos_netos = retornos_impulso_compra - retornos_impulso_venta
equal_weighted_returns_portafolio = np.dot(equal_weights_ganadores, df_ganadores_ret.transpose()) - np.dot(equal_weights_perdedores, df_perdedores_ret.transpose())

print(f"El retornos del portafolio 'ganadores' es: {retornos_impulso_compra} %")
print(f"El retornos del portafolio 'perdedores' es: {retornos_impulso_venta} %")
print(f"El retornos del portafolio es: {retornos_netos} %")
print('\n')
print(f"El desvio estandard del portafolio de los ganadores es: {cewrp_ganadores.std()} ")
print(f"El desvio estandard del portafolio de los perdedores es: {cewrp_perdedores.std()} ")
print('\n')

################# Portafolio Statistics #################

stat_ganadores = pd.Series(stats.ttest_1samp(equal_weighted_returns_ganadores, 0.0)).to_frame().T
stat_perdedores = pd.Series(stats.ttest_1samp(equal_weighted_returns_perdedores, 0.0)).to_frame().T
stat_portafolio = pd.Series(stats.ttest_1samp(equal_weighted_returns_portafolio, 0.0)).to_frame().T
t_output = pd.concat([stat_ganadores, stat_perdedores, stat_portafolio]).rename(columns={0: 't-stat', 1: 'p-value'})

print('t-value para portafolio de ganadores, perdedores y portafolio respectivamente :')
print(t_output['t-stat'])
print('\n')
####### Calculo de Beta & Alpha de los portafolios equal-weighted #######

# Obtengo los datos del S&P500

SP500 = reader.DataReader('^GSPC', data_source='yahoo', start='2021-3-31', end='2021-4-30')['Adj Close']
SP500 = SP500.pct_change(1).dropna()

# Beta & Alpha del portafolio 'Ganadores'

ewr = pd.Series(equal_weighted_returns_ganadores, index=df_ganadores_ret.index)

beta, alpha, _, _, _ = linregress(SP500, ewr)  # Solo miro beta y alpha

print(f"Beta del portafolio equal-weighted conformado por el decil 9: {round(beta, 2)},"'\n'
      f"Alpha del portafolio equal-weighted conformado por el decil 9: {round(alpha,4)}")

############### Metricas de Riesgo ###############
# Sortino
portafolio_ret = equal_weighted_returns_ganadores - equal_weighted_returns_perdedores
portafolio_ret = pd.Series(portafolio_ret, index=df_perdedores_ret.index)

threshold = 0
mean_return = portafolio_ret.mean()
downside = portafolio_ret[portafolio_ret < threshold].dropna()

risk_free_rate = 0
downside_std = downside.std()

sortino_ratio = (mean_return - risk_free_rate) / downside_std
print('\n')
print(f"Sortio-Ratio {sortino_ratio * np.sqrt(252)}")

# Sharpe Ratio
std = portafolio_ret.std()
sharpe_ratio = (mean_return-risk_free_rate)/std
print(f"Sharpe-Ratio {sharpe_ratio}")

import matplotlib.pyplot as plt
# Grafica comparativa de retornos acumulados

SP500_cum = ((1 + SP500).cumprod() - 1) * 100

cewrp_portafolio =  cewrp_ganadores - cewrp_perdedores
plt.figure(figsize=(10, 3), dpi=80)
cewrp_portafolio.plot(label='Equal Weighted Portfolio (Cumul Ret)')
SP500_cum.plot(label='SP500 (Cumul Ret)')
plt.legend()
plt.show()
cewrp_portafolio.to_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\Estrategias\Retornos acumulados de las estrategias\6,1.csv')
### End ###