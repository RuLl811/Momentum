import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import seaborn as sns
import matplotlib.pyplot as plt

##################################################################################################################
############################### Estudio de estacionalidad para la estrategia 6,6 #################################
##################################################################################################################

# Importo la lista de componentes del S&P500
df = pd.read_csv(r'C:\Users\rumtl\PycharmProjects\Tesis\Momentum strategy\S&P500_Seasonality.csv', index_col='Date', parse_dates=True)
df = df.dropna(axis=True)
df.index = pd.to_datetime(df.index)

# Acumulo los retornos diarios en sampleo mensual

monthly_ret = (((1 + df.pct_change()).cumprod() - 1) * 100).dropna()
monthly_ret = monthly_ret.resample('M').last()
past_6 = (monthly_ret + 1).rolling(6).sum()

# Fecha de inicio de formacion
start_formation = dt.datetime(2012, 8, 31)  # Dia de compra del portafolio

## Genero una funcion para luego poder iterar los retornos con diferentes

def momentum(start_formation):
    end_formation = start_formation + MonthEnd(6)
    end_measurement = start_formation - MonthEnd(1)
    ret_6 = past_6.loc[end_measurement]
    ret_6 = ret_6.reset_index()
    ret_6['deciles'] = pd.qcut(ret_6.iloc[:, 1], 10, labels=False, duplicates='drop')
    ganadores = ret_6[ret_6.deciles == 9]
    perdedores = ret_6[ret_6.deciles == 0]
    df_ganadores = df.loc[start_formation:end_formation, df.columns.isin(ganadores['index'])]
    df_perdedores = df.loc[start_formation:end_formation, df.columns.isin(perdedores['index'])]
    #Poderadores
    N_G = len(df_ganadores.columns)
    equal_weights_ganadores = N_G * [1 / N_G]
    N_P = len(df_perdedores.columns)
    equal_weights_perdedores = N_P * [1 / N_P]
    #Calculo de retornos
    df_ganadores_ret = df_ganadores.pct_change().dropna()
    df_perdedores_ret = df_perdedores.pct_change().dropna()
    equal_weighted_returns_ganadores = (np.dot(equal_weights_ganadores, df_ganadores_ret.transpose()))
    equal_weighted_returns_perdedores = (np.dot(equal_weights_perdedores, df_perdedores_ret.transpose()))
    MomentumProfits = (equal_weighted_returns_ganadores.mean() - equal_weighted_returns_perdedores.mean()) * 100
    return MomentumProfits

#Itero para obtener retornos de la estrategia 6,6
profits = []
dates = []

for i in range(60):
    profits.append(momentum(start_formation + MonthEnd(i)))
    dates.append(start_formation + MonthEnd(i))


# Desagrego el DataFrame por meses
# Configuro el tipo de dato, los concateno, renombro las columnas y seteo el indice
profits = pd.DataFrame(profits)
dates = pd.DataFrame(dates)
Retornos = pd.concat([dates, profits], axis=1)

Retornos.columns = ['Date', 'Retornos %']
Retornos['Date'] = pd.to_datetime(Retornos['Date']) # Convierto la columna dias en un tipo DateTime
Retornos = Retornos.set_index('Date')
# Creo dos columnas, una para el año y otra para los meses
Retornos['Año'] = Retornos.index.year
Retornos['Mes'] = Retornos.index.month

#   Grafico
sns.set(style='whitegrid')
plt.title("Estacionalidad de los Retornos", fontsize=16)
sns.despine(top=True,
            right=True,
            left=True,
            bottom=False)
sns.boxplot(x=Retornos['Mes'], y=Retornos['Retornos %'], palette="husl")

plt.show()

# Calculo de retornos promedios para testear el efecto 'Enero'
print(f"La media de los retornos promedios en los meses de Enero es de: {round(Retornos.loc[Retornos['Mes'] == 1].median()[0], 5)} %")
print(f"La media de los retornos promedios entre los meses Febrero-Diciembre es de: {round(Retornos.loc[Retornos['Mes'] > 1].median()[0], 5)} %")

