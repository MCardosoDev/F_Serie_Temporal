import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

def plot_ts(df):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df.meantemp, color='blue')
    plt.xlabel('Data')
    plt.ylabel('Temperatura Média')
    plt.title('Variação da Temperatura Média')
    plt.subplot(2, 2, 2)
    plt.plot(df.index, df.humidity, color='green')
    plt.xlabel('Data')
    plt.ylabel('Umidade')
    plt.title('Variação da Umidade')
    plt.subplot(2, 2, 3)
    plt.plot(df.index, df.wind_speed, color='orange')
    plt.xlabel('Data')
    plt.ylabel('Velocidade do Vento')
    plt.title('Variação da Velocidade do Vento')
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df.meanpressure, color='red')
    plt.xlabel('Data')
    plt.ylabel('Pressão Média')
    plt.title('Variação da Pressão Média')
    plt.tight_layout()
    plt.show()

def plot_decompose(df, title):
    resultados = seasonal_decompose(df)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 9))
    resultados.observed.plot(ax=ax1)
    ax1.set_title('Série Observada')
    resultados.trend.plot(ax=ax2)
    ax2.set_title('Componente de Tendência')
    resultados.seasonal.plot(ax=ax3)
    ax3.set_title('Componente de Sazonalidade')
    resultados.resid.plot(ax=ax4)
    ax4.set_title('Resíduo')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def augmented_dickey_fuller(df):
    result = adfuller(df)
    print('Teste ADf')
    print(f'Teste estatistico: {result[0]}')
    print(f'P-Value: {result[1]}')
    print(f'Valores criticos:')

    for key, value in result[4].items(): #type: ignore
        print(f'\t{key}: {value}')