import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analisis_rapido(df, n=5):

    """Funcion que proporciona un analisis rapido del DataFrame

    Parametros:
    df: DataFrame
    n: numero de filas (por defecto =5)
    """
    print(f"Las {n} primeras columnas son:")
    display(df.head(n))
    print(f"Informacion básica del dataframe:")
    display(df.info())

    print(f"El número de duplicados es: {df.duplicated().sum()}")
    display(df.isnull().mean().round(4)*100)



def eda(df, n=2):

    """Funcion que proporciona un EDA rápido
    """
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['O','category']).columns

    print("Variables numéricas:\n\n", num_cols)
    print("\nVariables categóricas:\n\n", cat_cols)

    print("Veamos las estadísticas básicas: \n")
    display(df.describe().T.round(n))
    display(df.describe(include=['O','category']).T.round(n))

    for col in cat_cols:
        print(f" \n------------ ESTAMOS ANALIZANDO LA COLUMNA: '{col}' ----------\n")
        print(f"Valores únicos {df[col].unique()}\n")
        print("Frecuencias de los valores únicos de las categorias:")
        display(df[col].value_counts())
    
    print("Countplot de las columnas categóricas:\n")
    for col in cat_cols:
        if df[col].nunique() > 200:
            print(f"Columna {col} tiene demasiadas categorías: {df[col].nunique()}\n\n")
            continue
        
        num_categories = df[col].nunique()
        width = max(7, num_categories * 0.5)
        height = 3

        plt.figure(figsize=(width, height))
        sns.countplot(x=df[col], order=df[col].value_counts().index)

        plt.title(f"Gráfico de barras de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=90)

        plt.show()
    
    print("Histogramas:\n")
    for col in num_cols:
        plt.figure(figsize=(10,5))
        sns.histplot(df[col], bins=30, edgecolor='black')

        plt.title(f"Distribucion de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        
        plt.show()

    print("Vayamos con los boxplot:\n")
    for col in num_cols:
        plt.figure(figsize=(10,1))
        sns.boxplot(x=df[col])

        plt.title(f"Distribucion de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        
        plt.show()

def matriz_correlacion(df):

    # Calcular matriz de correlacion
    corr_matrix = df.corr(numeric_only=True)

    #Crear la figura
    plt.figure(figsize=corr_matrix.shape)

    #Crear una máscara para mostrar solo la parte triangular
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    #Grafica del mapa de calor
    sns.heatmap(corr_matrix,
                annot=True,
                vmin=-1,
                vmax=1,
                mask=mask,
                cmap='cool')
    plt.show()