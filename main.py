# en la linea de comandos se ejecuta desde esta ruta local: cd.. E:  cd DataScience\Labs\PI-1\fastapi
#E:/DataScience/Labs/PI-1/fastapi
# inicializar en cdm con uvicorn main:app --reload

import pandas as pd
import uvicorn
import json
from fastapi import FastAPI
import os
from datetime import datetime
import numpy as np
from flask import Flask, render_template
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from fastapi.templating import Jinja2Templates
from main import app

# Crear una instancia del motor de plantillas Jinja2
templates = Jinja2Templates(directory="templates")

## Instancio FastApi
app = FastAPI()

# para trabajar el entorno virtual
gunicorn main:app

datos_cargados =0
#final_df=pd.DataFrame()

def json_serial(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Tipo de dato no serializable")

# Función para convertir fechas
def convertir_fecha(fecha):
    try:
        return datetime.strptime(fecha, '%B %d, %Y').strftime('%Y-%m-%d')
    except:
        return pd.NaT

def etl_plataformas_csv():
    netflix_df = pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/netflix.csv')
    amazon_df = pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/amazon.csv')
    hulu_df = pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/hulu.csv')
    disney_df = pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/disney.csv')

    # creo la nueva columna "id" para identificar plataformas
    netflix_df.insert(0, 'id', netflix_df['show_id'].apply(lambda x: 'n' + x))
    amazon_df.insert(0, 'id', amazon_df['show_id'].apply(lambda x: 'a' + x))
    hulu_df.insert(0, 'id', hulu_df['show_id'].apply(lambda x: 'h' + x))
    disney_df.insert(0, 'id', disney_df['show_id'].apply(lambda x: 'd' + x))

    # Verificar si los dataframes tienen las mismas columnas
    if set(amazon_df.columns).issubset(set(netflix_df.columns)) and set(hulu_df.columns).issubset(set(netflix_df.columns)) and set(disney_df.columns).issubset(set(netflix_df.columns)):
        # Unir los dataframes si tienen las mismas columnas
        peliculas_df = pd.concat([netflix_df, amazon_df, hulu_df, disney_df], ignore_index=True)
        peliculas_df = peliculas_df.astype(str)
        #print(peliculas_df.info())
    else:
        return {'error 750': "Los dataframes no tienen las mismas columnas" }
    
    # Encontrar columnas con valores NaN
    columnas_con_nan = peliculas_df.columns[peliculas_df.isna().any()].tolist()
    
    registros_con_nan = peliculas_df[peliculas_df['rating'].isna()]

    # Reemplazamos los valores nulos del campo "rating" por "G"
    peliculas_df['rating'] = peliculas_df['rating'].fillna('G')
    peliculas_df = peliculas_df.rename(columns={'rating': 'audiencia'})
    # Crear las nuevas columnas y extraer los valores
    # Extraer la parte numérica en la columna "duration_int"
    peliculas_df['duration_int'] = peliculas_df['duration'].str.extract('(\d+)', expand=False)

    # Extraer la parte string en la columna "duration_type"
    peliculas_df['duration_type'] = peliculas_df['duration'].str.extract('([a-zA-Z]+)', expand=False)

    # Convertir la columna 'duration_int' en integer
    peliculas_df['duration_int'] = pd.to_numeric(peliculas_df['duration_int'])
    # pasar a minusculas las cadenas de string
    peliculas_df = peliculas_df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
    # Convertir la columna a formato de fecha
    peliculas_df['date_added'] = peliculas_df['date_added'].apply(convertir_fecha)
    # Crear una lista para almacenar todos los dataframes
    ratings_df=pd.DataFrame()
    total_df=pd.DataFrame()
    # Recorrer los archivos CSV y leerlos en dataframes
    # Obtener la ruta base del archivo
    ruta_base = r'E:/DataScience/Labs/PI-1/fastapi/'  
    for i in range(1, 9):
        ruta_archivo = os.path.join(ruta_base, f'{i}.csv')
        ratings_df = pd.read_csv(ruta_archivo)
        total_df=  total_df.append(ratings_df)  
        #print("*******************************************len del totl dentro for ***** ", len(total_df), "paso : " , i)
    # Renombrar la columna 'movieId1 como id en los dataset de ratings
    total_df = total_df.rename(columns={'movieId': 'id'})
    
    # cambiar tipo de columna
    total_df['timestamp'] = pd.to_datetime(total_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
  
    #ratings_df = ratings_df.rename(columns={'timestamp': 'date_added'})
    
    # se deja como dataframe final df en df uniendo los dos de peliculas y ratings por el id que corresponde a moviedId
    # Fusionar los DataFrames por la columna 'id'
    #print("*******************************************antes del merge total_df ***** ", len(total_df))
    df = pd.merge(peliculas_df, total_df, on='id')

    #print("*******************************************despues merge df***** ", len(df))
    # calculo de score en rating por id obteniendo el promedio por titulo de la pelicula   
   
    df.groupby(['title', 'rating']).agg({'rating': 'mean'}) 
    df.rename(columns={'rating': 'scored'}, inplace=True) 
  
    df['plataforma'] = df['id'].apply(lambda x: 'amazon' if x.startswith('as') else 'netflix' if x.startswith('ns') else 'disney' if x.startswith('ds') else 'hulu' if x.startswith('hs') else 'No soportada')
    df['anio'] = pd.to_datetime(df['timestamp']).dt.year
    df.to_csv('archivo_final.csv', index=False)
    print("/************************************************* termino proceso etl:", len(df))
    print(df.info())
    #print(df.head(3))
    return df



@app.get("/")
def read_root():
    # df= etl_plataformas_csv()
    return {"message": "Iniciando: Sistema de Recomendación de Plataformas"}

# consulta 1 •	Película (sólo película, no serie, ni documentales, etc) con mayor duración según año, plataforma y tipo de duración. La función debe llamarse get_max_duration(year, platform, duration_type) y debe devolver sólo el string del nombre de la película.
@app.get('/get_max_duration/{anio}/{plataforma}/{dtype}')
def get_max_duration(anio: int, plataforma: str, dtype: str):
    type = "movie"
    plataforma = plataforma.lower()
    dtype = dtype.lower()
    final_df= etl_plataformas_csv()
    
    resultado = final_df.loc[final_df['anio'] == anio]
    if resultado.empty:
        return {'error 700': "El año no existe en la base de datos de la plataforma: " + str(anio)}
    elif plataforma not in final_df['plataforma'].values:
        return {'error 710': "La plataforma no está en las plataformas: " + plataforma}
    elif dtype not in final_df['duration_type'].values:
        return {'error 720': "El tipo de duración no existe en la base de datos: " + dtype}
    else:
        max_duration_idx = resultado['duration_int'].idxmax()
        respuesta = resultado.loc[max_duration_idx, 'title']
        return {'pelicula': respuesta}
    return respuesta

#consulta 2 •	Cantidad de películas (sólo películas, no series, ni documentales, etc) según plataforma, con un puntaje mayor a XX en determinado año. La función debe llamarse get_score_count(platform, scored, year) y debe devolver un int, con el total de películas que cumplen lo solicitado.
@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, anio: int):
    plataforma = plataforma.lower()
    final_df= etl_plataformas_csv()
    num_registros = len(final_df)
    #print("************************************************ inicia consulta 2", num_registros)
    # Filtra las películas que cumplen las condiciones de plataforma, año, tipo y puntaje promedio
    df_filtered = final_df[(final_df['plataforma'] == plataforma) & (final_df['anio'] == anio) & (final_df['type'] == 'movie') & (final_df['scored'] > scored)]
    
    # Cuenta el número total de películas filtradas
    respuesta_filtrada= df_filtered.count()['id']
    #print("************************************************ finaliza consulta 2 cantidad de peliculas", respuesta)
    resultado= {
        'plataforma': plataforma,
        'cantidad': respuesta_filtrada,
        'anio': anio,
        'score':scored
    }
    respuesta = json.dumps(resultado, default=json_serial)

    #print(" *********************** el resulatado es ", resultado, " en json es: ", resultado_json)
    
    return respuesta

#consulta 3 •	Cantidad de películas (sólo películas, no series, ni documentales, etc) según plataforma. La función debe llamarse get_count_platform(platform) y debe devolver un int, con el número total de películas de esa plataforma. Las plataformas deben llamarse amazon, netflix, hulu, disney.
@app.get('/get_count_platform/{plataforma}')
def get_count_platform(plataforma: str):
    #print("************************************************  inicia consulta 3")
    final_df= etl_plataformas_csv()
    plataforma=plataforma.lower()
    
    # Filtra las películas por plataforma y tipo de película
    filtered_df = final_df.loc[(final_df['plataforma'] == plataforma) & (final_df['type'] == 'movie')]
    # Cuenta el número de películas filtradas
    respuesta = filtered_df.shape[0]
    #print("**************************************cantidad de peliculas ********** ", respuesta)
    return {'plataforma': plataforma, 'peliculas': respuesta}

#consulta 4 •	Actor que más se repite según plataforma y año. La función debe llamarse get_actor(platform, year) y debe devolver sólo el string con el nombre del actor que más se repite según la plataforma y el año dado.
@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(plataforma: str, anio: int):
    
    final_df= etl_plataformas_csv()
    # Se Filtra el DataFrame por plataforma y año
    filtered_df = final_df[(final_df['plataforma'] == plataforma) & (final_df['anio'] == anio) & (final_df['cast'].notna()) & (final_df['cast'] != '')]
     # Se Cuenta la frecuencia de cada actor en el grupo filtrado
    actor_counts = filtered_df['cast'].value_counts()
    # Se Filtra los NaN de actor_counts
    actor_counts = actor_counts[actor_counts.notnull()]
    # Se selecciona el actor con la frecuencia máxima
    respuesta = actor_counts.idxmax()
    # Se Verifica si actor_counts está vacío
    if actor_counts.empty:
        respuesta = None
    else:
        respuesta = actor_counts.idxmax()
    return {
        'plataforma': plataforma,
        'anio': anio,
        'actor': respuesta,
        'apariciones': int(actor_counts.loc[respuesta])
    }

# consulta 5 •	La cantidad de contenidos/productos (todo lo disponible en streaming) que se publicó por país y año. La función debe llamarse prod_per_county(tipo,pais,anio) deberia devolver el tipo de contenido (pelicula,serie,documental) por pais y año en un diccionario con las variables llamadas 'pais' (nombre del pais), 'anio' (año), 'pelicula' (tipo de contenido).
@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_county(tipo: str, pais: str, anio: int):
    pais = pais.lower()
    tipo = tipo.lower()
    #print("************************************************ inicia consulta 5 filtro ", tipo)
    final_df= etl_plataformas_csv()
    # Filtra el DataFrame original para obtener sólo los datos del tipo de contenido, país y año especificados
    filtered_df = final_df[(final_df['country'] == pais) & (final_df['type'] == tipo) & (final_df['release_year'] == anio)]
    
    # Cuenta el número de productos publicados por país y año
    respuesta = len(filtered_df)
    #print("************************************************ fin consulta 5 firo ", respuesta) 
    return {'pais': pais, 'anio': anio, 'peliculas': respuesta}

#consulta 6 •	La cantidad total de contenidos/productos (todo lo disponible en streaming, series, documentales, peliculas, etc) según el rating de audiencia dado (para que publico fue clasificada la pelicula). La función debe llamarse get_contents(rating) y debe devolver el numero total de contenido con ese rating de audiencias.
@app.get('/get_contents/{rating}')
def get_contents(rating: str):
    rating=rating.lower()
    final_df= etl_plataformas_csv()
        
    # Se realiza el firltro de los datos por rating
    filtered_df = final_df[final_df['audiencia'] == rating]

    # Se obtienen los campos necesarios para la respuesta
    respuesta = filtered_df[['title', 'description', 'type']].to_dict('records')

    return {'rating': rating, 'contenido': respuesta}









#consulta 7
@app.get('/get_recomendation/{title}')
def get_recomendation(title):
    df = pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/archivo_final.csv')
    modelo_df= pd.read_csv('E:/DataScience/Labs/PI-1/fastapi/archivo_final.csv')
    
    
    return {'recomendacion':respuesta}

    
    
   

def main():
    if datos_cargados ==0:
        #final_df= etl_plataformas_csv()
        datos_cargados=1
