# Pi-RecomendacionPeliculas

# Description
Trabajo individual situándose en el rol de un MLOps Engineer para realizar un modelo de recomendación de películas

# Instrucciones de instalación
Requiere las siguientes librerias para su uso

import pandas as pd
import uvicorn
from fastapi import FastAPI
import os
from datetime import datetime
import numpy as np
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ejemplos de uso
Se requiere tener los archivos de las plataformas y 8 archivos.csv donde están los raitings de calificación de los usuarios.

El sistema de recomendación utiliza los siguientes archivos de carga de las plataformas:
Netflix, Amazon, Disney y Hulu. 
Se debe ingresar a la dirección local del API http://127.0.0.1:8000/ para probarlo localmente.

Se tienen 6 consultas para consumir en la aplicación, que son:
•	Película (sólo película, no serie, ni documentales, etc) con mayor duración según año, plataforma y tipo de duración. La función debe llamarse get_max_duration(year, platform, duration_type) y debe devolver sólo el string del nombre de la película:
http://127.0.0.1:8000/get_max_duration/2006/Netflix/min
![image](https://user-images.githubusercontent.com/115653073/232666238-88b784f5-335e-4fd6-b820-965ec0aafe30.png)

En este video se explica el paso a paso para llegar al resultado: https://youtu.be/opdjhz7kN1c 

•	Cantidad de películas (sólo películas, no series, ni documentales, etc) según plataforma, con un puntaje mayor a XX en determinado año. La función debe llamarse get_score_count(platform, scored, year) y debe devolver un int, con el total de películas que cumplen lo solicitado.
127.0.0.1:8000/get_score_count/amazon/3.0/2010
![image](https://user-images.githubusercontent.com/115653073/232666298-4c2fdc01-dd9d-4cd9-a9fd-f81b4f231a6a.png)

•	Cantidad de películas (sólo películas, no series, ni documentales, etc) según plataforma. La función debe llamarse get_count_platform(platform) y debe devolver un int, con el número total de películas de esa plataforma. Las plataformas deben llamarse amazon, netflix, hulu, disney.
127.0.0.1:8000/get_count_platform/disney
![image](https://user-images.githubusercontent.com/115653073/232666342-d71e4210-067f-4ec0-9da5-2fd88257e74b.png)

•	Actor que más se repite según plataforma y año. La función debe llamarse get_actor(platform, year) y debe devolver sólo el string con el nombre del actor que más se repite según la plataforma y el año dado.
127.0.0.1:8000/get_actor/Netflix/2010
![image](https://user-images.githubusercontent.com/115653073/232667258-8a9ca04c-aa12-4e19-b1c0-eb7306d2f8fe.png)

•	La cantidad de contenidos/productos (todo lo disponible en streaming) que se publicó por país y año. La función debe llamarse prod_per_county(tipo,pais,anio) deberia devolver el tipo de contenido (pelicula,serie,documental) por pais y año en un diccionario con las variables llamadas 'pais' (nombre del pais), 'anio' (año), 'pelicula' (tipo de contenido).
127.0.0.1:8000/prod_per_county/Movie/United States/2010
![image](https://user-images.githubusercontent.com/115653073/232667303-ecc2f260-937c-49d2-ab74-d97a8ede6338.png)

•	La cantidad total de contenidos/productos (todo lo disponible en streaming, series, documentales, peliculas, etc) según el rating de audiencia dado (para que publico fue clasificada la pelicula). La función debe llamarse get_contents(rating) y debe devolver el numero total de contenido con ese rating de audiencias.
127.0.0.1:8000/get_contents/tv-ma
![image](https://user-images.githubusercontent.com/115653073/232667353-fbf8f9a2-69cf-47f9-b10c-c7b431dcf720.png)

# Documentación

La concatenación de estos archivos se puede realizar pues todos poseen las mismas columnas y tipos de datos en estas columnas, asi:
 #   Column         Dtype 
0   show_id        object: posee un id tipo string consecutivo anteponiendo "s", ejemplo: s1,s2
 1   type          object: en esta columna se pueden visualizar los tipos de película, existen dos: movie y TV Show
 2   title         object: corresponde al titulo de las películas
 3   director      object: se encuentran los nombres del director de la película.
 4   cast          object: posee los nombres de los actores de las películas.
 5   country       object: están los nombres de los paises donde se creó la película.
 6   date_added    object: corresponde a una fecha que inicialmente esta en formato mes, día, año. Ejemplo: September 24, 2021.
 7   release_year  int64 : se encuentra el año de la versión de la pelicula (numero entero).
 8   rating        object: se refiere a la audiencia que se recomienda para ver la película, ejemplo: general, todos(all), +13
 9   duration      object: en este campo se tiene la duración de la película ( en minutos) o de la serie ( temporadas o season).
 10  listed_in     object: se encuentra la categoría de la película por ejemplo: drama, crimen, action, aventura,etc.
 11  description   object: es un resumen corto de lo que trata la película.
 
 Con estos dataset se pretende encontrar la información detallada de las películas que se desean analizar.
 
 2. Dataset de rating: son una serie de 8 archivos formato csv, en los cuales se encuentran las siguientes columnas:
 #   Column     Dtype  
---  ------     -----  
 0   userId     int64  : en esta columna se encuentra el Id del usuario que califica las películas
 1   rating     float64: en este archivo el rating se refiere a la calificación que le da el usuario a la película
 2   timestamp  object : esta columna posee las fechas en formato UNIX
 3   movieId    object : en esta columna esta la identificación de la película unida al nombre de la plataforma, ejemplo: as1, as2, son de Amazon.
 
 ## ETL
 
 Se procede a realizar un ETL inicial que cumple las siguientes caracteristicas:
 
 En el archivo donde esta la información de las peliculas con su plataforma:
 1. Se crea la nueva columna "id" con la función lambda e insert para identificar cada una de las plataformas y tener un Id que contenga el consecutivo y de acuerdo a la inicial del nombre de la plataforma.
 Esto es: si es Amazon: a, si es Neflix: n, si es Disney: d, si es Hulu: h
 2. Los valores nulos del campo rating deberán reemplazarse por el string “G” (corresponde al maturity rating: “general for all audiences”
 3. Las fechas, se convierten a el formato AAAA-mm-dd
 4. Todos los campos de texto se pasan a minúsculas, sin excepciones
 5. El campo duration se conviente en dos campos: duration_int y duration_type. 
 El primero es un integer y el segundo un string indicando la unidad de medición de duración: min (minutos) o season (temporadas).
 6. Se renombra la columna rating por audiencia para realizar posteriores consultas.
 
 En el archivo de ratings se procede a realizar lo siguiente:
 1. la columna timestamp que esta en formato UNIX se pasa a formato de fecha aaaa-mm-dd
 2. la columna movieId se renombra a id, esto con el fin de poder hacer un merged entre el resultado de la unión de los archivos de las plataformas y ratings.
 3. se calcula el score de cada pelicula en una columna nueva con el fin de poder hacer consultas sobre este campo.
 
 ## merge de archivos
 Al unir todos los archivos se posee un archivo unificado asi:
 Data columns (total 20 columns):
 #   Column         Dtype 		Descripción 
---  ------         -----  ----- ----- ----- ----- ----- 
 0   id             object : posee un id tipo string consecutivo anteponiendo la primera letra de la plataforma, ejemplo A es Amazon: "as", ejemplo: as1,ns2
 1   show_id        object : posee un id tipo string consecutivo anteponiendo "s", ejemplo: s1,s2
 2   type           object : en esta columna se pueden visualizar los tipos de película, existen dos: movie y TV Show
 3   title          object : corresponde al titulo de las películas
 4   director       object : se encuentran los nombres del director de la película.
 5   cast           object : posee los nombres de los actores de las películas.
 6   country        object : están los nombres de los paises donde se creó la película.
 7   date_added     object : corresponde a la fecha en que se adiciono a la plataforma,inicialmente esta en formato mes, día, año. Ejemplo: September 24, 2021.
 8   release_year   object : versión actual  del año de la pelicula(movie) o TV show.
 9   audiencia      object : se refiere a la audiencia que se recomienda para ver la película, ejemplo: general, todos(all), +13
 10  duration       object : en este campo se tiene la duración de la película ( en minutos) o de la serie ( temporadas o season).
 11  listed_in      object : se encuentra la categoría de la película por ejemplo: drama, crimen, action, aventura,etc.
 12  description    object : es un resumen corto de lo que trata la película.
 13  duration_int   float64: es el valor entero de la parte de la columna duration que fue extraido
 14  duration_type  object : es un string correspondiente al string de la columna duration que fue extraido
 15  userId         int64  : en esta columna se encuentra el Id del usuario que califica las películas
 16  scored         float64: es el calculo del promedio del rating por id que dieron los usuarios
 17  timestamp      object : es la fecha en la cual se dio el rating de  la pelicula
 18  plataforma     object : es el nombre de la plataforma y puede ser: netflix, amazon, hulu o disney
 19  anio           int64  : es el año como un numero entero
 
 Al revisar los datos en cada columna se encontró la necesidad de limpiar los datos de una manera adecuada según la información así:
 duration_int se cambia a tipo int64
 los datos que estan en audiencia hay muchos que pertenecen a duration por lo cual se pasa a duration y se vuelve a organizar los datos.
 
 ## Uso del sistema
 Para encontrar el sistema de recomendación se prueba en un browser tipo google crhome o firebox.
 Se ingresa a la dirección 
 
 
 ## EDA
Como Data Engineer, segui los siguientes pasos para realizar un EDA (Exploratory Data Analysis):

1. Para entender la naturaleza de los datos, investigue en el dataframe:
 - Cuántas filas y columnas hay, 
 - cuál es el rango de los valores en cada columna
 - cuántos valores nulos o faltantes hay en cada columna
Esto lo hice mediante el uso de las funciones .shape, .describe() y .info().

2. Luego revise la calidad de los datos:
- Encontre y elimine los valores multiples de espacios de cada columna con texto con la función replace:
el cual aplique a las siguientes columnas: title,director,cast,country,audiencia,description,duration_type
 - Verifique que el tipo de datos de cada columna sea el adecuado:
 se cambia date_added y timestamp a formato fecha
 - se crearon columnas requeridas para posteriores consultas:
 duration_int, duration_type, plataforma, scored y anio.
 - Identifique y trate los datos faltantes, los registros duplicados y los valores atípicos.
 - elimine las columnas que no aportan valor como duration y show_id
 - Investigue las relaciones entre las columnas: 
identificar si hay alguna relación entre las diferentes columnas del DataFrame, 
como la correlación entre la duración de una película y su puntuación promedio, si es muy larga o corta. 
Esto lo pude hacer mediante el uso de funciones como .corr() y .groupby().
3. Visualice los datos: 
para ayudar a comprender mejor los datos, crear gráficos y visualizaciones que muestren patrones y tendencias.
Utilice herramientas de matplotlib y seaborn.
 revisar la distribución de los valores en cada columna,
 - Exploré cada columna individualmente para:
   identificar valores atípicos o errores tipográficos, 
   y decidir si es necesario realizar algún tipo de transformación en los datos para hacerlos más útiles.
- Realice pruebas de hipótesis: con las 6 consultas me ayudo para depurar los datos que se necesitan, ademas mas algunas pruebas estadisticas.
- Haciendo pruebas estadísticas para comprobar si hay evidencia suficiente para respaldar una hipótesis determinada.

- Se documento los resultados paso a paso:
 Me aseguré de registrar todos los hallazgos y conclusiones para facilitar el análisis posterior y permitir
 una mejor comprensión de los datos.
 
 ## EDA
Luego del EDA, siguen los pasos:
Se prepara en local
Se crea la función para consumir el output como otra función de la API https://youtu.be/opdjhz7kN1c video de APIS.
Funcionando en local, se procedió a buscar donde realizar el deployment, con varias alternativas usando Render.
Una vez deployado, se realizan las consultas pedidas desde el deployment.

## Machine learning:
Se decide entrenar nuestro modelo de machine learning para armar un sistema de recomendación de películas utilizando las herramientas de ML.	
Se utiliza un Filtro basado en contenido: esta técnica se basa en características de las películas, como género, actores, director, etc. y recomienda películas similares basadas en las preferencias de los usuarios.
1. Carga de datos:
Se cargan los datos del raiting de los usuarios.Se usa python para la carga del archivo en csv con la libreria Pandas.

ratings_data = pd.read_csv('ratings.csv')
Preprocesamiento de datos:
Después de cargar los datos, los preproceso para eliminar valores nulos o faltantes.
Eliminar variables irrelevantes o duplicadas y transformar los datos para que sean adecuados para su uso en el modelo de filtro colaborativo. 
De acuerdo a la solicitud de recomendación que se requiere de item a item, es decir de contenido, la opción a tomar seria:
Construir el modelo basado en contenido: con base en las columnas listed_in, rating,audiencia, title y descripción usando una matix de similaridad con coseno.
movie_features = combined_data.pivot_table(index='audiencia', columns='listed_in', values='rating').fillna(0)
movie_features = movie_features.to_numpy()
similarity_matrix = cosine_similarity(movie_features)
