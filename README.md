# Análisis de Reviews de Apps

Este repositorio contiene un Jupyter Notebook para analizar reviews de apps extraídas de Google Play y App Store (utilizando el RSS de iTunes). El objetivo del proyecto es:

•	**Obtener reviews** de múltiples apps utilizando el scraper de Google Play y feedparser para iTunes.

•	**Analizar el sentimiento** de las reviews empleando el modelo BETO a través de Transformers.

•	**Clasificar las reviews** en *bugs* y *feature requests* combinando embeddings y análisis de sentimiento.

•	**Extraer tópicos recurrentes** de las reviews mediante Latent Dirichlet Allocation (LDA).

•	**Visualizar resultados**: distribución de sentimientos, evolución semanal, comparación entre rating y sentimiento, y WordClouds por sentimiento.

•	**Guardar resultados** en archivos CSV.

## Requerimientos

El proyecto fue desarrollado con Python 3.9.6 y utiliza las siguientes librerías:

•	google-play-scraper

•	sentence-transformers

•	matplotlib

•	seaborn

•	pandas

•	feedparser

•	transformers

•	nltk

•	scikit-learn

•	wordcloud

Puedes instalar todas las dependencias ejecutando:

    pip install google-play-scraper sentence-transformers matplotlib seaborn pandas feedparser transformers nltk scikit-learn wordcloud

## Uso
1.	Clona o descarga el repositorio y abre el notebook `app_reviews_analysis.ipynb` en Jupyter Notebook.

2.	Ejecuta las celdas en orden. El notebook se organiza en secciones:

	•	**Instalación y configuración**: se instalan los requerimientos y se importan las librerías.

	•	**Obtención de Reviews**: funciones para extraer reviews de Google Play y App Store para las apps definidas en un diccionario.

	•	**Análisis de Sentimiento**: se utiliza el modelo BETO para asignar un sentimiento a cada review.

	•	**Clasificación de Reviews**: se combinan embeddings y análisis de sentimiento para clasificar las reviews en bugs y feature requests.

	•	**Extracción de Tópicos**: se aplica LDA para identificar los temas más recurrentes, segmentados por sentimiento.

	•	**Visualizaciones**: se generan gráficos y word clouds para explorar los datos.

	•	**Exportación de Resultados**: se guardan los resultados en archivos CSV.

## Estructura del Notebook

•	**Instalación de Requerimientos y Configuración Inicial**:

Instala las dependencias y configura el entorno.
 
•	**Obtención de Reviews**:

Funciones `get_playstore_reviews` y `get_itunes_reviews` para extraer y consolidar reviews de ambas tiendas.

•	**Análisis de Sentimiento**:

 Uso de Transformers con el modelo BETO para determinar el sentimiento (POS, NEG, NEU) de cada review.

•	**Clasificación de Reviews**:

 Función `classify_keywords_with_sentiment` que utiliza embeddings para comparar cada review con frases clave y asignar etiquetas de bug o feature según el sentimiento.

•	**Extracción de Tópicos**:

 Uso de LDA para extraer los temas más recurrentes de las reviews y segmentarlos por sentimiento.

•	**Visualizaciones**:

 Gráficos de barras, líneas, boxplots, scatterplots y word clouds para visualizar la distribución y evolución de los datos.

•	**Exportación de Resultados**:

 Guardado de los DataFrames finales en archivos CSV para análisis posterior.

## Personalización

•	**Modificar Apps a Analizar**:
	
 Edita el diccionario apps en el notebook para agregar o quitar apps según necesites.

•	**Ajustar Parámetros**:
	
 Puedes ajustar umbrales de similitud, parámetros de LDA y otros valores para afinar el análisis.
