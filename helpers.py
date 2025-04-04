# helpers.py
import pandas as pd
import numpy as np
import datetime as dt
import re
import feedparser
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from google_play_scraper import reviews, app, Sort
from sentence_transformers import SentenceTransformer, util
from pysentimiento import create_analyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configuración de Seaborn (si se desea usar en helpers, o importarlo en el notebook)
import seaborn as sns
sns.set(style='whitegrid')

# Global: Modelo de embeddings
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# Global: Definición de keywords para bugs y features
keywords = {
    "bug": ["error", "problemas", "bug", "no abre", "no funciona", "no me deja", "no puedo", "no anda", "no carga"],
    "feature": [
        "sería bueno", "me gustaría que tenga", "necesito que agreguen", "falta", "sumen", "es mejor",
        "prefiero", "sería genial si agregaran", "quisiera que se incluya", "necesito que ofrezcan"
    ]
}

# Pre-codificar las keywords
keyword_embeddings = {k: model.encode(v, convert_to_tensor=True) for k, v in keywords.items()}

# Global: Seed words para clasificación de tópicos de bugs
topic_seeds = {
    "Acceso": [
        "acceso", "ingresar", "login", "entrar", "problema de ingreso", "no puedo acceder", "no ingreso", "pin", "correo"
    ],
    "Transacciones": [
        "dinero trabado", "saldo bloqueado", "fondos retenidos", "dinero congelado", "dinero retenido",
        "transacción pendiente", "no llega", "no se refleja el pago", "dinero perdido", "transferencia fallida", "pago rechazado"
    ],
    "Cuenta": [
        "cuenta bloqueada", "límites", "validación", "cuenta desactivada", "cuenta deshabilitada"
    ],
    "CC": [
        "no contestan", "no responden", "soporte", "atención", "atención", "contactar"
    ]
}

######################
# Funciones de Helpers
######################

def preprocess_text(text):
    """
    Preprocesa un texto: lo pasa a minúsculas, elimina caracteres no alfabéticos
    (conservando acentos y ñ) y reduce espacios múltiples.
    """
    text = text.lower()
    text = re.sub(r"[^a-záéíóúñü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_playstore_reviews(app_id, lang='es', country='AR', days_back=90):
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=2000
    )
    df = pd.DataFrame(result)
    df['date'] = pd.to_datetime(df['at'])
    df.rename(columns={'score': 'rating'}, inplace=True)
    cutoff = dt.datetime.now() - dt.timedelta(days=days_back)
    return df[df['date'] >= cutoff]

def get_itunes_reviews(app_store_id, days_back=90):
    url = f"https://itunes.apple.com/rss/customerreviews/id/{app_store_id}/json"
    feed = feedparser.parse(url)
    reviews_list = []
    for entry in feed.entries[1:]:
        if 'im_rating' in entry:
            try:
                rating = int(entry['im_rating'])
            except:
                rating = None
        else:
            rating = None
        title = entry.get('title', '')
        if 'content' in entry and len(entry.content) > 0:
            content = entry.content[0].value
        else:
            content = entry.get('summary', '')
        review_date = pd.to_datetime(entry.get('updated'))
        reviews_list.append({
            'date': review_date,
            'rating': rating,
            'title': title,
            'content': content
        })
    df = pd.DataFrame(reviews_list)
    if df.empty or "date" not in df.columns:
        return df
    cutoff = dt.datetime.now() - dt.timedelta(days=days_back)
    df = df[df['date'] >= cutoff]
    return df

def extract_topics(reviews, n_topics=5, n_top_words=10, min_df=2, max_df=0.95):
    """
    Extrae los temas más recurrentes de un listado de reviews utilizando LDA.
    
    Parámetros:
      reviews (list o pd.Series): Lista o serie de textos de reviews.
      n_topics (int): Número de temas a extraer.
      n_top_words (int): Número de palabras clave que se mostrarán por tema.
      min_df (int): Frecuencia mínima para que una palabra se incluya.
      max_df (float): Fracción máxima de documentos en la que una palabra puede aparecer.
      
    Retorna:
      topics (dict): Diccionario de temas con sus palabras clave.
      weights (dict): Peso promedio de cada tema en el conjunto de reviews.
    """
    preprocessed_reviews = [preprocess_text(text) for text in reviews if isinstance(text, str) and text.strip() != '']
    if not preprocessed_reviews:
        return {}, {}
    spanish_stopwords = stopwords.words('spanish')
    vectorizer = CountVectorizer(stop_words=spanish_stopwords, min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(preprocessed_reviews)
    if X.shape[1] == 0:
        return {}, {}
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f"Tema {topic_idx+1}"] = top_features
    topic_distributions = lda.transform(X)
    avg_weights = topic_distributions.mean(axis=0)
    weights = {f"Tema {i+1}": avg_weights[i] for i in range(n_topics)}
    return topics, weights

def analyze_sentiment(text):
    """
    Analiza el sentimiento de un texto usando pysentimiento (modelo BETO para español).
    """
    analyzer = create_analyzer(task="sentiment", lang="es")
    if not text or not isinstance(text, str):
        return None
    result = analyzer.predict(text)
    return result.output

def classify_keywords_with_sentiment(row, threshold=0.6, bug_strict_threshold=0.7, feature_strict_threshold=0.75):
    """
    Evalúa la similitud entre una review y frases clave, combinándola con el sentimiento precomputado
    (en la columna 'sentiment') para ajustar umbrales.
    
    Para bugs:
      - Si el sentimiento es NEG, se permite un umbral ligeramente menor (threshold - 0.1).
      - Si el sentimiento es POS, se exige un umbral mayor (bug_strict_threshold).
      - Si es NEU, se usa el umbral base.
      
    Para feature requests:
      - Si el sentimiento es POS, se exige un umbral mayor (feature_strict_threshold).
      - En caso contrario, se usa el umbral base.
    
    Además, si la review es genérica (por ejemplo, "excelente" o "muy mala") o es extremadamente corta,
    se devuelve False para ambas categorías.
    
    Retorna una Series con dos valores booleanos: "is_bug" e "is_feature".
    """
    text = row["content"]
    sentiment_label = row["sentiment"]
    if not text or not isinstance(text, str) or not sentiment_label:
        return pd.Series({"is_bug": False, "is_feature": False})
    cleaned_text = text.strip().lower()
    generic_praise = {"excelente", "muy bueno", "perfecto", "genial", "estupendo"}
    generic_negative = {"muy mala", "malisima", "pésima", "terrible", "horrible"}
    if cleaned_text in generic_praise or cleaned_text in generic_negative or len(cleaned_text.split()) <= 2:
        return pd.Series({"is_bug": False, "is_feature": False})
    sentiment_label = sentiment_label.upper()
    text_embedding = model.encode(text, convert_to_tensor=True)
    scores = {}
    for category, emb_list in keyword_embeddings.items():
        sim = util.cos_sim(text_embedding, emb_list)
        scores[category] = sim.max().item()
    if sentiment_label == "NEG":
        bug_threshold = threshold - 0.1
    elif sentiment_label == "POS":
        bug_threshold = bug_strict_threshold
    else:
        bug_threshold = threshold
    is_bug = scores["bug"] > bug_threshold
    if sentiment_label == "POS":
        feature_threshold = feature_strict_threshold
    else:
        feature_threshold = threshold
    is_feature = scores["feature"] > feature_threshold
    return pd.Series({"is_bug": is_bug, "is_feature": is_feature})

def classify_review_topic(text, threshold=0.5):
    """
    Clasifica una review en un tópico de bugs basado en seed words y embeddings.
    
    Retorna el tópico asignado o "Otros" si no se supera el umbral.
    """
    if not text or not isinstance(text, str):
        return "No Clasificado"
    text_embedding = model.encode(text, convert_to_tensor=True)
    best_topic = "Otros"
    best_sim = 0.0
    for topic, seeds in topic_seeds.items():
        seeds_embedding = model.encode(seeds, convert_to_tensor=True)
        sim = util.cos_sim(text_embedding, seeds_embedding).max().item()
        if sim > best_sim:
            best_sim = sim
            best_topic = topic
    if best_sim >= threshold:
        return best_topic
    else:
        return "Otros"