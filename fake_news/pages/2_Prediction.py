
import streamlit as st
import joblib  # Utilisation de joblib pour charger le modèle
import plotly.graph_objects as go

# Fonction pour afficher la probabilité de prédiction
def plot_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Fake News (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig.update_layout(height=300, margin={'t': 50, 'b': 0, 'l': 0, 'r': 0})
    return fig

# Charger le modèle de régression logistique
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_regression_model.pkl")  # Charger le modèle enregistré
        return model
    except FileNotFoundError:
        st.error("Le fichier 'logistic_regression_model.pkl' n'a pas été trouvé.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# Vérification du modèle
if model is not None:
    st.write(f"**Modèle chargé :** {type(model)}")

# Charger le vectorizer TF-IDF utilisé lors de l'entraînement
@st.cache_resource
def load_vectorizer():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Charger le vectorizer TF-IDF
        return vectorizer
    except FileNotFoundError:
        st.error("Le fichier 'tfidf_vectorizer.pkl' n'a pas été trouvé.")
        return None

vectorizer = load_vectorizer()

# Vérifier si le vectorizer est chargé
if vectorizer is None:
    st.stop()

# Prétraitement du texte
def preprocess_text(text):
    # Appliquer le prétraitement (en minuscules, suppression des stopwords, lemmatisation)
    text = text.lower()  # Convertir en minuscules
    # Ajoute ici la logique de suppression des stopwords, lemmatisation, etc.
    return text

# Interface utilisateur pour la prédiction
st.title("Prédiction - Fake News")
st.write("Entrez un article pour prédire s'il s'agit d'une Fake News.")

# Zone de saisie de l'article
article_text = st.text_area("Entrez l'article ici", height=200)

# Ajouter un bouton pour valider la saisie
if st.button("Valider l'article et prédire"):
    # Si l'utilisateur entre un texte, faire la prédiction
    if article_text:
        processed_text = preprocess_text(article_text)

        # Vectoriser l'article
        vectorized_text = vectorizer.transform([processed_text])

        # Prédiction avec le modèle
        prediction = model.predict(vectorized_text)
        prob_fake_news = model.predict_proba(vectorized_text)[0][1]  # Probabilité de Fake News

        # Afficher la probabilité
        gauge_fig = plot_gauge(prob_fake_news)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Afficher le résultat
        st.write(f"**Prédiction :** {'Fake News' if prediction[0] == 1 else 'Article Réel'}")
        st.write(f"**Probabilité de Fake News :** {prob_fake_news * 100:.2f}%")
    else:
        st.write("Veuillez entrer un article pour effectuer la prédiction.")
