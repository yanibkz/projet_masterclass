import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Accueil - Détection des Fake News", layout="wide")

# Styles CSS personnalisés
st.markdown(
    """
    <style>
    .header {
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .section {
        background-color: #f2f2f2;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal avec style
st.markdown("""
    <div class="header">
        <h1>Bienvenue sur le Dashboard : Détection des Fake News</h1>
    </div>
    """, unsafe_allow_html=True)

# Description du projet dans une section stylisée
st.markdown("""
    <div class="section">
        <h2>À propos de ce projet</h2>
        <p>
            Ce <strong>dashboard interactif</strong> a été conçu pour détecter les <strong>Fake News</strong> en utilisant un modèle de régression logistique. Il permet de prédire si un article est une Fake News ou non en fonction de ses caractéristiques textuelles.
        </p>
        <h3>Fonctionnalités principales</h3>
        <ul>
            <li><strong>Prédiction de Fake News</strong> : Affiche la probabilité qu'un article soit une Fake News.</li>
            <li><strong>Visualisation des résultats</strong> : Affiche des graphiques pour mieux comprendre les prédictions.</li>
        </ul>
        <h3>Objectifs du Dashboard</h3>
        <p>
            Ce dashboard vise à fournir une interface simple et accessible pour prédire et comprendre les caractéristiques des Fake News. 
        </p>
    </div>
    """, unsafe_allow_html=True)
