import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from math import pi

# Charger le modèle et le scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Prédiction du Cancer du Sein", layout="wide")

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Aller à :", ["Accueil", "Prédiction", "À propos"])

# ----------- PAGE ACCUEIL -----------
if page == "Accueil":
    st.title("🎗️ Application de Prédiction du Cancer du Sein")
    st.markdown("""
    Cette application permet de prédire si une tumeur est **bénigne** ou **maligne** à partir de 30 caractéristiques cliniques.  
    Elle utilise un modèle **XGBoost** entraîné sur les données du **Wisconsin Breast Cancer Dataset**.

    👉 Allez dans l'onglet *Prédiction* pour tester une nouvelle entrée.
    """)
    st.image("image.png", width=550)

# ----------- PAGE PRÉDICTION -----------
elif page == "Prédiction":
    st.title("🧪 Saisie des caractéristiques cliniques")

    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    user_input = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            val = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0, format="%.5f")
            user_input.append(val)

    if st.button("Prédire"):
        input_array = np.array([user_input])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]

        st.markdown("---")
        st.subheader("🧾 Résultat de la prédiction")
        if prediction == 1:
            st.error(f"⚠️ Tumeur **MALIGNE** — Probabilité : {proba[1]*100:.2f}%")
        else:
            st.success(f"✅ Tumeur **BÉNIGNE** — Probabilité : {proba[0]*100:.2f}%")

        st.subheader("📋 Données saisies")
        df_user = pd.DataFrame([user_input], columns=features)
        st.dataframe(df_user.T.rename(columns={0: "Valeurs saisies"}))

        # Barplot
        st.subheader("📈 Visualisation : 10 premières variables")
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.barh(features[:10], user_input[:10], color="orchid")
        ax1.set_title("Valeurs cliniques principales")
        st.pyplot(fig1)

        # Radar chart
        st.subheader("🕸️ Profil radar (6 variables clés)")
        radar_vars = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'symmetry_mean']
        radar_values = [df_user[var][0] for var in radar_vars]
        radar_values += radar_values[:1]
        angles = [n / float(len(radar_vars)) * 2 * pi for n in range(len(radar_vars))]
        angles += angles[:1]
        fig2, ax2 = plt.subplots(subplot_kw={'polar': True})
        ax2.plot(angles, radar_values, linewidth=2, linestyle='solid', color='crimson')
        ax2.fill(angles, radar_values, 'crimson', alpha=0.3)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_vars)
        ax2.set_title("Profil clinique du patient")
        st.pyplot(fig2)

# ----------- PAGE À PROPOS -----------
elif page == "À propos":
    st.title("ℹ️ À propos")
    st.markdown("""
    **Étudiante :** Kouamé Mahan Amena Grâce Emmanuella  
    **Encadrant :** Coulibaly Aboubacar  
    **Établissement :** ITA Deux-Plateaux  
    **Filière :** Licence Professionnelle en Sciences Informatiques  
    **Projet :** Prédiction du cancer du sein par le Machine Learning  
    """)
