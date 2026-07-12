import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==========================
# Génération des données
# ==========================

np.random.seed(42)

n = 300

jours = np.random.randint(0, 7, n)
mois = np.random.randint(1, 13, n)
temperature = np.random.randint(10, 38, n)
pluie = np.random.randint(0, 2, n)
vacances = np.random.randint(0, 2, n)
heure = np.repeat(8, n)

voyageurs = (
    350
    + (5 - jours) * 25
    - pluie * 80
    - vacances * 60
    + temperature * 4
    + np.random.randint(-40, 40, n)
)

data = pd.DataFrame({
    "Jour": jours,
    "Mois": mois,
    "Heure": heure,
    "Temperature": temperature,
    "Pluie": pluie,
    "Vacances": vacances,
    "Voyageurs": voyageurs
})

# ==========================
# Entraînement du modèle
# ==========================

X = data.drop("Voyageurs", axis=1)
y = data["Voyageurs"]

modele = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

modele.fit(X, y)

# ==========================
# Interface Streamlit
# ==========================

st.set_page_config(page_title="Prédiction Voyageurs ETO", layout="centered")

st.title("🚌 Prédiction du nombre de voyageurs")
st.subheader("Station Place d'Armes - ETO Oran")

jour = st.selectbox(
    "Jour",
    [0,1,2,3,4,5,6],
    format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x]
)

mois = st.slider("Mois",1,12,7)

temperature = st.slider("Température (°C)",10,45,28)

pluie = st.selectbox("Pluie",["Non","Oui"])

vacances = st.selectbox("Vacances scolaires",["Non","Oui"])

if st.button("Prédire"):

    entree = pd.DataFrame({
        "Jour":[jour],
        "Mois":[mois],
        "Heure":[8],
        "Temperature":[temperature],
        "Pluie":[1 if pluie=="Oui" else 0],
        "Vacances":[1 if vacances=="Oui" else 0]
    })

    prediction = modele.predict(entree)

    st.success(f"Nombre estimé de voyageurs : {round(prediction[0])}")

    if prediction[0] > 500:
        st.warning("⚠️ Forte affluence : prévoir davantage de bus.")
    elif prediction[0] > 420:
        st.info("🚌 Affluence moyenne.")
    else:
        st.success("✅ Affluence faible.")