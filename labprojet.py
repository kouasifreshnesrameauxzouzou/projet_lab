import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Charger les données
data = pd.read_csv("C:\\Users\\kouas\\Desktop\\ACTIVE\\labphase\\Bank_Personal_Loan_Modelling.csv")


# Titre de l'application
st.title('Analyse des prêts personnels')

# Afficher les 3 premières lignes des données
st.subheader('Aperçu des données')
st.write(data.head(3))

# Afficher les informations sur les données
st.subheader('Informations sur les données')
st.write(data.info())

# Afficher les statistiques descriptives des données
st.subheader('Statistiques descriptives')
st.write(data.describe().transpose())

# Vérifier les valeurs manquantes
st.subheader('Valeurs manquantes')
st.write(data.isnull().sum())

# Afficher le graphique de distribution de la variable cible "Personal Loan" avec Plotly Express
st.subheader('Distribution de la variable cible')
fig_dist = px.histogram(data, x='Personal Loan')
st.plotly_chart(fig_dist)

# Afficher le graphique de la relation entre l'âge et le revenu avec Plotly Express
st.subheader('Relation entre l\'âge et le revenu')
fig_age_income = px.scatter(data, x='Age', y='Income', color='Personal Loan')
st.plotly_chart(fig_age_income)

# Afficher le graphique de la relation entre l'éducation et le montant du prêt hypothécaire avec Plotly Express
st.subheader('Relation entre l\'éducation et le montant du prêt hypothécaire')
fig_edu_mortgage = px.box(data, x='Education', y='Mortgage')
st.plotly_chart(fig_edu_mortgage)

# Séparation des données en ensembles d'entraînement et de test
X = data.drop(['ID', 'Personal Loan'], axis=1)
y = data['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement d'un modèle de forêt aléatoire
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

st.subheader('Évaluation du modèle')
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_rep)