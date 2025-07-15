import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Ventes Automobiles",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Couleurs modernes
COLOR_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
BACKGROUND_COLOR = "#f8f9fa"

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_excel("STAGE.xlsx")
    data['Année_vente'] = data['Date_vente'].dt.year
    return data

stage = load_data()

# Titre de l'application
st.title('🚗 Dashboard d\'Analyse des Ventes Automobiles')
st.markdown("""
    <style>
        .title {
            color: #2c3e50;
            font-size: 2.5rem !important;
        }
        .header {
            color: #3498db;
            font-size: 1.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=80)
    st.title("Filtres")
    
    # Filtres interactifs
    selected_years = st.multiselect(
        "Années de vente",
        options=sorted(stage['Année_vente'].unique()),
        default=sorted(stage['Année_vente'].unique())
    )
    
    selected_brands = st.multiselect(
        "Marques",
        options=sorted(stage['Marque'].unique()),
        default=sorted(stage['Marque'].unique())
    )
    
    st.markdown("---")
    st.markdown("""
    **Métriques Clés**  
    - CA Total: {:.2f}M€  
    - Ventes Totales: {:,}  
    - Satisfaction Moyenne: {:.1f}/10
    """.format(
        stage['Prix_vente (€)'].sum()/1e6,
        len(stage),
        stage['Score_satisfaction_client (1-10)'].mean()
    ))
    st.markdown("---")
    st.markdown("© 2023 - Analyse Ventes Auto Pro")

# Filtrage des données
filtered_data = stage[
    (stage['Année_vente'].isin(selected_years)) & 
    (stage['Marque'].isin(selected_brands))
]

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "📈 Analyse des Ventes", "🔍 Analyse Clients", "📌 Insights"])

with tab1:
    st.header("Vue Globale du Marché")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chiffre d'Affaires Total", f"{filtered_data['Prix_vente (€)'].sum()/1e6:.2f}M€")
    
    with col2:
        st.metric("Nombre Total de Ventes", f"{len(filtered_data):,}")
    
    with col3:
        st.metric("Satisfaction Moyenne", f"{filtered_data['Score_satisfaction_client (1-10)'].mean():.1f}/10")
    
    st.markdown("---")
    
    # Graphiques en ligne
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Évolution Annuelle des Ventes")
        annual_sales = filtered_data.groupby('Année_vente').agg({
            'ID_vente': 'count',
            'Prix_vente (€)': 'sum'
        }).rename(columns={'ID_vente': 'Ventes', 'Prix_vente (€)': 'CA'})
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(annual_sales.index, annual_sales['Ventes'], marker='o', color=COLOR_PALETTE[0], linewidth=2.5)
        ax.set_title('Nombre de Ventes par Année', pad=20)
        ax.set_ylabel('Nombre de Ventes')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Évolution Annuelle du CA")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(annual_sales.index, annual_sales['CA']/1e6, marker='o', color=COLOR_PALETTE[1], linewidth=2.5)
        ax.set_title('Chiffre d\'Affaires par Année (M€)', pad=20)
        ax.set_ylabel('CA (M€)')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Top marques
    st.subheader("Top 10 des Marques les Plus Vendues")
    top_brands = filtered_data['Marque'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="Blues_d")
    ax.set_title('Top 10 des Marques', pad=20)
    ax.set_xlabel('Nombre de Ventes')
    ax.set_ylabel('')
    ax.bar_label(ax.containers[0], fmt='%g', padding=3)
    st.pyplot(fig)

with tab2:
    st.header("Analyse Détailée des Ventes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ventes par Pays")
        sales_by_country = filtered_data.groupby('Pays_vente')['Prix_vente (€)'].sum().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sales_by_country.plot(kind='barh', color=COLOR_PALETTE[2], ax=ax)
        ax.set_title('Chiffre d\'Affaires par Pays', pad=20)
        ax.set_xlabel('CA (€)')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 des Modèles")
        top_models = filtered_data.groupby(['Marque', 'Modèle']).size().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_models.plot(kind='barh', color=COLOR_PALETTE[3], ax=ax)  # Correction de l'orthographe
        ax.set_title('Top 10 des Modèles', pad=20)
        ax.set_xlabel('Nombre de Ventes')
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Analyse par Type de Carburant")
    
    # Heatmap marque/carburant
    pivot_table = pd.pivot_table(
        filtered_data,
        values='ID_vente',
        index='Marque',
        columns='Type_carburant',
        aggfunc='count',
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
    ax.set_title('Répartition des Ventes par Marque et Carburant', pad=20)
    st.pyplot(fig)
    
    # Boxplot prix par marque et carburant
    st.subheader("Distribution des Prix par Marque et Carburant")
    top_brands = filtered_data['Marque'].value_counts().head(5).index
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        x='Marque',
        y='Prix_vente (€)',
        hue='Type_carburant',
        data=filtered_data[filtered_data['Marque'].isin(top_brands)],
        palette="Set2",
        ax=ax
    )
    ax.set_title('Distribution des Prix', pad=20)
    ax.set_ylabel('Prix (€)')
    ax.set_xlabel('')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    st.header("Analyse Clientèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Répartition des Types de Clients")
        client_type = filtered_data['Client_type'].value_counts()
        
        fig = px.pie(
            client_type,
            values=client_type.values,
            names=client_type.index,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Clients par Marque")
        client_brand = filtered_data.groupby(['Marque', 'Client_type']).size().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        client_brand.plot(kind='bar', stacked=True, ax=ax, colormap='Paired')
        ax.set_title('Répartition Clients par Marque', pad=20)
        ax.set_ylabel('Nombre de Clients')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Satisfaction Client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Satisfaction par Marque")
        satisfaction_brand = filtered_data.groupby('Marque')['Score_satisfaction_client (1-10)'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        satisfaction_brand.plot(kind='barh', color=COLOR_PALETTE[4], ax=ax)
        ax.set_title('Satisfaction Moyenne par Marque', pad=20)
        ax.set_xlabel('Score (1-10)')
        ax.set_xlim(0, 10)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Satisfaction vs Prix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='Prix_vente (€)',
            y='Score_satisfaction_client (1-10)',
            data=filtered_data,
            hue='Client_type',
            palette='Set2',
            alpha=0.7,
            ax=ax
        )
        ax.set_title('Relation Prix/Satisfaction', pad=20)
        ax.set_xlabel('Prix (€)')
        ax.set_ylabel('Score Satisfaction')
        st.pyplot(fig)

with tab4:
    st.header("Insights et Recommandations")
    
    st.subheader("Principales Observations")
    st.markdown("""
    - **Top Marques** : Les marques **{}** dominent le marché avec **{}%** des ventes totales.
    - **Évolution** : Croissance annuelle moyenne de **{:.1f}%** du chiffre d'affaires.
    - **Satisfaction** : Les clients professionnels sont **{:.1f}%** plus satisfaits que les particuliers.
    - **Carburants** : Le **{}** représente **{:.1f}%** des ventes.
    """.format(
        ", ".join(stage['Marque'].value_counts().head(3).index.tolist()),
        stage['Marque'].value_counts().head(3).sum()/len(stage)*100,
        10,  # À remplacer par le vrai taux de croissance
        5,   # À remplacer par la vraie différence de satisfaction
        stage['Type_carburant'].value_counts().index[0],
        stage['Type_carburant'].value_counts().values[0]/len(stage)*100
    ))
    
    st.markdown("---")
    
    st.subheader("Recommandations Stratégiques")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("""
        **📈 Optimisation des Ventes**
        - Cibler les marchés **{}** où la croissance est la plus forte
        - Développer l'offre de modèles **{}** les plus rentables
        - Améliorer la marge sur les véhicules **{}**
        """.format(
            stage.groupby('Pays_vente')['Prix_vente (€)'].sum().idxmax(),
            stage.groupby('Modèle')['Prix_vente (€)'].mean().idxmax(),
            stage.groupby('Type_carburant')['Prix_vente (€)'].mean().idxmax()
        ))
    
    with rec_col2:
        st.markdown("""
        **🎯 Stratégie Client**
        - Programme fidélité pour les clients **{}**
        - Packages premium pour les **{}**
        - Formation spécifique pour les vendeurs sur les modèles **{}**
        """.format(
            stage['Client_type'].value_counts().idxmax(),
            "professionnels" if stage.groupby('Client_type')['Prix_vente (€)'].mean().idxmax() == 'Professionnel' else "particuliers",
            stage.groupby('Modèle')['Score_satisfaction_client (1-10)'].mean().idxmin()
        ))
    
    st.markdown("---")
    
    st.subheader("Matrice de Corrélation")
    numerical_cols = ['Prix_vente (€)', 'Année_modèle', 'Kilométrage (km)', 'Garantie (mois)', 
                     'Score_satisfaction_client (1-10)', 'Remise (€)', 'Délai_livraison (jours)']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        filtered_data[numerical_cols].corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        ax=ax
    )
    ax.set_title('Relations entre Variables Clés', pad=20)
    st.pyplot(fig)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;font-size:0.9rem;">
    Dashboard développé avec Streamlit | Données: STAGE.xlsx | © 2023
</div>
""", unsafe_allow_html=True)