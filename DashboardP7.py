# ====================================================================
# Chargement des librairies
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
import requests
import json 
import os
from sklearn.cluster import KMeans
from PIL import Image
plt.style.use('fivethirtyeight')

import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
from streamlit_shap import st_shap
from xplotter.insights import *

# ====================================================================
# Version : 0.0.1 - 01/02/2023
# ====================================================================

__version__ = '0.0.0'

# ====================================================================

# ====================================================================
# CHARGEMENT DES DONNEES
# ====================================================================

# ====================================================================
# IMAGES
# ====================================================================
# Logo de l'entreprise
# logo =  Image.open("imageslogo.png")
path = "images" 
logo = (os.path.join(path,"logo.png"))
# Légende des courbes
st.title("Crédit Banks - Home")

########################
# Lecture des fichiers #
########################
def main():

    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]
        targets = data.TARGET.value_counts()
        return nb_credits, rev_moy, credits_moy, targets

URL_API= "http://127.0.0.1:5003"
@st.cache #mise en cache de la fonction pour exécution unique
def lecture_X_test_original():
    X_test_original = pd.read_csv("Data/X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

@st.cache 
def lecture_X_test_clean():
    X_test_clean = pd.read_csv("Data/X_test_clean.csv")
    #st.dataframe(X_test_clean)
    return X_test_clean

@st.cache 
def lecture_description_variables():
    description_variables = pd.read_csv("Data/description_variable.csv", sep=";")
    return description_variables

@st.cache 
def calcul_valeurs_shap():
        model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
        explainer = shap.TreeExplainer(model_LGBM)
        shap_values = explainer.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))
        return shap_values
@st.cache
# Utiliser une fonction pour créer un identifiant unique à chaque appel
def get_widget_id():
    global widget_id
    widget_id += 1
    return str(widget_id)




if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Home Crédit - Prêt à dépenser :</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")



        #Title display
    html_temp = """
    <div style="background-color: blue; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision dashboard…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    # Chargement du logo de l'entreprise
    st.sidebar.image(logo, width=240, caption=" Dashboard - Aide à la décision",
                 use_column_width='always')
    #Customer ID selection
    st.sidebar.header("**Information général**")
    
    ##########################################################
    # Création et affichage du sélecteur du numéro de client #
    ##########################################################
    liste_clients = list(lecture_X_test_original()['sk_id_curr'])
    col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")
    ################################################################
    #  URL_API - Score_value
    ################################################################
    idApi = URL_API +"/predictScore"+"?id_client="+ str(np.array(ID_client))
    #
    response = requests.get(idApi)
    #
    content = json.loads(response.content.decode('utf-8'))
    #
    score_value1 = pd.Series(content).values
    score_value = score_value1[0]


    #  URL_API - Solvabilité 
    idApi = URL_API +"/Solvabilite"+"?id_client="+ str(ID_client)
    #
    response = requests.get(idApi)
    #
    content = json.loads(response.content.decode("utf-8"))
    #
    solvabilité1 = pd.Series(content).values
    solvabilite=solvabilité1[0]

    #  URL_API - Decision
    idApi = URL_API +"/decision"+"?id_client="+ str(ID_client)
    #
    response = requests.get(idApi)
    #
    content = json.loads(response.content.decode("utf-8"))
    #
    dec1 = pd.Series(content).values
    decision=dec1[0]

    #################################################################

    #Loading selectbox
    #chk_id = st.sidebar.selectbox("Client ID", ID_client)
    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    #################################################
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
        st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
        st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{solvabilite}** \
                et décide de lui **{decision}** le crédit. ")
    # Impression du graphique jauge
    with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 30, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'blue', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#008000'},
                                        {'range': [20, 40], 'color': 'azure'},
                                        {'range': [40, 60], 'color': 'yellow'},
                                        {'range': [60, 80], 'color': 'orange'},
                                        {'range': [80, 100], 'color': 'red'}],
                                'threshold': {'line': {'color': 'blue', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 30 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': '#772b58', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

    ################################
    # Explication de la prédiction #
    ################################
    # Titre 2
    
    if  st.sidebar.checkbox("Détails: Score-Client (+ information ...) "):

        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    2. Comment <strong>le score client</strong> est-il calculé ?</h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("")

        # Calcul des valeurs Shap
        explainer_shap = shap.TreeExplainer(model_LGBM)
        shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))

        # récupération de l'index correspondant à l'identifiant du client
        idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

        # Graphique force_plot
        st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.") 
        st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être \
                en défaut de paiement et celles qui la diminuent, ainsi que l’amplitude de cet impact.")
        st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                                shap_values[1][idx,:], 
                                lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                                link='logit',
                                figsize=(20, 8),
                                ordering_keys=True,
                                text_rotation=0,
                                contribution_threshold=0.05))
        # Graphique decision_plot
        st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
                Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
                avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
                “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées. ")
        st.write("Seules les **15 variables explicatives** les plus importantes sont affichées par ordre décroissant.")
        st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                                shap_values[1][idx,:], 
                                lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                                feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                                feature_order='importance',
                                feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                                link='logit'))
    else:
        st.markdown("", unsafe_allow_html=False)

    # Titre 3
    if  st.sidebar.checkbox("Détails des variables:"):

        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    3. Lexique des variables </h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("")

        st.write("La base de données globale contient plusieur centaines de variables explicatives. Certaines d'entre elles étaient peu \
                renseignées ou peu voir non disciminantes et d'autres très corrélées (2 variables corrélées entre elles \
                apportent la même information : l'une d'elles est donc redondante).")
        st.write("Après leur analyse, env **+50 variables se sont avérées pertinentes** pour prédire si le client aura ou non des difficultés de paiement.")

        pd.set_option('display.max_colwidth', None)
        st.dataframe(lecture_description_variables())

    else:
        st.markdown("", unsafe_allow_html=False)

    if  st.checkbox("Afficher les informations client | Profil Clients:"):
        ##########################################################
        # Création et affichage du sélecteur du numéro de client #
        ##########################################################
        # Titre 1
        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    Info. Le profil client : </h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("")


        widget_id = (id for id in range(1, 100_00))
        liste_clients = list(lecture_X_test_original()['sk_id_curr'])
        col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
        with col1:
            ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                    (liste_clients),key=next(widget_id))
            st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
        with col2:
            st.write("")

        #################################################
        # Lecture du modèle de prédiction et des scores #
        #################################################
        model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
        y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
        y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

        # Récupération du score du client
        y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                        lecture_X_test_clean()['sk_id_curr']], axis=1)
        #st.dataframe(y_pred_lgbm_proba_df)
        score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID_client]
        score_value = score.proba_classe_1.iloc[0]

        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1%}**.")
        st.write(f"**Il y a donc un risque de {score_value:.1%} que le client ait des difficultés de paiement.**")

        #st.dataframe(lecture_X_test_original())

        ########################################################
        # Récupération et affichage des informations du client #
        ########################################################

        data_client=lecture_X_test_original()[lecture_X_test_original().sk_id_curr == ID_client]

        col1, col2 = st.columns(2)
        with col1:
            # Titre H2
            st.markdown("""
                        <h2 style="color:#fff8dc;text-align:center;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                        Situation</h2>
                        """, 
                        unsafe_allow_html=True)
            st.write("")
            st.write(f"Genre : **{data_client['code_gender'].values[0]}**")
            st.write(f"Tranche d'âge : **{data_client['age_client'].values[0]}**")
            st.write(f"Ancienneté de la piède d'identité : **{data_client['anciennete_cni'].values[0]}**")
            st.write(f"Situation familiale : **{data_client['name_family_status'].values[0]}**")
            st.write(f"Taille de la famille : **{data_client['taille_famille'].values[0]}**")
            st.write(f"Nombre d'enfants : **{data_client['nbr_enfants'].values[0]}**")
            st.write(f"Niveau d'éducation : **{data_client['name_education_type'].values[0]}**")
            st.write(f"Revenu Total Annuel : **{data_client['total_revenus'].values[0]} $**")
            st.write(f"Type d'emploi : **{data_client['name_income_type'].values[0]}**")
            st.write(f"Ancienneté dans son entreprise actuelle : **{data_client['anciennete_entreprise'].values[0]}**")
            st.write(f"Type d'habitation : **{data_client['name_housing_type'].values[0]}**")
            st.write(f"Densité de la Population de la région où vit le client : **{data_client['pop_region'].values[0]}**")
            st.write(f"Evaluations de *'Prêt à dépenser'* de la région où vit le client : \
                       **{data_client['region_rating_client'].values[0]}**")

        with col2:
            # Titre H2
            st.markdown("""
                        <h2 style="color:red;text-align:center;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                        Profil emprunteur</h2>
                        """, 
                        unsafe_allow_html=True)
            st.write("")
            st.write(f"Type de Crédit demandé par le client : **{data_client['name_contract_type'].values[0]}**")
            st.write(f"Montant du Crédit demandé par le client : **{data_client['montant_credit'].values[0]} $**")
            st.write(f"Durée de remboursement du crédit : **{data_client['duree_remboursement'].values[0]}**")
            st.write(f"Taux d'endettement : **{data_client['taux_endettement'].values[0]}**")
            st.write(f"Score normalisé du client à partir d'une source de données externe : \
                      **{data_client['ext_source_2'].values[0]:.1%}**")
            st.write(f"Nombre de demande de prêt réalisée par le client : \
                       **{data_client['nb_demande_pret_precedente'].values[0]:.0f}**")
            st.write(f"Montant des demandes de prêt précédentes du client : \
                      **{data_client['montant_demande_pret_precedente'].values[0]} $**")
            st.write(f"Montant payé vs Montant attendu en % : **{data_client['montant_paye_vs_du'].values[0]:.1f}%**")
            st.write(f"Durée mensuelle moyenne des crédits précédents : **{data_client['cnt_instalment'].values[0]:.1f} mois**")
            st.write(f"Nombre de Crédit à la Consommation précédent du client : \
                      **{data_client['prev_contrat_type_consumer_loans'].values[0]:.0f}**")
            st.write(f"Nombre de Crédit Revolving précédent du client : \
                      **{data_client['prev_contrat_type_revolving_loans'].values[0]:.0f}**")
            st.write(f"Nombre de Crédit précédent refusé : \
                      **{data_client['prev_contrat_statut_refused'].values[0]:.0f}**")
            st.write(f"Nombre de crédits cloturés enregistrés au bureau du crédit : \
                      **{data_client['bureau_credit_actif_closed'].values[0]:.0f}**")
            st.write(f"Nombre de crédits de type *'carte de crédit'* enregistrés au bureau du crédit : \
                      **{data_client['bureau_credit_type_credit_card'].values[0]:.0f}**")
            st.write(f"Nombre d'années écoulées depuis la décision précédente : \
                      **{data_client['nb_year_depuis_decision_precedente'].values[0]:.0f} ans**")

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
    ###############################################################
    # Comparaison du profil du client à son groupe d'appartenance #
    ###############################################################    
    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()
    calcul_valeurs_shap()

    # Titre 1
    if  st.sidebar.checkbox("Contribution des variables :"):
        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    5. Les variables les plus importantes pour comprendre nos prédiction ?</h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("")

        st.write("L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. \
                Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. \
                Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte \
                la prédiction de manière positive ou négative.")

        st.write("Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit \
                avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter \
                ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. \
                Cette approche est inspirée de la théorie des jeux.")

        st.write("*__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.\
                Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.")

        fig = plt.figure()
        plt.title("Diagramme Feature importance", 
                fontname='Roboto Condensed',
                fontsize=20, 
                fontstyle='italic')
        if  st.checkbox('Feature importance : '):
            st_shap(shap.summary_plot(calcul_valeurs_shap()[1], 
                                feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns,
                                plot_size=(12, 6),
                                color='#daa520',
                                plot_type="bar",
                                max_display=10,
                                show = False))
            plt.show()
    else:
        st.markdown("", unsafe_allow_html=False)

    # Titre 2
    if  st.sidebar.checkbox("Niveau d'importance des critères:"):
        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    2. Quel est le niveau d'impact de chacune de nos variables sur nos prédiction ?</h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("")

        st.write("Le diagramme des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
                Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
                Chaque point représente une valeur de Shap (pour un client).")
        st.write("Les points jaune représentent des valeurs élevées de la variable et les points verts-blue des valeurs basses de la variable.")

        fig = plt.figure()
        plt.title("Interprétation Globale :\n Impact de chaque caractéristique sur la prédiction\n", 
                fontname='Roboto Condensed',
                fontsize=20, 
                fontstyle='italic')
        st_shap(shap.summary_plot(calcul_valeurs_shap()[1], 
                                features=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1),
                                feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns,
                                plot_size=(12, 16),
                                cmap='viridis',
                                plot_type="dot",
                                max_display=10,
                                show = False))
        plt.show()
    #else:
    #    st.markdown(" ", unsafe_allow_html=False)

        st.write("14 variables ont un impact significatif sur la prédiction.")
        
        if st.checkbox("légende des critères"):
           st.markdown("""
            1. Plus la valeur du 'Score normalisé à partir d'une source de données externe' est faible (points de couleur vert), 
               et plus la valeur Shap est élevée et donc plus le modèle prédit que le client aura des difficultés de paiement.<br>
            2. Plus la dernière demande de crédit du client, avant la demande actuelle, enregistrée au bureau des crédits, est récente 
               (points de couleur vert), plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
            3. Plus le montant payé par le client par rapport au montant attendu est faible (points de couleur vert), 
               plus la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
            4. Si le client est un homme, la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
            5. Plus la durée mensuelle du contrat pécédent du client est élevé (points de couleur fuchsia), 
               plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
            6. Plus le nombre de contrats pécédents refusés pour le client est élevé (points de couleur fuchsia), 
               plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
            7. Plus le client est jeune (points de couleur vert), plus la valeur Shap est élevée et
               donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
            8. Lorsque le client n'est pas allé dans l'enseignement supérieur (points vert), 
               la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
            9. Nombre de crédits soldés du client enregistrés au bureau du crédit : *impact indéfini* <br>
            10. Plus le nombre de versements réalisés par la client est faible (points de couleur vert), 
                plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
            11. Plus l'ancienneté du client dans son entreprise est faible (points de couleur vert), 
                plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
            12. Plus le nombre de Cartes de Crédit du client enregistrées au bureau du crédit est élevé (points de couleur fuchsia),
                plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
            13. Plus le montant de la demande de prêt actuelle du client est élevé (points de couleur fuchsia), 
                plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
            14. Plus le montant de la demande de prêt précédente du client est faible (points de couleur vert), 
                plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
                    """, 
                    unsafe_allow_html=True)

    else:
        st.markdown(" ", unsafe_allow_html=False)
    
    # Titre 2
    if  st.sidebar.checkbox("Plus de détails :"):
        st.markdown("""
                    <h1 style="color:#fff8dc;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                    Bonus. Graphique de dépendance</h1>
                    """, 
                    unsafe_allow_html=True)
        st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
                  sur l'ensemble de données avec un graphique de dépendance.")
        st.write("Le dependence plot permet d’analyser les variables deux par deux en suggérant une possiblité d’observation des interactions.\
                  Le scatter plot représente une dépendence entre une variable (en x) et les shapley values (en y) \
                  colorée par la variable la plus corrélées.")

        ################################################################################
        # Création et affichage du sélecteur des variables et des graphs de dépendance #
        ################################################################################
        liste_variables = lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list()

        col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
        with col1:
            ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                    (liste_variables))
            st.write("Vous avez sélectionné la variable :", ID_var)

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(121)
        shap.dependence_plot(ID_var, 
                            calcul_valeurs_shap()[1], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1), 
                            interaction_index=None,
                            alpha = 0.5,
                            x_jitter = 0.5,
                            title= "Graphique de Dépendance",
                            ax=ax1,
                            show = False)
        ax2 = fig.add_subplot(122)
        shap.dependence_plot(ID_var, 
                            calcul_valeurs_shap()[1], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1), 
                            interaction_index='auto',
                            alpha = 0.5,
                            x_jitter = 0.5,
                            title= "Graphique de Dépendance et Intéraction",
                            ax=ax2,
                            show = False)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.markdown(" ", unsafe_allow_html=False)
    ##########Graphique Unique varié 
    if  st.sidebar.checkbox("Feature Analyze :"):

        df = pd.read_csv("Data/X_test_original.csv")

        num_cols = df.select_dtypes(include=['number']).columns
        num_df = df[num_cols]

        # Sélection des colonnes non-numériques
        non_num_cols = df.select_dtypes(exclude=['number']).columns
        non_num_df = df[non_num_cols]
        # Liste des variables
        liste_variables2 = num_df.drop("SK_ID_CURR", axis=1).columns.tolist()

        model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
        y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
        y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

        # Récupération du score du client
        y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                lecture_X_test_clean()['sk_id_curr']], axis=1)
        
        #Prédit score dans le graphique(point d'affichage dans l'analyse bivariée)
        dfID = y_pred_lgbm_proba_df
        valueID =dfID["sk_id_curr"]

        # Merge tableau pour deuxième graphique
        df01 = y_pred_lgbm_proba_df
        df02 = num_df
        df01.columns = ['proba_classe_1','SK_ID_CURR']
        # Fusionner les dataframes sur la colonne "id"
        merged_df = pd.merge(df01, df02, on='SK_ID_CURR', how='left')
        # Score pred
        value0 = merged_df['proba_classe_1'] 
        value1ID = merged_df["SK_ID_CURR"]
        #%ise en avant de notre client coordonnée X et Y :
        #a = merged_df.loc[merged_df["SK_ID_CURR"] == 100001] 
        #a1 = a["SK_ID_CURR"]
        #a2 = a1[0]
        #
        #a = merged_df.loc[merged_df["SK_ID_CURR"] == 100001] 
        #b1 = a["proba_classe_1"]
        #b2 = b1[0]


        col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
        with col1:
            ID_var2 = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                    (liste_variables2))
            st.write("Vous avez sélectionné la variable :", ID_var2)

            # Création de l'histogramme interactif avec Plotly Express
            fig = px.histogram(num_df, x=ID_var2, nbins=20, text_auto=True,color_discrete_sequence = ['darkred'], barmode="overlay")

            # Modifier les couleurs des barres en bleu
            #fig.update_traces(marker=dict(color='red'), selector=dict(type='bar'))
            fig.update_layout(bargap=0.1, bargroupgap=0.1)

            # Affichage de l'histogramme interactif avec Plotly
            st.plotly_chart(fig)
        
        #Graphique multivariée 
        col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
        widget_id = (id for id in range(7, 100_00))
        widget_id2 = (id for id in range(4, 100_00))
        widget_id3 = (id for id in range(9, 100_00))
        with col1:
            ID_var3 = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                       (liste_variables2),key=next(widget_id))
            st.write("Vous avez sélectionné la variable :", ID_var3)
            
            ID_var4 = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                       (liste_variables2),key=next(widget_id2))
            st.write("Vous avez sélectionné la variable :", ID_var4)

            ID_var5 = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                       (value1ID),key=next(widget_id3))
            st.write("Vous avez sélectionné la variable :", ID_var5)

            # Choisir la colonne pour la légende
            legende_colonne = float(score_value)  # num_cols[0]    #"nom_colonne_legende"

            # Création du graphique à l'aide de la fonction scatter de Plotly avec la légende
            fig2 = px.scatter(merged_df, x=ID_var3, y=ID_var4, color=value0,
                             labels={ID_var3: ID_var3, ID_var4: ID_var4, legende_colonne: 'Score Prédiction'})
            

            st.plotly_chart(fig2)


if __name__ == '__main__':
    main()