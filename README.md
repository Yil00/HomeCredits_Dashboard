# Projet 7_dashboard streamlit
## Implémentation d'un modèle de scoring :
____________________________________________
La société financière Prêt à dépenser propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

**Context :** 
___________________
L’entreprise souhaite mettre en œuvre un outil de scoring crédit qui calcule la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
_______________________________________________
**Dasboard - "HOME CREDID"**(by Streamlit) :

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. 

Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

Spécifications du dashboard : Il contient les fonctionnalités suivantes :

    - Il permet de visualiser le score et l’interprétation de ce score pour chaque client.
    - Il permet de visualiser des informations descriptives relatives à un client.
    - Obtenir des détails sur les critères d'obtiens du futur crédit.

Le dashboard réalisé avec Streamlit.
________________________________________________
*Détails :*
- Les données originales sont téléchargeables sur [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

- L'API de Prédiction a été réalisée avec Flask et déployée sur Heroku.

- Le dashboard a destination des chargés de clientèles a été réalisé avec Streamlit. 