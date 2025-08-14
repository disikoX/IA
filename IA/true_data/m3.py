# m3_segmentation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

INCIDENTS_CSV = "incidents.csv"
LOGINS_CSV = "logins.csv"

inc = pd.read_csv(INCIDENTS_CSV, parse_dates=["Date"])
log = pd.read_csv(LOGINS_CSV, parse_dates=["DateHeure"])

# ---------- Segmentation des entreprises ----------
# Features: fréquence d’incidents, impact moyen, indispo moyenne, diversité des attaques
agg_ent = (
    inc.groupby("Entreprise")
      .agg(
          freq_incidents=("Entreprise","size"),
          impact_moy=("ImpactAriary","mean"),
          indispo_moy=("IndispoHeures","mean"),
          nb_types=("TypeAttaque","nunique"),
          secteur=("Secteur","first"),
          taille=("Taille","first")
      )
      .reset_index()
)

X_ent = agg_ent[["freq_incidents","impact_moy","indispo_moy","nb_types","taille"]].fillna(0)
scaler_ent = StandardScaler()
X_ent_scaled = scaler_ent.fit_transform(X_ent)

k_ent = KMeans(n_clusters=3, n_init=10, random_state=42)
agg_ent["cluster_esn"] = k_ent.fit_predict(X_ent_scaled)

# PCA 2D pour visualiser (à tracer dans M2/M8 si besoin)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_ent_scaled)
agg_ent["pc1"] = coords[:,0]
agg_ent["pc2"] = coords[:,1]

print("=== Segments entreprises ===")
print(agg_ent[["Entreprise","secteur","taille","freq_incidents","impact_moy","indispo_moy","nb_types","cluster_esn"]].head())

agg_ent.to_csv("segments_entreprises.csv", index=False)

# ---------- Segmentation des utilisateurs ----------
# Construire des features par utilisateur : nb échecs, nb succès, ratio échecs, pays distincts, IP distinctes
log["is_fail"] = (log["Resultat"] == "failure").astype(int)
log["is_succ"] = (log["Resultat"] == "success").astype(int)

agg_user = (
    log.groupby(["Utilisateur","Role","Departement"])
       .agg(
           nb_echecs=("is_fail","sum"),
           nb_succes=("is_succ","sum"),
           nb_total=("Resultat","size"),
           nb_pays=("Localisation","nunique"),
           nb_ip=("IPSource","nunique")
       ).reset_index()
)

agg_user["ratio_echec"] = np.where(agg_user["nb_total"]>0, agg_user["nb_echecs"]/agg_user["nb_total"], 0)

X_user = agg_user[["nb_echecs","nb_succes","nb_total","nb_pays","nb_ip","ratio_echec"]]
scaler_usr = StandardScaler()
X_user_scaled = scaler_usr.fit_transform(X_user)

k_usr = KMeans(n_clusters=4, n_init=10, random_state=42)
agg_user["cluster_risque"] = k_usr.fit_predict(X_user_scaled)

print("\n=== Segments utilisateurs ===")
print(agg_user.head())

agg_user.to_csv("segments_utilisateurs.csv", index=False)
