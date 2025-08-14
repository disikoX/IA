# m6_prediction_corrige.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------- PARTIE 1 : Entreprises ----------
inc = pd.read_csv("incidents.csv", parse_dates=["Date"])

# Construire une série mensuelle par entreprise
g = inc.groupby("Entreprise")
frames = []

for ent, df in g:
    s = df.set_index("Date").resample("ME").agg({
        "Entreprise": "size",
        "ImpactAriary": "sum",
        "IndispoHeures": "sum"
    }).rename(columns={
        "Entreprise": "nb_incidents",
        "ImpactAriary": "impact_total",
        "IndispoHeures": "indispo_total"
    }).fillna(0)
    s["Entreprise"] = ent
    frames.append(s.reset_index())

panel = pd.concat(frames, ignore_index=True)

# Lag features + cible
panel = panel.sort_values(["Entreprise","Date"])
panel["nb_incidents_prev1"] = panel.groupby("Entreprise")["nb_incidents"].shift(1).fillna(0)
panel["impact_prev1"] = panel.groupby("Entreprise")["impact_total"].shift(1).fillna(0)
panel["indispo_prev1"] = panel.groupby("Entreprise")["indispo_total"].shift(1).fillna(0)

# Cible: y=1 si nb_incidents du mois courant >0
panel["y_current"] = (panel["nb_incidents"] > 0).astype(int)
panel["y_next"] = panel.groupby("Entreprise")["y_current"].shift(-1)
panel = panel.dropna(subset=["y_next"])

X_ent = panel[["nb_incidents_prev1","impact_prev1","indispo_prev1"]]
y_ent = panel["y_next"].astype(int)

# Split train/test
Xtr, Xte, ytr, yte = train_test_split(X_ent, y_ent, test_size=0.25, random_state=42, stratify=y_ent)

# RandomForest pour prédire incidents prochain mois
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(Xtr, ytr)
pred_proba = rf.predict_proba(Xte)[:,1]
pred = (pred_proba >= 0.5).astype(int)
print("=== Entreprises: RandomForest ===")
print(classification_report(yte, pred, digits=3))
print("AUC:", roc_auc_score(yte, pred_proba))

# ---------- PARTIE 2 : Utilisateurs ----------
log = pd.read_csv("logins.csv", parse_dates=["DateHeure"])
log["Resultat"] = log["Resultat"].str.lower().str.strip()
log["is_fail"] = (log["Resultat"]=="échec").astype(int)

# Label proxy “compromis”: échec suivi d’un succès depuis IP différente <1h
log = log.sort_values(["Utilisateur","DateHeure"])
log["IP_prev"] = log.groupby("Utilisateur")["IPSource"].shift(1)
log["Date_prev"] = log.groupby("Utilisateur")["DateHeure"].shift(1)
log["Result_prev"] = log.groupby("Utilisateur")["Resultat"].shift(1)

def compromised_row(row):
    if pd.isna(row["Date_prev"]): 
        return 0
    dt = (row["DateHeure"] - row["Date_prev"]).total_seconds() / 3600.0
    return int((row["Result_prev"]=="échec") and (row["Resultat"]=="succès") 
               and (row["IP_prev"]!=row["IPSource"]) and (dt<=1.0))

log["compromis_signal"] = log.apply(compromised_row, axis=1)

# Agrégation par utilisateur
agg_user = (
    log.groupby(["Utilisateur","Role","Departement"])
       .agg(
           nb_echecs=("is_fail","sum"),
           nb_total=("Resultat","size"),
           nb_ip=("IPSource","nunique"),
           nb_pays=("Localisation","nunique"),
           nb_compromis=("compromis_signal","sum")
       ).reset_index()
)
agg_user["ratio_echec"] = np.where(agg_user["nb_total"]>0, agg_user["nb_echecs"]/agg_user["nb_total"], 0)
agg_user["y_compromis"] = (agg_user["nb_compromis"]>0).astype(int)

Xu = agg_user[["nb_echecs","nb_total","nb_ip","nb_pays","ratio_echec"]]
yu = agg_user["y_compromis"]

Xtr, Xte, ytr, yte = train_test_split(Xu, yu, test_size=0.25, random_state=42, stratify=yu)

lr = LogisticRegression(max_iter=200, class_weight="balanced")
lr.fit(Xtr, ytr)
proba = lr.predict_proba(Xte)[:,1]
pred = (proba>=0.5).astype(int)

print("\n=== Utilisateurs: LogisticRegression ===")
print(classification_report(yte, pred, digits=3))
print("AUC:", roc_auc_score(yte, proba))

# Sauvegarde scores de risque
agg_user["risk_score"] = lr.predict_proba(Xu)[:,1]
agg_user[["Utilisateur","Role","Departement","risk_score"]].to_csv("risque_utilisateur.csv", index=False)
print("\nScores de risque écrits dans risque_utilisateur.csv")
