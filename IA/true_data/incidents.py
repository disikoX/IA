# generate_incidents.py
import pandas as pd
import numpy as np
import random
from datetime import datetime
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# -------------------------
# CONFIG
# -------------------------
NB_ENTREPRISES = 30        # plus d'entreprises
NB_INCIDENTS = 1000        # plus d'incidents

secteurs = ["Finance", "Santé", "Industrie", "Technologie", "Commerce", "Éducation", "Transport", "Énergie"]
types_attaque = ["phishing", "ransomware", "malware", "ddos", "intrusion", "fuite de données", "exploitation de vulnérabilité", "account takeover"]
vecteurs = ["email", "téléchargement", "port ouvert", "RDP exposé", "clé USB", "API vulnérable", "site compromis"]
campagnes = ["FormationPhishing", "MFA", "PatchUrgent", "AuditSécurité", None]

entreprises = [fake.company() for _ in range(NB_ENTREPRISES)]
taille_entreprise = {e: random.randint(50, 2000) for e in entreprises}

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 8, 1)

incidents_data = []
for _ in range(NB_INCIDENTS):
    e = random.choice(entreprises)
    secteur = random.choice(secteurs)
    taille = taille_entreprise[e]
    type_a = random.choice(types_attaque)
    date_incident = fake.date_time_between(start_date=start_date, end_date=end_date)
    vecteur = random.choice(vecteurs)
    impact = round(abs(np.random.normal(200000, 500000)), 2)  # €
    indispo = max(0, int(np.random.normal(24, 12)))  # heures
    donnees_comp = random.choice(["Oui", "Non"])
    campagne = random.choice(campagnes)
    incidents_data.append([e, secteur, taille, type_a, date_incident, vecteur, impact, indispo, donnees_comp, campagne])

df_incidents = pd.DataFrame(incidents_data, columns=[
    "Entreprise", "Secteur", "Taille", "TypeAttaque", "Date", "Vecteur", 
    "ImpactAriary", "IndispoHeures", "DonneesCompromises", "CampagneSécurité"
])
df_incidents.sort_values("Date", inplace=True)
df_incidents.to_csv("incidents.csv", index=False)
print(f"[OK] incidents.csv généré avec {len(df_incidents)} lignes pour {NB_ENTREPRISES} entreprises")
