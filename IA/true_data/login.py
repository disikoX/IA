# generate_logins.py
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
NB_UTILISATEURS = 150      # plus d'utilisateurs
NB_LOGINS = 20000          # plus de tentatives

roles = ["Employé", "Manager", "Admin système", "Développeur", "Support IT", "Stagiaire"]
departements = ["RH", "IT", "Ventes", "Marketing", "Production", "Comptabilité", "Logistique"]
countries = ["France", "USA", "Canada", "Allemagne", "Japon", "Brésil", "Espagne", "Italie", "Unknown"]

utilisateurs = [fake.user_name() for _ in range(NB_UTILISATEURS)]
role_user = {u: random.choice(roles) for u in utilisateurs}
dept_user = {u: random.choice(departements) for u in utilisateurs}

start_login_date = datetime(2024, 1, 1)
end_login_date = datetime(2025, 8, 1)

logins_data = []
for _ in range(NB_LOGINS):
    user = random.choice(utilisateurs)
    role = role_user[user]
    dept = dept_user[user]
    date_login = fake.date_time_between(start_date=start_login_date, end_date=end_login_date)
    ip = fake.ipv4_public()
    country = random.choice(countries)
    # Probabilité d'échec plus forte pour certains rôles
    p_fail = 0.2 if role != "Admin système" else 0.1
    result = np.random.choice(["succès", "échec"], p=[1-p_fail, p_fail])
    logins_data.append([user, role, dept, date_login, ip, country, result])

df_logins = pd.DataFrame(logins_data, columns=[
    "Utilisateur", "Role", "Departement", "DateHeure", "IPSource", "Localisation", "Resultat"
])
df_logins.sort_values("DateHeure", inplace=True)
df_logins.to_csv("logins.csv", index=False)
print(f"[OK] logins.csv généré avec {len(df_logins)} lignes pour {NB_UTILISATEURS} utilisateurs")
