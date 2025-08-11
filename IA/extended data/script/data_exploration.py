import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sales = pd.read_excel("sales_data_extended.xlsx")
cust = pd.read_excel("customers_data_extended.xlsx")

print("Statistiques des ventes :")
print(sales.describe())

plt.figure(figsize=(10, 6))
sns.histplot(cust["Age"], bins=20, kde=True)
plt.title("Répartition par âge")
plt.savefig("age_distribution.png")
