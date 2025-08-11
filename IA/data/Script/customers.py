import pandas as pd
import numpy as np

np.random.seed(42)
names = [f"Client_{i}" for i in range(5, 50)]
ages = np.random.randint(18, 65, 45)
genders = np.random.choice(["Male", "Female"], 45, p=[0.48, 0.52])
locations = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Miami",
    "Seattle",
    "Denver",
]
incomes = np.random.randint(30000, 120000, 45)

new_customers = pd.DataFrame(
    {
        "Customer_ID": range(2005, 2050),
        "Name": names,
        "Age": ages,
        "Gender": genders,
        "Location": np.random.choice(locations, 45),
        "Join_Date": pd.date_range("2020-01-01", "2023-06-30", periods=45),
        "Total_Spent": np.random.lognormal(5.8, 0.8, 45).astype(int),
        "Income": incomes,
        "Preferred_Channel": np.random.choice(["Online", "In-Store"], 45, p=[0.6, 0.4]),
        "Email_Open_Rate": np.round(np.random.beta(4, 2, 45), 2),
    }
)

# Fusionner avec les anciens
original = pd.read_excel("customers_data.xlsx")
customers_extended = pd.concat([original, new_customers], ignore_index=True)
customers_extended.to_excel("customers_data_extended.xlsx", index=False)
