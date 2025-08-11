from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fusion
df = (
    sales.groupby("Customer_ID")
    .agg(
        Total_Spent=("Sale_Price", "sum"),
        Num_Purchases=("Sale_ID", "count"),
        Avg_Order_Value=("Sale_Price", "mean"),
    )
    .reset_index()
)

df = df.merge(cust[["Customer_ID", "Age", "Gender"]], on="Customer_ID")

# Clustering
X = df[["Age", "Total_Spent", "Num_Purchases", "Avg_Order_Value"]]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
df["Cluster"] = kmeans.labels_
