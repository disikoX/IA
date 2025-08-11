import pandas as pd
import numpy as np

products = pd.read_excel("products_data.xlsx")
customers = pd.read_excel("customers_data_extended.xlsx")

products["Price"] = pd.to_numeric(products["Price"], errors="coerce")

dates = pd.date_range("2023-01-01", "2023-06-30")
data = []

for i in range(100):
    sale_id = 1000 + i
    product = products.sample(1).iloc[0]
    customer = customers.sample(1).iloc[0]
    date = np.random.choice(dates)
    quantity = np.random.randint(1, 4)
    channel = np.random.choice(["Online", "In-Store"], p=[0.7, 0.3])
    discount = np.random.choice([0, 0.1], p=[0.8, 0.2])  # 10% de réduction parfois
    sale_price = product["Price"] * quantity * (1 - discount)

    data.append(
        {
            "Sale_ID": sale_id,
            "Product_ID": product["Product_ID"],
            "Customer_ID": customer["Customer_ID"],
            "Date": date,
            "Quantity": quantity,
            "Sale_Price": round(sale_price, 2),
            "Channel": channel,
            "Discount_Applied": discount > 0,
        }
    )

sales_extended = pd.DataFrame(data)
sales_extended.to_excel("sales_data_extended.xlsx", index=False)
