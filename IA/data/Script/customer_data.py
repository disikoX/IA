import numpy as np
import pandas as pd


customers = pd.read_excel("customers_data.xlsx")

feedback = []
date_range = pd.date_range("2023-01-01", "2023-06-30")
for i in range(50):
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.3, 0.35])
    comment = ""
    if rating == 5:
        comment = np.random.choice(
            ["Great quality!", "Love the style!", "Fast delivery"]
        )
    elif rating >= 3:
        comment = np.random.choice(["Good value", "Nice design", "Could be better"])
    else:
        comment = np.random.choice(["Too expensive", "Poor fit", "Not as expected"])

    feedback.append(
        {
            "Feedback_ID": 5000 + i,
            "Customer_ID": np.random.choice(customers["Customer_ID"]),
            "Rating": rating,
            "Comment": comment,
            "Date": np.random.choice(date_range),
        }
    )

pd.DataFrame(feedback).to_excel("customer_feedback.xlsx", index=False)
