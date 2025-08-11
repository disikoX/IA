import pandas as pd
import numpy as np

channels = ["Online", "Social", "Email", "In-Store", "TV"]
data = []

for month in range(1, 7):
    for i, channel in enumerate(channels, 1):
        start = f"2023-{month:02d}-01"
        end = f"2023-{month:02d}-28"
        budget = np.random.choice([500, 1000, 1500, 2000, 3000])
        impressions = budget * np.random.randint(15, 30)
        clicks = int(impressions * np.random.uniform(0.02, 0.1))
        conversions = int(clicks * np.random.uniform(0.05, 0.25))

        data.append(
            {
                "Campaign_ID": f"{month*10 + i}",
                "Channel": channel,
                "Start_Date": start,
                "End_Date": end,
                "Budget": budget,
                "Impressions": impressions,
                "Clicks": clicks,
                "Conversions": conversions,
            }
        )

marketing_extended = pd.DataFrame(data)
marketing_extended.to_excel("marketing_data_extended.xlsx", index=False)
