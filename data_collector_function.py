import pandas as pd

churn_data = pd.read_csv(
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/refs/heads/master/data/Telco-Customer-Churn.csv"
)

churn_data_csv = churn_data.to_csv("./data/customer_churn.csv")
