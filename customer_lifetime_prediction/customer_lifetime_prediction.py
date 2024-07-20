import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

# 1. Loading the Dataset
retail_df = pd.read_excel('/Users/rishabhbassi/Downloads/customer_lifetime_pred/online_retail_II.xlsx')

# 2. Data Preprocessing
retail_df.drop_duplicates(inplace=True)
retail_df.dropna(subset=['CustomerID'], inplace=True)
retail_df['Total'] = retail_df['Quantity'] * retail_df['UnitPrice']

# 3. RFM Table
latest_date = retail_df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = retail_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'count',
    'Total': 'sum'
}).reset_index()
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Total': 'MonetaryValue'
}, inplace=True)

# 4. Fitting the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(rfm['Frequency'], rfm['Recency'], rfm['MonetaryValue'])

# Predicting future transactions
rfm['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(30, rfm['Frequency'], rfm['Recency'], rfm['MonetaryValue'])

# 5. Fitting the Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(rfm['Frequency'], rfm['MonetaryValue'])

# Predicting monetary value
rfm['predicted_monetary_value'] = ggf.conditional_expected_average_profit(rfm['Frequency'], rfm['MonetaryValue'])

# Calculating CLTV
rfm['CLTV'] = ggf.customer_lifetime_value(
    bgf,
    rfm['Frequency'],
    rfm['Recency'],
    rfm['MonetaryValue'],
    time=12,  # months
    discount_rate=0.01  # monthly discount rate
)

# Visualizing CLTV
sns.histplot(rfm['CLTV'], bins=30)
plt.title('CLTV Distribution')
plt.show()

# Summary of results
print(rfm[['CustomerID', 'predicted_purchases', 'predicted_monetary_value', 'CLTV']].head())

# Plotting the expected number of transactions
plot_period_transactions(bgf)
plt.show()
