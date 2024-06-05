import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset manually
data = {
    'total_bill': [16.99, 10.34, 21.01, 23.68, 24.59],
    'tip': [1.01, 1.66, 3.50, 3.31, 3.61],
    'sex': ['Female', 'Male', 'Male', 'Male', 'Female'],
    'smoker': ['No', 'No', 'No', 'No', 'No'],
    'day': ['Sun', 'Sun', 'Sun', 'Sun', 'Sun'],
    'time': ['Dinner', 'Dinner', 'Dinner', 'Dinner', 'Dinner'],
    'size': [2, 3, 3, 2, 4]
}

# Convert to DataFrame
tips = pd.DataFrame(data)

# Display the first few rows of the dataset
print(tips.head())

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot of total_bill vs tip
sns.scatterplot(ax=axs[0, 0], data=tips, x='total_bill', y='tip', hue='sex', style='smoker')
axs[0, 0].set_title('Total Bill vs Tip by Sex and Smoker')

# Bar plot of total bill by day
sns.barplot(ax=axs[0, 1], data=tips, x='day', y='total_bill', ci=None)
axs[0, 1].set_title('Total Bill by Day')

# Box plot of total bill by time of day
sns.boxplot(ax=axs[1, 0], data=tips, x='time', y='total_bill')
axs[1, 0].set_title('Total Bill by Time of Day')

# Count plot of party size
sns.countplot(ax=axs[1, 1], data=tips, x='size')
axs[1, 1].set_title('Count of Party Size')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
