import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Uncomment if necessary for data visualizations to work
%matplotlib inline

# Print version information
print("TensorFlow version: " + tf.__version__)
print("TensorFlow Decision Forests version: " + tfdf.__version__)

# Load the dataset
train_file_path = "../input/house-prices-advanced-regression-techniques/train.csv"
data = pd.read_csv(train_file_path)
print(f"Full training dataset shape: {data.shape}")

# Display the first few rows of the dataset
print(data.head(3))

# Remove the Id column as it's not needed for training
data.drop('Id', axis=1, inplace=True)
print(data.head(3))

# Inspect the dataset's column types
data.info()

# Display statistics and distribution of the target variable 'SalePrice'
print(data['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.histplot(data['SalePrice'], color='g', bins=100, kde=True, alpha=0.4)

# Filter numerical columns and display their distributions
numerical_data = data.select_dtypes(include=['float64', 'int64'])
print(numerical_data.head())
numerical_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# Split the dataset into training and testing sets
def split_data(df, test_ratio=0.30):
    test_indices = np.random.rand(len(df)) < test_ratio
    return df[~test_indices], df[test_indices]

train_data, test_data = split_data(data)
print(f"Training examples: {len(train_data)}, Testing examples: {len(test_data)}")

# Convert the data to TensorFlow Dataset format
label_column = 'SalePrice'
train_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=label_column, task=tfdf.keras.Task.REGRESSION)
test_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label=label_column, task=tfdf.keras.Task.REGRESSION)

# Initialize and train a Random Forest model
random_forest_model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
random_forest_model.compile(metrics=["mse"])
random_forest_model.fit(train_tf_dataset)

# Visualize a tree from the trained Random Forest model
tfdf.model_plotter.plot_model_in_colab(random_forest_model, tree_idx=0, max_depth=3)

# Evaluate the model using Out of Bag (OOB) data and the test dataset
training_logs = random_forest_model.make_inspector().training_logs()
plt.plot([log.num_trees for log in training_logs], [log.evaluation.rmse for log in training_logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (OOB)")
plt.show()

# Display general statistics from OOB data
model_inspector = random_forest_model.make_inspector()
print(model_inspector.evaluation())

# Evaluate the model on the test dataset
evaluation_results = random_forest_model.evaluate(test_tf_dataset, return_dict=True)
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")

# Display variable importances
print("Available variable importances:")
for importance_type in model_inspector.variable_importances().keys():
    print(f"\t{importance_type}")

# Plot the variable importances
variable_importances = model_inspector.variable_importances()["NUM_AS_ROOT"]
features = [vi[0].name for vi in variable_importances]
importances = [vi[1] for vi in variable_importances]

plt.figure(figsize=(12, 4))
bars = plt.barh(range(len(features)), importances)
plt.yticks(range(len(features)), features)
plt.gca().invert_yaxis()

for bar, importance in zip(bars, importances):
    plt.text(bar.get_width(), bar.get_y(), f"{importance:.4f}", va='top')

plt.xlabel("NUM_AS_ROOT")
plt.title("Variable Importances")
plt.tight_layout()
plt.show()

# Prepare for submission
test_file_path = "../input/house-prices-advanced-regression-techniques/test.csv"
test_df = pd.read_csv(test_file_path)
ids = test_df.pop('Id')

test_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.Task.REGRESSION)
predictions = random_forest_model.predict(test_tf_dataset).squeeze()

submission_df = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
print(submission_df.head())

sample_submission_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = predictions
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print(sample_submission_df.head())
