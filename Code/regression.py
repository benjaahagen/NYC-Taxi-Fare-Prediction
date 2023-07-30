import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Define preprocessing function
def preprocess_data(data, columns_to_drop):
    # Convert datetime columns to pandas datetime
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
    data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

    # Calculate trip duration in minutes
    data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Convert store_and_fwd_flag to numerical values
    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

    # Drop unnecessary columns
    data = data.drop(columns=columns_to_drop)

    return data

# Load the data
main_sample = pd.read_csv('Main Sample.csv')
new_sample = pd.read_csv('New Sample.csv')

# Columns to drop
columns_to_drop = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

# Preprocess the data
main_sample = preprocess_data(main_sample, columns_to_drop)
new_sample = preprocess_data(new_sample, columns_to_drop)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(main_sample.drop(columns=['total_amount']), main_sample['total_amount'], test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE and R2 scores
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Predict on the new sample data
new_sample_pred = model.predict(new_sample)

# Add the predicted values to a new DataFrame and display it
new_sample_result = new_sample.copy()
new_sample_result['predicted_total_amount'] = new_sample_pred
print(new_sample_result)

# Calculate the correlation matrix
corr_matrix = main_sample.corr()

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt='.2f', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 12})

# Set the title and its properties
plt.title('Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)

# Save the map as an image
plt.savefig('Correlation_Heatmap.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('Actual vs Predicted Values', fontsize=16)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Save the plot as an image
plt.savefig('scatter_actual_vs_predicted.png', dpi=300, bbox_inches='tight')

plt.show()

# Histogram of Predicted Values
plt.figure(figsize=(8, 6))
sns.histplot(new_sample_result['predicted_total_amount'], bins=20, color='skyblue', edgecolor='black', linewidth=1.2, kde=True)
plt.xlabel('Predicted Total Amount', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Predicted Values', fontsize=16)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# Save the plot as an image
plt.savefig('histogram_predicted_values.png', dpi=300, bbox_inches='tight')

plt.show()
