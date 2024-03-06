# Analyzing Boston Housing Dataset
## Introduction:
The Boston Housing Dataset is a popular dataset used for regression analysis and predictive modeling. In this project, we'll perform exploratory data analysis and apply various machine learning techniques to analyze the dataset and make predictions about housing prices.

### 1. Exploratory Data Analysis (EDA):
Load the dataset and visualize the first few rows to understand its structure.
View basic statistics of the dataset to gain insights into the data distribution.
Create a histogram of the target variable (MEDV) to understand its distribution.
Generate a heatmap of the correlation matrix to identify relationships between features.
### 2. Machine Learning Models:
#### a. Regression:
Utilize a suitable regression model to predict housing prices.
Split the dataset into training and testing sets.
Train a Random Forest Regressor on the training data and evaluate its performance using mean squared error and R-squared score.
#### b. Clustering:
Apply K-Means clustering to identify patterns and groupings within the dataset.
Scale the data and fit a K-Means model with a specified number of clusters.
Print the cluster labels to analyze the clustering results.
#### c. Classification:
Train a Linear Regression classifier to predict housing prices.
Assess the accuracy of the classifier on the testing set using the R-squared score.
## Conclusion:
Through exploratory data analysis and the application of machine learning techniques, we gain valuable insights into the Boston Housing Dataset. By utilizing regression, clustering, and classification models, we're able to make predictions and analyze patterns within the data, providing valuable information for understanding housing prices in the Boston area.

## Code:
```python

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load the dataset
boston = load_boston()

# View the first few rows of the dataset
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df.head()

# View basic statistics of the dataset
df.describe()

# Exploratory Data Analysis
# Create a histogram of the target variable (MEDV)
sns.histplot(boston.target, kde=True)
plt.show()

# Create a heatmap of the correlation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()
# Regressor - Random Forest Regressor
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, boston.target, test_size=0.2, random_state=42)

# Create a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_regressor.predict(X_test)

# Calculate the mean squared error and R-squared score of the regressor
print('Mean squared error:', mean_squared_error(y_test, y_pred))
print('R-squared score:', r2_score(y_test, y_pred))

# Clusterer - K Means Clustering
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Print the cluster labels
print(labels)

# Classifier - Linear Regression
# Train a linear regression classifier
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = lr.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = r2_score(y_test, y_pred)
print('Accuracy:', accuracy)


```
## Output:
- View the first few rows of the dataset.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/a0582c2a-9992-4ce6-a006-5314f025fec5)

- View basic statistics of the dataset.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/5ea91dc3-a9ee-4f0b-ac82-e76c26f98f85)
  
- Create a histogram of the target variable (MEDV).
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/f3685d18-ab9c-41ea-a79a-1d9a8f98011d)

- Create a heatmap of the correlation matrix.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/7eead854-6f35-4748-b4e4-d1dd9fc2e75d)

- Train the regressor on the training data.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/283df05c-aca2-4683-b19c-aafa5460d189)

- Calculate the mean squared error and R-squared score of the regressor.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/0d2d76a7-0aa6-45dd-90c9-f05a10304286)

- Apply K-Means clustering. 
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/04050c85-74cd-4e7d-8fd1-35f27bc5e156)

- Print the cluster labels.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/de896c0e-34b6-4657-8638-6f195f67d4d4)

- Train a linear regression classifier and calculate the accuracy of the classifier.
![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/cd2ae814-6ef3-4150-90a9-74170f7cc567)

![image](https://github.com/KeerthikaNagarajan/Analyzing-Boston-Housing-Dataset/assets/93427089/4abd479f-1707-47c5-aae4-64fad3136f41)


## Summary:
This project demonstrates the application of machine learning techniques to analyze the Boston Housing Dataset, providing valuable insights into housing prices and patterns within the data. Through exploratory data analysis and model evaluation, we gain a deeper understanding of the factors influencing housing prices and their predictive capabilities.
