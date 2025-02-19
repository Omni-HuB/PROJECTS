import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=names)

# # Checking for any missing values after imputation
# print("Missing values in the dataset before imputation:")
# print(data.isnull().sum())

# Handling missing values
data = data.replace('?', pd.NA)
data = data.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the mean
data.fillna(data.mean(), inplace=True)

# Checking for any missing values after imputation
print("Missing values in the dataset after imputation:")
print(data.isnull().sum())

# Convert categorical variables to dummy/indicator variables if required
data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('target', axis=1))
data = pd.DataFrame(scaled_data, columns=data.columns[:-1])
data['target'] = data['target'].astype(int)


# Data distribution visualization

# Pairplot for visualizing relationships between variables
sns.pairplot(data, hue='target')
plt.title('Pairplot for the Dataset')
plt.show()

# Correlation heatmap to identify correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Split the dataset into train and test sets
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(X_train)
print(X_test)

print()
print()

print(y_train)
print(y_test)

# Train decision trees with 'entropy' and 'gini impurity'
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
accuracy_entropy = dt_entropy.score(X_test, y_test)

dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)
accuracy_gini = dt_gini.score(X_test, y_test)


print(accuracy_entropy)
print(accuracy_gini)

best_criterion = 'entropy' if accuracy_entropy > accuracy_gini else 'gini'

print(f"Best criterion for attribute selection: {best_criterion}")

# Hyperparameter search for decision trees using Grid Search
param_grid = {
    'min_samples_split': [20, 50, 100],
    'max_features': [ 'sqrt', 'log2',None]
}

grid_search = GridSearchCV(DecisionTreeClassifier(criterion=best_criterion, random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best combination of hyperparameters: {best_params}")

# Extract the available options for max_features in Decision Tree
max_features_options_dt = X.shape[1] if best_params['max_features'] is None else best_params['max_features']
print(f"Available options for max_features in Decision Tree: {max_features_options_dt}")

# Train a random forest classifier and perform Grid Search
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 50, 100],
    'min_samples_split': [50, 100, 20],
    'max_features': ['sqrt', 'log2',None]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)
best_rf_params = rf_grid_search.best_params_
print(f"Best combination of hyperparameters for Random Forest: {best_rf_params}")

# Extract the available options for max_features in Random Forest
max_features_options_rf = X.shape[1] if best_rf_params['max_features'] is None else best_rf_params['max_features']
print(f"Available options for max_features in Random Forest: {max_features_options_rf}")

# Training the final Random Forest Classifier
rf_classifier = RandomForestClassifier(**best_rf_params, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Classification report for the Random Forest Classifier:")
print(classification_report(y_test, y_pred))
