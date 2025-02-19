import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data_file(file_path):
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    data = pd.read_csv(file_path, names=names)
    return data

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    data = pd.read_csv(url, names=names)
    return data

def handle_missing_values(data):
    data = data.replace('?', pd.NA)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    return data

def preprocess_data(data):
    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop('num', axis=1))
    data = pd.DataFrame(scaled_data, columns=data.columns[:-1])
    data['num'] = data['num'].astype(int)
    return data

def visualize_data(data):
    sns.pairplot(data, hue='num')
    plt.title('Pairplot for the Dataset')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def split_data(data):
    X = data.drop('num', axis=1)
    y = data['num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_decision_trees(X_train, y_train, X_test, y_test):
    dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_entropy.fit(X_train, y_train)
    accuracy_entropy = dt_entropy.score(X_test, y_test)

    dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
    dt_gini.fit(X_train, y_train)
    accuracy_gini = dt_gini.score(X_test, y_test)

    print("Feature importances (Entropy):", dt_entropy.feature_importances_)
    print("Feature importances (Gini):", dt_gini.feature_importances_)

    if accuracy_entropy > accuracy_gini:
        best_criterion = 'entropy'
    else:
        best_criterion = 'gini'

    return best_criterion, accuracy_entropy, accuracy_gini


def grid_search_decision_trees(X_train, y_train, best_criterion):
    parameters_grid = {
        'min_samples_split': [20, 50, 100],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(criterion=best_criterion, random_state=42), parameters_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    max_features_options_dt = X_train.shape[1] if best_params['max_features'] is None else best_params['max_features']

    return best_params, max_features_options_dt

def grid_search_random_forest(X_train, y_train):

    randomf_param_grid = {
        'n_estimators': [50, 200, 100],
        'max_depth': [10, 50, 100],
        'min_samples_split': [50, 100, 20],
        'max_features': ['sqrt', 'log2', None]
    }

    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                                  
    randomf_param_grid, cv=5)

    rf_grid_search.fit(X_train, y_train)
    best_rf_params = rf_grid_search.best_params_
    max_features_options_rf = X_train.shape[1] if best_rf_params['max_features'] is None else best_rf_params['max_features']
    
    return best_rf_params, max_features_options_rf

def train_random_forest(X_train, y_train, X_test, y_test, best_rf_params):
    rf_classifier = RandomForestClassifier(**best_rf_params, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    print("Classification report for the Random Forest Classifier:")
    print(classification_report(y_test, y_pred))




if __name__ == '__main__':

  # file_path = 'your_local_file_path_here.csv'  # Replace with the correct file path
  #   data = load_data(file_path)
   
  ## loading data via online dataset  
    data = load_data()
    print("Missing values in the dataset after imputation:")
    print(data.isnull().sum())

    data = handle_missing_values(data)
    data = preprocess_data(data)
    visualize_data(data)

    X_train, X_test, y_train, y_test = split_data(data)

    print("X-train data")
    print(X_train)
    print()
    print()
    print("X-test data")
    print(X_test)
    print()
    print()
    print("Y-test data")
    print(y_test)
    print()
    print()
    print("Y-train data")
    print(y_train)


    print()
    print()

    best_criterion,accuracy_entropy,accuracy_gini = train_decision_trees(X_train, y_train, X_test, y_test)

    print(f"best criterion  : {best_criterion}")
    print(f"accuracy_entropy  : {accuracy_entropy}")
    print(f"accuacy_gini  : {accuracy_gini}")

    best_params, max_features_options_dt = grid_search_decision_trees(X_train, y_train, best_criterion)
    print(f"Best combination of hyperparameters: {best_params}")
    print(f"Available options for max_features in Decision Tree: {max_features_options_dt}")

    best_rf_params, max_features_options_rf = grid_search_random_forest(X_train, y_train)
    print(f"Best combination of hyperparameters for Random Forest: {best_rf_params}")
    print(f"Available options for max_features in Random Forest: {max_features_options_rf}")

    train_random_forest(X_train, y_train, X_test, y_test, best_rf_params)
    print()
    print()