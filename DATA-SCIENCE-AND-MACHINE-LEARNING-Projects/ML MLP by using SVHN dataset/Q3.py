import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import random




def load_data():
    url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    file_path = "train_32x32.mat"
    urllib.request.urlretrieve(url, file_path)

    # Assuming you have loaded the dataset into X and y
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


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
    data.fillna(data.mean(), inplace=True)  # Filling missing values with mean
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

def visualize_class_distribution(y_train):
    plt.figure(figsize=(12, 6))
    sns.countplot(y_train)
    plt.title("Distribution of Class Labels in Training Set")
    plt.show()

def visualize_samples(X_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        index = random.randint(0, len(X_train))
        plt.subplot(1, 5, i+1)
        plt.imshow(X_train[index])
        plt.title(f"Class: {y_train[index]}")
        plt.axis('off')
    plt.show()

def train_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100)

    param_grid = {
        'hidden_layer_sizes': [(100, 50), (150, 100), (50,)],
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'solver': ['adam', 'sgd'],
        'batch_size': [32, 64, 128],
    }

    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    activations = ['relu', 'tanh', 'logistic', 'identity']
    models = {}

    for activation in activations:
        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation=activation, max_iter=100)
        model.fit(X_train, y_train)
        models[activation] = model

    return models, best_params

def plot_training_loss(models):
    plt.figure(figsize=(10, 6))
    for activation, model in models.items():
        plt.plot(model.loss_curve_, label=activation)

    plt.title("Training Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.show()

def evaluate_best_model(models, best_params, X_test, y_test):
    best_model = models[best_params['activation']]
    accuracy = best_model.score(X_test, y_test)
    print(f"Best Accuracy on Test Set: {accuracy}")

def visualize_incorrect_predictions(X_test, y_test, best_model):
    y_pred = best_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(15, 10))

    for i in range(len(conf_matrix)):
        misclassified_indices = np.where((y_test != y_pred) & (y_test == i))[0][:3]
        for j, index in enumerate(misclassified_indices):
            plt.subplot(len(conf_matrix), 3, i*3 + j + 1)
            plt.imshow(X_test[index])
            plt.title(f"True: {y_test[index]}, Predicted: {y_pred[index]}")
            plt.axis('off')

    plt.show()

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    visualize_class_distribution(y_train)
    visualize_samples(X_train, y_train)

    models, best_params = train_model(X_train, y_train)
    plot_training_loss(models)
    analyze_model_effectiveness(models)
    evaluate_best_model(models, best_params, X_test, y_test)
    visualize_incorrect_predictions(X_test, y_test, models[best_params['activation']])

if __name__ == "__main__":
    main()
