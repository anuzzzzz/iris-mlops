import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    """Reads the Iris dataset from a URL."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(url, header=None, names=columns)
    return df

def train_model():
    """Loads data, trains a model, and evaluates it."""
    df = get_data()
    
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.savefig('confusion_matrix.png')
    
    return accuracy
