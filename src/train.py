import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    """Loads the Iris dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(url, header=None, names=col_names)
    return df

def train_model():
    """Trains a model and returns accuracy. Also saves a confusion matrix plot."""
    df = get_data()
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    # Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    
    return accuracy
