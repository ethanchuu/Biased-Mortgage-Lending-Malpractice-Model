import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Clean and prepare features/target
def prepare_data(df, target_col='default_flag'):
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
def run_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model)
    return model, y_pred, y_test

# Decision Tree
def run_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model)
    return model, y_pred, y_test

# KNN
def run_knn(X_train, X_test, y_train, y_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, model)
    return model, y_pred, y_test

# Evaluation
def evaluate_model(y_true, y_pred, model):
    print("Model:", model.__class__.__name__)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Cross-validation
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    return scores

# Feature importance for Decision Tree
def plot_feature_importance(model, X_train):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_train.columns
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', bbox_inches='tight')
        print("Feature importance plot saved to feature_importance.png")
        plt.close()

# ROC Curve
def plot_roc_curve(model, X_test, y_test):
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'roc_curve_{model.__class__.__name__}.png')
        print(f"ROC curve saved to roc_curve_{model.__class__.__name__}.png")
        plt.close()

# Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No Default', 'Default'])
    plt.yticks(tick_marks, ['No Default', 'Default'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.savefig(f'confusion_matrix_{model_name}.png')
    print(f"Confusion matrix saved to confusion_matrix_{model_name}.png")
    plt.close()