from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def evaluation_model(y_test, y_pred ):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))