from sklearn.metrics import classification_report

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    # Generate and print a classification report
    report = classification_report(y_true, y_pred)
    print(report)
