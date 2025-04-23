from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from knn_classifier_training import train_pipeline
from evaluation_model import evaluation_model
from prediction_model import prediction

def recovery_dataset():
    adult = fetch_ucirepo(id=2) 
    
    X = adult.data.features 
    y = adult.data.targets     

    return X, y

def check_null(X, y):
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    return X_clean, y_clean

def column_identification(X_clean, y_clean):

    categorical_cols = X_clean.select_dtypes(include=['object']).columns
    numeric_cols = X_clean.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    return preprocessor

def main():
    try: 
        X, y = recovery_dataset()
        X_clean, y_clean = check_null(X, y)
        preprocessor = column_identification(X_clean, y_clean)
        pipe, X_test, y_test = train_pipeline(X_clean, y_clean, preprocessor) 
        prediction = prediction(pipe, X_test, y_test)
        evaluation_model(y_test, prediction)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    print("End of execution.")

