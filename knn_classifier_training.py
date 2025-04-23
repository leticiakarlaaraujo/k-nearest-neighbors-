from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

def train_pipeline(X_clean, y_clean, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', KNeighborsClassifier())
    ])

    try:
        param_grid_gs = {'knn__n_neighbors': list(range(3, 16))}

        grid_search = GridSearchCV(pipeline, param_grid_gs, cv=2, scoring='accuracy')
        print("GridSearchCV: Starting the fit...")
        grid_search.fit(X_train, y_train.values.ravel())  
        print("GridSearchCV: Fit completed.")

        print("GridSearchCV: Best k:", grid_search.best_params_)
        print("GridSearchCV: Best score:", grid_search.best_score_)
        
        return grid_search, X_test, y_test
    
    except Exception as e:
        print(f"Error during GridSearchCV: {e}")