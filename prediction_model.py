def prediction(pipe, X_test, y_test):
    best_model = pipe.best_estimator_
    y_pred = best_model.predict(X_test)

    return y_pred, y_test 