# K-Nearest Neighbors (KNN) Classifier for Adult Census Income Prediction

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether an individual's income exceeds $50,000 per year based on the Adult Census Income dataset from the UCI Machine Learning Repository.

## Project Structure

The project is organized into the following Python files:

*   **`etl_pipeline.py`**: This file contains the main pipeline for data extraction, transformation, preprocessing, model training, prediction, and evaluation.
*   **`knn_classifier_training.py`**: This file handles the training of the KNN classifier using a pipeline and GridSearchCV for hyperparameter tuning.
*   **`prediction_model.py`**: This file contains the function responsible for making predictions using the trained model.
*   **`evaluation_model.py`**: This file contains functions for evaluating the performance of the model using various metrics.

## Data

The project utilizes the **Adult Census Income** dataset, which is publicly available from the UCI Machine Learning Repository (ID: 2). This dataset contains demographic and employment-related information about individuals, with the target variable being whether their income is above or below $50,000 per year.

## Dependencies

The following Python libraries are required to run this project:

*   `ucimlrepo`: For fetching the dataset from the UCI repository.
*   `scikit-learn`: For machine learning tasks, including:
    *   `train_test_split`: Splitting data into training and testing sets.
    *   `KNeighborsClassifier`: The KNN classifier.
    *   `Pipeline`: Creating a machine learning pipeline.
    *   `GridSearchCV`: Hyperparameter tuning.
    *   `StandardScaler`: Scaling numerical features.
    *   `OneHotEncoder`: Encoding categorical features.
    *   `ColumnTransformer`: Applying different transformations to different columns.
    *   `accuracy_score`, `precision_score`, `recall_score`, `classification_report`: Evaluation metrics.
*   `pandas`: For data manipulation and cleaning.

You can install these dependencies using pip:

```bash
pip install ucimlrepo scikit-learn pandas# k-nearest-neighbors-