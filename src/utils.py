# src/utils.py
import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save object to pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load object from pickle file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate models using GridSearchCV
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[model_name]
            
            # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            
            # Set best parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)