import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import optuna
from optuna import Trial
import warnings
warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_columns = None
        self.data_dir = '../data'
        self.models_dir = '../models'
        
    def load_and_prepare_data(self):
        """Load and prepare the training data"""
        print("Loading feature data...")
        
        # Load features
        features_df = pd.read_csv(f'{self.data_dir}/match_features.csv')
        
        # Remove non-numeric columns for training
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['date', 'home_team', 'away_team', 'target_result', 
                                            'target_home_score', 'target_away_score']]
        
        X = features_df[self.feature_columns]
        
        # Create target variables
        # For match result (H/D/A)
        y_result = features_df['target_result']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        print(f"Features shape: {X.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X, y_result, features_df
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Convert string labels to numeric
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_train_numeric = y_train.map(label_map)
        y_val_numeric = y_val.map(label_map)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train_numeric,
                 eval_set=[(X_val, y_val_numeric)],
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_val)
        # Convert back to string labels
        reverse_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred_labels = [reverse_map[pred] for pred in y_pred]
        
        accuracy = accuracy_score(y_val, y_pred_labels)
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        
        self.models['xgboost'] = model
        return model, accuracy
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        # Convert string labels to numeric
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_train_numeric = y_train.map(label_map)
        y_val_numeric = y_val.map(label_map)
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train_numeric,
                 eval_set=[(X_val, y_val_numeric)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        # Predictions
        y_pred = model.predict(X_val)
        # Convert back to string labels
        reverse_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred_labels = [reverse_map[pred] for pred in y_pred]
        
        accuracy = accuracy_score(y_val, y_pred_labels)
        print(f"LightGBM Accuracy: {accuracy:.4f}")
        
        self.models['lightgbm'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        
        self.models['random_forest'] = model
        return model, accuracy
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting model...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
        
        self.models['gradient_boosting'] = model
        return model, accuracy
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimize XGBoost hyperparameters using Optuna"""
        print(f"Optimizing XGBoost with {n_trials} trials...")
        
        # Convert string labels to numeric
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_train_numeric = y_train.map(label_map)
        y_val_numeric = y_val.map(label_map)
        
        def objective(trial: Trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_numeric, verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val_numeric, y_pred)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Train best model
        best_params = study.best_params
        best_params.update({'random_state': 42, 'eval_metric': 'mlogloss'})
        
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train_numeric, verbose=False)
        
        y_pred = best_model.predict(X_val)
        reverse_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred_labels = [reverse_map[pred] for pred in y_pred]
        
        accuracy = accuracy_score(y_val, y_pred_labels)
        print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
        print(f"Best parameters: {best_params}")
        
        self.models['xgboost_optimized'] = best_model
        return best_model, accuracy
    
    def train_all_models(self, optimize=True):
        """Train all models and find the best one"""
        X, y, features_df = self.load_and_prepare_data()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        model_scores = {}
        
        # Train all models
        models_to_train = [
            ('Random Forest', self.train_random_forest),
            ('Gradient Boosting', self.train_gradient_boosting),
            ('XGBoost', self.train_xgboost),
            ('LightGBM', self.train_lightgbm)
        ]
        
        if optimize:
            models_to_train.append(('XGBoost Optimized', self.optimize_xgboost))
        
        for model_name, train_func in models_to_train:
            try:
                model, accuracy = train_func(X_train, y_train, X_val, y_val)
                model_scores[model_name] = accuracy
                
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.best_model_name = model_name
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        print("\n=== Model Comparison ===")
        for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name}: {score:.4f}")
        
        print(f"\nBest model: {self.best_model_name} with accuracy {self.best_score:.4f}")
        
        # Detailed evaluation of best model
        self.evaluate_best_model(X_val, y_val)
        
        # Save models
        self.save_models()
        
        return self.best_model, self.best_score
    
    def evaluate_best_model(self, X_val, y_val):
        """Detailed evaluation of the best model"""
        print(f"\n=== Detailed Evaluation of {self.best_model_name} ===")
        
        if 'xgboost' in self.best_model_name.lower():
            # For XGBoost models, convert predictions
            y_pred_numeric = self.best_model.predict(X_val)
            reverse_map = {0: 'H', 1: 'D', 2: 'A'}
            y_pred = [reverse_map[pred] for pred in y_pred_numeric]
        else:
            y_pred = self.best_model.predict(X_val)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_val, y_pred, labels=['H', 'D', 'A'])
        print("    H   D   A")
        for i, row in enumerate(cm):
            print(f"{['H', 'D', 'A'][i]} {row[0]:3d} {row[1]:3d} {row[2]:3d}")
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\nTop 10 Most Important Features:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i in range(min(10, len(feature_importance))):
                feat = feature_importance.iloc[i]
                print(f"{feat['feature']}: {feat['importance']:.4f}")
    
    def save_models(self):
        """Save all trained models"""
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{self.models_dir}/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save best model separately
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{self.models_dir}/best_model.pkl")
            joblib.dump(self.feature_columns, f"{self.models_dir}/feature_columns.pkl")
            print(f"Saved best model ({self.best_model_name}) to {self.models_dir}/best_model.pkl")
    
    def load_best_model(self):
        """Load the best trained model"""
        try:
            self.best_model = joblib.load(f"{self.models_dir}/best_model.pkl")
            self.feature_columns = joblib.load(f"{self.models_dir}/feature_columns.pkl")
            print("Best model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    predictor = FootballPredictor()
    
    print("Starting model training...")
    print("Target: Achieve 85%+ accuracy like the tennis prediction model")
    
    best_model, best_score = predictor.train_all_models(optimize=True)
    
    print(f"\n=== Training Complete ===")
    print(f"Best accuracy achieved: {best_score:.4f} ({best_score*100:.2f}%)")
    
    if best_score >= 0.85:
        print(" TARGET ACHIEVED! 85%+ accuracy reached!")
    else:
        print(f"Target not yet reached. Need to improve by {(0.85 - best_score)*100:.2f} percentage points.")