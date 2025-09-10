import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
from optuna import Trial
import joblib
import warnings
from elo_system import FootballEloSystem
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class EnhancedFootballPredictor:
    """
    Enhanced Football Predictor inspired by 85% accuracy tennis prediction model.
    
    Key improvements:
    - ELO rating system (most important feature in tennis model)
    - XGBoost optimization (performed best in tennis: 85% vs 76% RF)
    - Competition-specific features (like surface-specific in tennis)
    - Aggressive hyperparameter tuning
    """
    
    def __init__(self):
        self.elo_system = FootballEloSystem()
        self.best_model = None
        self.best_score = 0
        self.feature_columns = None
        self.data_dir = '../data'
        self.models_dir = '../models'
        self.target_accuracy = 0.85  # Tennis model achieved 85%
        
    def prepare_elo_enhanced_features(self, match_results):
        """
        Create enhanced features with ELO system (key insight from tennis model)
        """
        print("Building ELO ratings from match history...")
        
        # Build ELO system from all matches
        self.elo_system.build_from_match_data(match_results)
        
        enhanced_features = []
        
        print("Creating ELO-enhanced features for each match...")
        
        for _, match in match_results.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            competition = match.get('competition', 'premier_league')
            
            # Get ELO features for both teams (inspired by tennis ELO features)
            home_elo_features = self.elo_system.get_team_elo_features(home_team, competition)
            away_elo_features = self.elo_system.get_team_elo_features(away_team, competition)
            
            # Calculate ELO differences (most important feature in tennis model)
            elo_difference = self.elo_system.get_elo_difference(home_team, away_team, competition, 'home')
            overall_elo_diff = home_elo_features['overall_elo'] - away_elo_features['overall_elo']
            home_elo_diff = home_elo_features['home_elo'] - away_elo_features['away_elo']
            
            # Get win probabilities
            win_probs = self.elo_system.get_win_probability(home_team, away_team, competition, 'home')
            
            # Enhanced feature set (tennis-inspired)
            match_features = {
                # Target variables
                'target_result': match['result'],
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                
                # ELO FEATURES (MOST IMPORTANT - like tennis)
                'elo_difference': elo_difference,
                'overall_elo_difference': overall_elo_diff,
                'home_away_elo_difference': home_elo_diff,
                
                # Home team ELO features
                'home_overall_elo': home_elo_features['overall_elo'],
                'home_home_elo': home_elo_features['home_elo'],
                'home_recent_elo_change': home_elo_features['recent_elo_change'],
                'home_form_trend': home_elo_features['recent_form_trend'],
                'home_recent_win_rate': home_elo_features['recent_win_rate'],
                
                # Away team ELO features  
                'away_overall_elo': away_elo_features['overall_elo'],
                'away_away_elo': away_elo_features['away_elo'],
                'away_recent_elo_change': away_elo_features['recent_elo_change'],
                'away_form_trend': away_elo_features['recent_form_trend'],
                'away_recent_win_rate': away_elo_features['recent_win_rate'],
                
                # Win probability features (confidence indicators)
                'home_win_probability': win_probs['home_win'],
                'draw_probability': win_probs['draw'],
                'away_win_probability': win_probs['away_win'],
                
                # Competition-specific features (like surface in tennis)
                'competition_weight': self.elo_system.competition_weights.get(competition.lower(), 32),
                'is_high_importance': 1 if competition.lower() in ['world_cup', 'euros', 'champions_league'] else 0,
                
                # Form comparison features
                'form_difference': home_elo_features['recent_form_trend'] - away_elo_features['recent_form_trend'],
                'win_rate_difference': home_elo_features['recent_win_rate'] - away_elo_features['recent_win_rate'],
                'elo_change_difference': home_elo_features['recent_elo_change'] - away_elo_features['recent_elo_change'],
                
                # Home advantage indicators
                'home_advantage_factor': 1,
                'venue_advantage': home_elo_features['home_elo'] - away_elo_features['away_elo']
            }
            
            # Add competition-specific ELO if available
            comp_key = f'{competition.lower()}_elo'
            if comp_key in home_elo_features and comp_key in away_elo_features:
                match_features[f'{competition}_elo_difference'] = (
                    home_elo_features[comp_key] - away_elo_features[comp_key]
                )
                match_features[f'home_{comp_key}'] = home_elo_features[comp_key]
                match_features[f'away_{comp_key}'] = away_elo_features[comp_key]
            
            enhanced_features.append(match_features)
        
        return pd.DataFrame(enhanced_features)
    
    def optimize_xgboost_aggressive(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Aggressive XGBoost optimization (tennis model achieved 85% with XGBoost)
        """
        print(f"🎯 Aggressive XGBoost optimization targeting 85% accuracy...")
        print(f"Running {n_trials} trials (tennis model used extensive tuning)")
        
        # Convert string labels to numeric
        label_map = {'H': 0, 'D': 1, 'A': 2}
        y_train_numeric = y_train.map(label_map)
        y_val_numeric = y_val.map(label_map)
        
        def objective(trial: Trial):
            # More aggressive parameter space (inspired by tennis success)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 2.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'tree_method': 'auto',
                'objective': 'multi:softprob'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train_numeric, 
                     eval_set=[(X_val, y_val_numeric)],
                     verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val_numeric, y_pred)
            
            return accuracy
        
        # Optimize with aggressive settings
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best trial accuracy: {study.best_value:.4f}")
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            'random_state': 42, 
            'eval_metric': 'mlogloss',
            'tree_method': 'auto',
            'objective': 'multi:softprob'
        })
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train_numeric,
                       eval_set=[(X_val, y_val_numeric)],
                       verbose=False)
        
        y_pred = final_model.predict(X_val)
        reverse_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred_labels = [reverse_map[pred] for pred in y_pred]
        
        accuracy = accuracy_score(y_val, y_pred_labels)
        
        print(f"🚀 Final optimized XGBoost accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy >= self.target_accuracy:
            print(f"🎉 TARGET ACHIEVED! {accuracy:.1%} >= {self.target_accuracy:.1%}")
        else:
            print(f"🎯 Still need {(self.target_accuracy - accuracy)*100:.1f} more percentage points")
        
        print(f"🏆 Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return final_model, accuracy, best_params, study
    
    def analyze_feature_importance(self, model, feature_names):
        """
        Analyze feature importance like tennis model
        (tennis found: ELO, surface difference, total ELO most important)
        """
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS (Tennis-style)")
        print("="*60)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        for i in range(min(15, len(feature_importance_df))):
            feat = feature_importance_df.iloc[i]
            print(f"{i+1:2d}. {feat['feature']:<30} {feat['importance']:.4f}")
        
        # Check if ELO features dominate (like in tennis)
        elo_features = feature_importance_df[
            feature_importance_df['feature'].str.contains('elo', case=False, na=False)
        ]
        
        total_elo_importance = elo_features['importance'].sum()
        print(f"\n🎾 ELO Features Total Importance: {total_elo_importance:.4f} ({total_elo_importance/importance.sum()*100:.1f}%)")
        
        if total_elo_importance > 0.4:  # If ELO features account for >40% of importance
            print("✅ ELO features are dominant (like in tennis model)")
        else:
            print("⚠️  ELO features may need more weight (tennis model was ELO-driven)")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances (Tennis-Inspired Model)')
        plt.gca().invert_yaxis()
        
        # Highlight ELO features
        for i, feat in enumerate(top_features['feature']):
            if 'elo' in feat.lower():
                plt.gca().get_yticklabels()[i].set_color('red')
                plt.gca().get_yticklabels()[i].set_weight('bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/feature_importance_tennis_inspired.png', dpi=300)
        plt.show()
        
        return feature_importance_df
    
    def train_tennis_inspired_model(self):
        """
        Main training function inspired by tennis prediction success
        """
        print("🎾 TENNIS-INSPIRED FOOTBALL PREDICTION TRAINING")
        print("="*60)
        print("Target: 85% accuracy (matching tennis prediction performance)")
        
        # Load and prepare data with ELO features
        print("\n1️⃣  Loading and preparing enhanced data...")
        try:
            match_results = pd.read_csv(f'{self.data_dir}/all_match_results.csv')
        except FileNotFoundError:
            print("❌ Match results not found. Please run data_collector.py first.")
            return None, None
        
        # Create ELO-enhanced features
        enhanced_features = self.prepare_elo_enhanced_features(match_results)
        
        # Prepare training data
        feature_columns = [col for col in enhanced_features.columns 
                          if col not in ['target_result', 'date', 'home_team', 'away_team']]
        
        X = enhanced_features[feature_columns].fillna(0)
        y = enhanced_features['target_result']
        
        self.feature_columns = feature_columns
        
        print(f"📊 Dataset prepared:")
        print(f"   • Matches: {len(X):,}")
        print(f"   • Features: {len(feature_columns)} (ELO-enhanced)")
        print(f"   • Classes: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n2️⃣  Training/Validation split:")
        print(f"   • Training: {len(X_train):,} matches")
        print(f"   • Validation: {len(X_val):,} matches")
        
        # Aggressive XGBoost optimization (tennis model's best performer)
        print(f"\n3️⃣  XGBoost optimization (tennis model achieved 85% with XGBoost):")
        
        best_model, best_accuracy, best_params, study = self.optimize_xgboost_aggressive(
            X_train, y_train, X_val, y_val, n_trials=150
        )
        
        self.best_model = best_model
        self.best_score = best_accuracy
        
        # Detailed evaluation
        print(f"\n4️⃣  Detailed model evaluation:")
        self.evaluate_detailed(X_val, y_val, best_model)
        
        # Feature importance analysis (tennis-style)
        print(f"\n5️⃣  Feature importance analysis:")
        feature_importance_df = self.analyze_feature_importance(best_model, feature_columns)
        
        # Save everything
        print(f"\n6️⃣  Saving models and results...")
        self.save_enhanced_model(best_model, best_params, feature_importance_df, study)
        
        # Final results
        print(f"\n🏁 FINAL RESULTS")
        print(f"="*60)
        print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"🎾 Tennis Target: 85%")
        print(f"📈 Achievement: {'🎉 TARGET REACHED!' if best_accuracy >= 0.85 else f'Need +{(0.85-best_accuracy)*100:.1f}pp'}")
        
        return best_model, best_accuracy
    
    def evaluate_detailed(self, X_val, y_val, model):
        """Detailed evaluation like tennis model"""
        y_pred_numeric = model.predict(X_val)
        reverse_map = {0: 'H', 1: 'D', 2: 'A'}
        y_pred = [reverse_map[pred] for pred in y_pred_numeric]
        
        print("📊 Classification Report:")
        print(classification_report(y_val, y_pred))
        
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_val, y_pred, labels=['H', 'D', 'A'])
        print("     Predicted")
        print("        H    D    A")
        for i, label in enumerate(['H', 'D', 'A']):
            print(f"Actual {label} {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")
        
        # Per-class accuracies
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        print(f"\n📊 Per-class Accuracies:")
        print(f"   Home Win (H): {class_accuracies[0]:.3f}")
        print(f"   Draw (D):     {class_accuracies[1]:.3f}")
        print(f"   Away Win (A): {class_accuracies[2]:.3f}")
    
    def save_enhanced_model(self, model, params, feature_importance, study):
        """Save enhanced model and metadata"""
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save model
        joblib.dump(model, f"{self.models_dir}/tennis_inspired_xgboost.pkl")
        joblib.dump(self.feature_columns, f"{self.models_dir}/tennis_inspired_features.pkl")
        
        # Save ELO system
        self.elo_system.save_elo_data(f"{self.models_dir}/elo_system.json")
        
        # Save metadata
        metadata = {
            'accuracy': self.best_score,
            'target_accuracy': self.target_accuracy,
            'best_params': params,
            'n_features': len(self.feature_columns),
            'model_type': 'XGBoost (Tennis-Inspired)',
            'feature_importance': feature_importance.head(20).to_dict('records'),
            'optimization_trials': study.n_trials,
            'best_trial_number': study.best_trial.number
        }
        
        import json
        with open(f"{self.models_dir}/tennis_inspired_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Enhanced model saved to {self.models_dir}/")

def main():
    predictor = EnhancedFootballPredictor()
    
    print("🚀 Starting Tennis-Inspired Football Prediction Training")
    print("🎾 Target: Match 85% accuracy achieved in tennis prediction")
    
    best_model, best_accuracy = predictor.train_tennis_inspired_model()
    
    if best_model:
        if best_accuracy >= 0.85:
            print(f"\n🏆 SUCCESS! Achieved {best_accuracy:.1%} accuracy!")
            print("🎾 Matched tennis prediction performance!")
        else:
            print(f"\n🎯 Good progress: {best_accuracy:.1%} accuracy")
            print(f"💪 Continue optimizing to reach 85% target")

if __name__ == "__main__":
    main()