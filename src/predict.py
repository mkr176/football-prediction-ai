import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')

class MatchPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineer()
        self.models_dir = '../models'
        self.data_dir = '../data'
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(f"{self.models_dir}/best_model.pkl")
            self.feature_columns = joblib.load(f"{self.models_dir}/feature_columns.pkl")
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first using train_models.py")
            return False
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict the outcome of a single match"""
        if self.model is None:
            if not self.load_model():
                return None
        
        if match_date is None:
            match_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Predicting: {home_team} vs {away_team} on {match_date}")
        
        # Load historical data
        try:
            match_results = pd.read_csv(f'{self.data_dir}/all_match_results.csv')
            team_stats = pd.read_csv(f'{self.data_dir}/all_team_stats.csv')
        except FileNotFoundError:
            print("Error: Data files not found. Please run data_collector.py first.")
            return None
        
        # Calculate features for this match
        home_form = self.feature_engineer.calculate_form_features(
            match_results, home_team, match_date
        )
        away_form = self.feature_engineer.calculate_form_features(
            match_results, away_team, match_date
        )
        
        home_season = self.feature_engineer.get_team_season_stats(
            team_stats, home_team, '2023-24'
        )
        away_season = self.feature_engineer.get_team_season_stats(
            team_stats, away_team, '2023-24'
        )
        
        h2h = self.feature_engineer.calculate_head_to_head(
            match_results, home_team, away_team, match_date
        )
        
        # Create feature vector
        match_features = {
            'home_recent_form_points': home_form['recent_form_points'],
            'home_recent_goals_for': home_form['recent_goals_for'],
            'home_recent_goals_against': home_form['recent_goals_against'],
            'home_recent_goal_difference': home_form['recent_goal_difference'],
            'home_points_per_game': home_season['points_per_game'],
            'home_goals_for_per_game': home_season['goals_for_per_game'],
            'home_goals_against_per_game': home_season['goals_against_per_game'],
            'home_win_percentage': home_season['win_percentage'],
            'home_possession': home_season['avg_possession'],
            'home_shots_per_game': home_season['shots_per_game'],
            'home_pass_accuracy': home_season['pass_accuracy'],
            
            'away_recent_form_points': away_form['recent_form_points'],
            'away_recent_goals_for': away_form['recent_goals_for'],
            'away_recent_goals_against': away_form['recent_goals_against'],
            'away_recent_goal_difference': away_form['recent_goal_difference'],
            'away_points_per_game': away_season['points_per_game'],
            'away_goals_for_per_game': away_season['goals_for_per_game'],
            'away_goals_against_per_game': away_season['goals_against_per_game'],
            'away_win_percentage': away_season['win_percentage'],
            'away_possession': away_season['avg_possession'],
            'away_shots_per_game': away_season['shots_per_game'],
            'away_pass_accuracy': away_season['pass_accuracy'],
            
            'h2h_home_wins': h2h['h2h_home_wins'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'h2h_draws': h2h['h2h_draws'],
            'h2h_home_goals_avg': h2h['h2h_home_goals_avg'],
            'h2h_away_goals_avg': h2h['h2h_away_goals_avg'],
            
            'form_difference': home_form['recent_form_points'] - away_form['recent_form_points'],
            'points_per_game_difference': home_season['points_per_game'] - away_season['points_per_game'],
            'goal_difference_difference': home_season['goal_difference_per_game'] - away_season['goal_difference_per_game'],
            'possession_difference': home_season['avg_possession'] - away_season['avg_possession'],
            'home_advantage': 1
        }
        
        # Ensure all required features are present
        feature_vector = []
        for col in self.feature_columns:
            if col in match_features:
                feature_vector.append(match_features[col])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Make prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            
            # Handle XGBoost numeric predictions
            if isinstance(prediction, (int, np.integer)):
                result_map = {0: 'H', 1: 'D', 2: 'A'}
                prediction = result_map[prediction]
                prob_dict = {'H': probabilities[0], 'D': probabilities[1], 'A': probabilities[2]}
            else:
                prob_dict = {
                    'H': probabilities[0] if prediction == 'H' else probabilities[1] if len(probabilities) > 1 else 0,
                    'D': probabilities[1] if prediction == 'D' else probabilities[0] if len(probabilities) > 1 else 0,
                    'A': probabilities[2] if prediction == 'A' else probabilities[0] if len(probabilities) > 1 else 0
                }
        else:
            prob_dict = {'H': 0.33, 'D': 0.33, 'A': 0.33}
        
        # Format results
        results = {
            'prediction': prediction,
            'confidence': max(prob_dict.values()) if prob_dict else 0.33,
            'probabilities': prob_dict,
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date
        }
        
        self._print_prediction_results(results)
        return results
    
    def _print_prediction_results(self, results):
        """Print formatted prediction results"""
        print(f"\n=== MATCH PREDICTION ===")
        print(f"Match: {results['home_team']} vs {results['away_team']}")
        print(f"Date: {results['match_date']}")
        print(f"\nPredicted Result: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print(f"\nProbabilities:")
        prob = results['probabilities']
        print(f"  Home Win (H): {prob.get('H', 0):.2%}")
        print(f"  Draw (D):     {prob.get('D', 0):.2%}")
        print(f"  Away Win (A): {prob.get('A', 0):.2%}")
        
        # Interpretation
        prediction = results['prediction']
        confidence = results['confidence']
        
        if prediction == 'H':
            outcome = f"{results['home_team']} to win"
        elif prediction == 'A':
            outcome = f"{results['away_team']} to win"
        else:
            outcome = "Draw"
        
        confidence_level = "High" if confidence > 0.6 else "Medium" if confidence > 0.45 else "Low"
        
        print(f"\n📊 Prediction: {outcome}")
        print(f"🎯 Confidence Level: {confidence_level}")
    
    def predict_multiple_matches(self, matches):
        """Predict multiple matches"""
        predictions = []
        
        for match in matches:
            home_team = match.get('home_team')
            away_team = match.get('away_team')
            match_date = match.get('date')
            
            result = self.predict_match(home_team, away_team, match_date)
            if result:
                predictions.append(result)
        
        return predictions
    
    def predict_world_cup_matches(self):
        """Predict World Cup matches - example fixtures"""
        print("🏆 WORLD CUP PREDICTIONS 🏆")
        
        # Example World Cup matches (you can update these)
        world_cup_matches = [
            {'home_team': 'Brazil', 'away_team': 'Argentina', 'date': '2024-12-01'},
            {'home_team': 'France', 'away_team': 'Spain', 'date': '2024-12-02'},
            {'home_team': 'Germany', 'away_team': 'Italy', 'date': '2024-12-03'},
            {'home_team': 'England', 'away_team': 'Netherlands', 'date': '2024-12-04'},
            {'home_team': 'Portugal', 'away_team': 'Belgium', 'date': '2024-12-05'}
        ]
        
        predictions = self.predict_multiple_matches(world_cup_matches)
        
        print(f"\n=== WORLD CUP PREDICTIONS SUMMARY ===")
        for pred in predictions:
            outcome = pred['prediction']
            confidence = pred['confidence']
            
            if outcome == 'H':
                result = f"{pred['home_team']} wins"
            elif outcome == 'A':
                result = f"{pred['away_team']} wins"
            else:
                result = "Draw"
            
            print(f"{pred['home_team']} vs {pred['away_team']}: {result} ({confidence:.1%} confidence)")
        
        return predictions
    
    def evaluate_predictions(self, test_matches_file=None):
        """Evaluate model performance on test data"""
        if test_matches_file is None:
            print("No test file provided. Using validation logic...")
            return
        
        try:
            test_matches = pd.read_csv(test_matches_file)
            correct_predictions = 0
            total_predictions = 0
            
            for _, match in test_matches.iterrows():
                prediction = self.predict_match(
                    match['home_team'], 
                    match['away_team'], 
                    match['date']
                )
                
                if prediction and prediction['prediction'] == match['actual_result']:
                    correct_predictions += 1
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            print(f"\nEvaluation Results:")
            print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
            print(f"Accuracy: {accuracy:.2%}")
            
            return accuracy
            
        except Exception as e:
            print(f"Error evaluating predictions: {e}")

def main():
    predictor = MatchPredictor()
    
    # Example usage
    print("Football Match Prediction System")
    print("Inspired by 85% accuracy tennis predictions")
    
    while True:
        print("\nOptions:")
        print("1. Predict single match")
        print("2. Predict World Cup matches")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            home_team = input("Enter home team: ").strip()
            away_team = input("Enter away team: ").strip()
            match_date = input("Enter match date (YYYY-MM-DD) or press Enter for today: ").strip()
            
            if not match_date:
                match_date = datetime.now().strftime('%Y-%m-%d')
            
            predictor.predict_match(home_team, away_team, match_date)
            
        elif choice == '2':
            predictor.predict_world_cup_matches()
            
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()