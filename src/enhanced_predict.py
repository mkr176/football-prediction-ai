import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from elo_system import FootballEloSystem
import warnings
warnings.filterwarnings('ignore')

class EnhancedMatchPredictor:
    """
    Enhanced Match Predictor using tennis-inspired ELO system.
    Targets 85% accuracy like the successful tennis prediction model.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.elo_system = FootballEloSystem()
        self.models_dir = '../models'
        self.data_dir = '../data'
        self.confidence_threshold = 0.75  # High confidence threshold like tennis model
        
    def load_enhanced_model(self):
        """Load the tennis-inspired enhanced model"""
        try:
            # Load the enhanced XGBoost model
            self.model = joblib.load(f"{self.models_dir}/tennis_inspired_xgboost.pkl")
            self.feature_columns = joblib.load(f"{self.models_dir}/tennis_inspired_features.pkl")
            
            # Load ELO system
            self.elo_system.load_elo_data(f"{self.models_dir}/elo_system.json")
            
            print(" Enhanced tennis-inspired model loaded successfully")
            print(f" Model features: {len(self.feature_columns)} (ELO-enhanced)")
            return True
        except Exception as e:
            print(f" Error loading enhanced model: {e}")
            print("Please train the enhanced model first using enhanced_train_models.py")
            return False
    
    def create_prediction_features(self, home_team, away_team, competition='premier_league', match_date=None):
        """
        Create tennis-inspired features for prediction
        """
        if match_date is None:
            match_date = datetime.now()
        
        # Get ELO features for both teams
        home_elo_features = self.elo_system.get_team_elo_features(home_team, competition)
        away_elo_features = self.elo_system.get_team_elo_features(away_team, competition)
        
        # Calculate key ELO differences (most important features in tennis model)
        elo_difference = self.elo_system.get_elo_difference(home_team, away_team, competition, 'home')
        overall_elo_diff = home_elo_features['overall_elo'] - away_elo_features['overall_elo']
        home_elo_diff = home_elo_features['home_elo'] - away_elo_features['away_elo']
        
        # Get win probabilities
        win_probs = self.elo_system.get_win_probability(home_team, away_team, competition, 'home')
        
        # Create comprehensive feature vector (tennis-inspired)
        features = {
            # Core ELO features (most important in tennis)
            'elo_difference': elo_difference,
            'overall_elo_difference': overall_elo_diff,
            'home_away_elo_difference': home_elo_diff,
            
            # Individual team ELO features
            'home_overall_elo': home_elo_features['overall_elo'],
            'home_home_elo': home_elo_features['home_elo'],
            'home_recent_elo_change': home_elo_features['recent_elo_change'],
            'home_form_trend': home_elo_features['recent_form_trend'],
            'home_recent_win_rate': home_elo_features['recent_win_rate'],
            
            'away_overall_elo': away_elo_features['overall_elo'],
            'away_away_elo': away_elo_features['away_elo'],
            'away_recent_elo_change': away_elo_features['recent_elo_change'],
            'away_form_trend': away_elo_features['recent_form_trend'],
            'away_recent_win_rate': away_elo_features['recent_win_rate'],
            
            # Probability features
            'home_win_probability': win_probs['home_win'],
            'draw_probability': win_probs['draw'],
            'away_win_probability': win_probs['away_win'],
            
            # Competition and context features
            'competition_weight': self.elo_system.competition_weights.get(competition.lower(), 32),
            'is_high_importance': 1 if competition.lower() in ['world_cup', 'euros', 'champions_league'] else 0,
            
            # Form comparison features
            'form_difference': home_elo_features['recent_form_trend'] - away_elo_features['recent_form_trend'],
            'win_rate_difference': home_elo_features['recent_win_rate'] - away_elo_features['recent_win_rate'],
            'elo_change_difference': home_elo_features['recent_elo_change'] - away_elo_features['recent_elo_change'],
            
            # Home advantage
            'home_advantage_factor': 1,
            'venue_advantage': home_elo_features['home_elo'] - away_elo_features['away_elo']
        }
        
        # Add competition-specific ELO if available
        comp_key = f'{competition.lower()}_elo'
        if comp_key in home_elo_features and comp_key in away_elo_features:
            features[f'{competition}_elo_difference'] = (
                home_elo_features[comp_key] - away_elo_features[comp_key]
            )
            features[f'home_{comp_key}'] = home_elo_features[comp_key]
            features[f'away_{comp_key}'] = away_elo_features[comp_key]
        
        return features
    
    def predict_match_enhanced(self, home_team, away_team, competition='premier_league', match_date=None):
        """
        Enhanced match prediction using tennis-inspired model
        """
        if self.model is None:
            if not self.load_enhanced_model():
                return None
        
        if match_date is None:
            match_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f" Enhanced Prediction: {home_team} vs {away_team}")
        print(f"📅 Date: {match_date}")
        print(f" Competition: {competition.upper()}")
        
        # Create features
        match_features = self.create_prediction_features(home_team, away_team, competition, match_date)
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            if col in match_features:
                feature_vector.append(match_features[col])
            else:
                feature_vector.append(0)  # Default for missing features
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction_numeric = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Convert numeric prediction to label
        result_map = {0: 'H', 1: 'D', 2: 'A'}
        prediction = result_map[prediction_numeric]
        
        # Create probability dictionary
        prob_dict = {
            'H': probabilities[0],
            'D': probabilities[1], 
            'A': probabilities[2]
        }
        
        # Calculate confidence (like tennis model approach)
        max_prob = max(probabilities)
        confidence_level = "Very High" if max_prob > 0.8 else "High" if max_prob > 0.7 else "Medium" if max_prob > 0.6 else "Low"
        
        # Get ELO-based insights
        elo_difference = match_features['elo_difference']
        elo_strength = "Strongly Favored" if abs(elo_difference) > 200 else "Favored" if abs(elo_difference) > 100 else "Even Match"
        
        results = {
            'prediction': prediction,
            'confidence': max_prob,
            'confidence_level': confidence_level,
            'probabilities': prob_dict,
            'home_team': home_team,
            'away_team': away_team,
            'competition': competition,
            'match_date': match_date,
            'elo_difference': elo_difference,
            'elo_assessment': elo_strength,
            'model_type': 'Tennis-Inspired XGBoost'
        }
        
        self._print_enhanced_prediction(results, match_features)
        return results
    
    def _print_enhanced_prediction(self, results, features):
        """Print detailed prediction results (tennis-inspired format)"""
        print(f"\n{'='*70}")
        print(f" TENNIS-INSPIRED FOOTBALL PREDICTION")
        print(f"{'='*70}")
        
        print(f"Match: {results['home_team']} vs {results['away_team']}")
        print(f"Competition: {results['competition'].upper()}")
        print(f"Date: {results['match_date']}")
        
        print(f"\n PREDICTION RESULTS:")
        prediction = results['prediction']
        if prediction == 'H':
            outcome = f" {results['home_team']} to WIN"
        elif prediction == 'A':
            outcome = f"  {results['away_team']} to WIN"
        else:
            outcome = " DRAW"
        
        print(f"   Result: {outcome}")
        print(f"   Confidence: {results['confidence']:.1%} ({results['confidence_level']})")
        
        print(f"\n WIN PROBABILITIES:")
        prob = results['probabilities']
        print(f"    {results['home_team']:<20} {prob['H']:.1%}")
        print(f"    Draw{'':<16} {prob['D']:.1%}")
        print(f"     {results['away_team']:<20} {prob['A']:.1%}")
        
        print(f"\n ELO ANALYSIS (Key Feature from Tennis Model):")
        print(f"   ELO Difference: {features['elo_difference']:+.0f} points")
        print(f"   Assessment: {results['elo_assessment']}")
        print(f"   {results['home_team']} ELO: {features['home_overall_elo']:.0f}")
        print(f"   {results['away_team']} ELO: {features['away_overall_elo']:.0f}")
        
        print(f"\n RECENT FORM:")
        print(f"    {results['home_team']} Win Rate: {features['home_recent_win_rate']:.1%}")
        print(f"     {results['away_team']} Win Rate: {features['away_recent_win_rate']:.1%}")
        print(f"   Form Difference: {features['form_difference']:+.3f}")
        
        # Confidence assessment (tennis-inspired)
        if results['confidence'] >= 0.85:
            confidence_msg = " VERY HIGH CONFIDENCE (Tennis-level accuracy expected)"
        elif results['confidence'] >= 0.75:
            confidence_msg = " HIGH CONFIDENCE (Strong prediction)"
        elif results['confidence'] >= 0.65:
            confidence_msg = " MEDIUM CONFIDENCE (Decent prediction)"
        else:
            confidence_msg = "  LOW CONFIDENCE (Uncertain match)"
        
        print(f"\n{confidence_msg}")
        
        # Match importance
        importance = features['competition_weight']
        if importance >= 50:
            print(f" HIGH IMPORTANCE MATCH (Weight: {importance})")
        elif importance >= 35:
            print(f" IMPORTANT MATCH (Weight: {importance})")
        else:
            print(f" REGULAR MATCH (Weight: {importance})")
    
    def predict_multiple_matches_enhanced(self, matches):
        """Predict multiple matches with enhanced model"""
        predictions = []
        
        print(f" Predicting {len(matches)} matches with tennis-inspired model...")
        
        for i, match in enumerate(matches, 1):
            print(f"\n--- Match {i}/{len(matches)} ---")
            
            result = self.predict_match_enhanced(
                home_team=match.get('home_team'),
                away_team=match.get('away_team'),
                competition=match.get('competition', 'premier_league'),
                match_date=match.get('date')
            )
            
            if result:
                predictions.append(result)
        
        return predictions
    
    def analyze_prediction_confidence(self, predictions):
        """Analyze prediction confidence distribution (tennis-inspired analysis)"""
        if not predictions:
            return
        
        print(f"\n PREDICTION CONFIDENCE ANALYSIS (Tennis-Inspired)")
        print(f"{'='*60}")
        
        confidences = [p['confidence'] for p in predictions]
        
        print(f" Confidence Statistics:")
        print(f"   Average Confidence: {np.mean(confidences):.1%}")
        print(f"   Highest Confidence: {max(confidences):.1%}")
        print(f"   Lowest Confidence: {min(confidences):.1%}")
        
        # Confidence categories (like tennis model)
        very_high = sum(1 for c in confidences if c >= 0.85)
        high = sum(1 for c in confidences if 0.75 <= c < 0.85)
        medium = sum(1 for c in confidences if 0.65 <= c < 0.75)
        low = sum(1 for c in confidences if c < 0.65)
        
        total = len(predictions)
        
        print(f"\n Confidence Distribution:")
        print(f"    Very High (85%+): {very_high:2d} ({very_high/total*100:.0f}%)")
        print(f"    High (75-84%):    {high:2d} ({high/total*100:.0f}%)")
        print(f"    Medium (65-74%):  {medium:2d} ({medium/total*100:.0f}%)")
        print(f"     Low (<65%):       {low:2d} ({low/total*100:.0f}%)")
        
        # Tennis-level predictions
        tennis_level = very_high + high
        print(f"\n Tennis-Level Predictions (75%+ confidence): {tennis_level}/{total} ({tennis_level/total*100:.0f}%)")
    
    def save_predictions(self, predictions, filename=None):
        """Save predictions to file"""
        if filename is None:
            filename = f"enhanced_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = f"../results/{filename}"
        
        import os
        os.makedirs('../results', exist_ok=True)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f" Predictions saved to {filepath}")

def main():
    predictor = EnhancedMatchPredictor()
    
    print(" Enhanced Football Prediction System")
    print(" Tennis-Inspired AI (Targeting 85% Accuracy)")
    
    while True:
        print(f"\n{'='*50}")
        print("COMPETITION PREDICTION OPTIONS:")
        print("1. Single Match Prediction")
        print("2. Multiple Match Predictions") 
        print("3. World Cup Tournament Simulation")
        print("4. Premier League Predictions")
        print("5. Champions League Predictions")
        print("6. European Championship (Euros)")
        print("7. Copa America")
        print("8. Custom Tournament")
        print("9. Exit")
        
        choice = input("\nSelect competition (1-9): ").strip()
        
        if choice == '1':
            home_team = input("Home team: ").strip()
            away_team = input("Away team: ").strip()
            competition = input("Competition (default: premier_league): ").strip() or 'premier_league'
            
            predictor.predict_match_enhanced(home_team, away_team, competition)
            
        elif choice == '2':
            # Example multiple matches
            matches = [
                {'home_team': 'Manchester City', 'away_team': 'Arsenal', 'competition': 'premier_league'},
                {'home_team': 'Liverpool', 'away_team': 'Chelsea', 'competition': 'premier_league'},
                {'home_team': 'Real Madrid', 'away_team': 'Barcelona', 'competition': 'la_liga'},
            ]
            
            predictions = predictor.predict_multiple_matches_enhanced(matches)
            predictor.analyze_prediction_confidence(predictions)
            
        elif choice == '3':
            print(" World Cup simulation with tennis-inspired model...")
            from enhanced_world_cup_predictor import EnhancedWorldCupPredictor
            wc_predictor = EnhancedWorldCupPredictor()
            wc_predictor.predict_full_tournament()
            
        elif choice == '4':
            print(" Premier League predictions...")
            matches = [
                {'home_team': 'Manchester City', 'away_team': 'Arsenal', 'competition': 'premier_league'},
                {'home_team': 'Liverpool', 'away_team': 'Chelsea', 'competition': 'premier_league'},
                {'home_team': 'Manchester United', 'away_team': 'Tottenham', 'competition': 'premier_league'},
                {'home_team': 'Newcastle', 'away_team': 'Brighton', 'competition': 'premier_league'},
            ]
            predictions = predictor.predict_multiple_matches_enhanced(matches)
            predictor.analyze_prediction_confidence(predictions)
            
        elif choice == '5':
            print(" Champions League predictions...")
            matches = [
                {'home_team': 'Real Madrid', 'away_team': 'Manchester City', 'competition': 'champions_league'},
                {'home_team': 'Barcelona', 'away_team': 'PSG', 'competition': 'champions_league'},
                {'home_team': 'Bayern Munich', 'away_team': 'Arsenal', 'competition': 'champions_league'},
                {'home_team': 'Liverpool', 'away_team': 'Juventus', 'competition': 'champions_league'},
            ]
            predictions = predictor.predict_multiple_matches_enhanced(matches)
            predictor.analyze_prediction_confidence(predictions)
            
        elif choice == '6':
            print(" European Championship (Euros) predictions...")
            matches = [
                {'home_team': 'France', 'away_team': 'Germany', 'competition': 'euros'},
                {'home_team': 'Spain', 'away_team': 'Italy', 'competition': 'euros'},
                {'home_team': 'England', 'away_team': 'Netherlands', 'competition': 'euros'},
                {'home_team': 'Portugal', 'away_team': 'Belgium', 'competition': 'euros'},
            ]
            predictions = predictor.predict_multiple_matches_enhanced(matches)
            predictor.analyze_prediction_confidence(predictions)
            
        elif choice == '7':
            print(" Copa America predictions...")
            matches = [
                {'home_team': 'Brazil', 'away_team': 'Argentina', 'competition': 'copa_america'},
                {'home_team': 'Uruguay', 'away_team': 'Colombia', 'competition': 'copa_america'},
                {'home_team': 'Chile', 'away_team': 'Peru', 'competition': 'copa_america'},
                {'home_team': 'Ecuador', 'away_team': 'Venezuela', 'competition': 'copa_america'},
            ]
            predictions = predictor.predict_multiple_matches_enhanced(matches)
            predictor.analyze_prediction_confidence(predictions)
            
        elif choice == '8':
            print("  Custom Tournament...")
            print("Enter your own matches:")
            custom_matches = []
            
            while True:
                home = input("Home team (or 'done' to finish): ").strip()
                if home.lower() == 'done':
                    break
                away = input("Away team: ").strip()
                comp = input("Competition (default: friendly): ").strip() or 'friendly'
                
                custom_matches.append({
                    'home_team': home,
                    'away_team': away, 
                    'competition': comp
                })
            
            if custom_matches:
                predictions = predictor.predict_multiple_matches_enhanced(custom_matches)
                predictor.analyze_prediction_confidence(predictions)
            else:
                print("No matches entered.")
            
        elif choice == '9':
            print(" Thanks for using the tennis-inspired football predictor!")
            break
        
        else:
            print(" Invalid choice. Please try again.")

if __name__ == "__main__":
    main()