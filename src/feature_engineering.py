import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        self.data_dir = '../data'
    
    def calculate_form_features(self, match_results, team_name, date, lookback_games=5):
        """Calculate team form features based on recent matches"""
        # Filter matches for the team before the given date
        team_matches = match_results[
            ((match_results['home_team'] == team_name) | 
             (match_results['away_team'] == team_name)) &
            (pd.to_datetime(match_results['date']) < pd.to_datetime(date))
        ].sort_values('date', ascending=False).head(lookback_games)
        
        if len(team_matches) == 0:
            return {
                'recent_form_points': 0,
                'recent_goals_for': 0,
                'recent_goals_against': 0,
                'recent_wins': 0,
                'recent_draws': 0,
                'recent_losses': 0
            }
        
        points = 0
        goals_for = 0
        goals_against = 0
        wins = 0
        draws = 0
        losses = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team_name:
                goals_for += match['home_score']
                goals_against += match['away_score']
                if match['result'] == 'H':
                    points += 3
                    wins += 1
                elif match['result'] == 'D':
                    points += 1
                    draws += 1
                else:
                    losses += 1
            else:
                goals_for += match['away_score']
                goals_against += match['home_score']
                if match['result'] == 'A':
                    points += 3
                    wins += 1
                elif match['result'] == 'D':
                    points += 1
                    draws += 1
                else:
                    losses += 1
        
        return {
            'recent_form_points': points,
            'recent_goals_for': goals_for,
            'recent_goals_against': goals_against,
            'recent_wins': wins,
            'recent_draws': draws,
            'recent_losses': losses,
            'recent_goal_difference': goals_for - goals_against
        }
    
    def calculate_head_to_head(self, match_results, home_team, away_team, date, lookback_matches=10):
        """Calculate head-to-head statistics between two teams"""
        h2h_matches = match_results[
            (((match_results['home_team'] == home_team) & (match_results['away_team'] == away_team)) |
             ((match_results['home_team'] == away_team) & (match_results['away_team'] == home_team))) &
            (pd.to_datetime(match_results['date']) < pd.to_datetime(date))
        ].sort_values('date', ascending=False).head(lookback_matches)
        
        if len(h2h_matches) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_draws': 0,
                'h2h_home_goals_avg': 0,
                'h2h_away_goals_avg': 0
            }
        
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_total = 0
        away_goals_total = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                home_goals_total += match['home_score']
                away_goals_total += match['away_score']
                if match['result'] == 'H':
                    home_wins += 1
                elif match['result'] == 'A':
                    away_wins += 1
                else:
                    draws += 1
            else:
                home_goals_total += match['away_score']
                away_goals_total += match['home_score']
                if match['result'] == 'A':
                    home_wins += 1
                elif match['result'] == 'H':
                    away_wins += 1
                else:
                    draws += 1
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_home_goals_avg': home_goals_total / len(h2h_matches),
            'h2h_away_goals_avg': away_goals_total / len(h2h_matches),
            'h2h_matches_played': len(h2h_matches)
        }
    
    def get_team_season_stats(self, team_stats, team_name, season):
        """Get team's season statistics"""
        team_data = team_stats[
            (team_stats['team_name'] == team_name) & 
            (team_stats['season'] == season)
        ]
        
        if len(team_data) == 0:
            return {
                'points_per_game': 0,
                'goals_for_per_game': 0,
                'goals_against_per_game': 0,
                'goal_difference_per_game': 0,
                'win_percentage': 0,
                'avg_possession': 50,
                'shots_per_game': 10,
                'pass_accuracy': 80
            }
        
        team_data = team_data.iloc[0]
        matches_played = max(team_data['matches_played'], 1)
        
        return {
            'points_per_game': team_data['points'] / matches_played,
            'goals_for_per_game': team_data['goals_for'] / matches_played,
            'goals_against_per_game': team_data['goals_against'] / matches_played,
            'goal_difference_per_game': team_data['goal_difference'] / matches_played,
            'win_percentage': team_data['wins'] / matches_played * 100,
            'avg_possession': team_data['avg_possession'],
            'shots_per_game': team_data['shots_per_game'],
            'pass_accuracy': team_data['pass_accuracy']
        }
    
    def create_match_features(self, match_results, team_stats, upcoming_matches=None):
        """Create comprehensive features for match prediction"""
        
        if upcoming_matches is None:
            # Create features for historical matches
            features = []
            
            for _, match in match_results.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                match_date = match['date']
                season = '2023-24'  # You might want to derive this from the date
                
                # Home team features
                home_form = self.calculate_form_features(match_results, home_team, match_date)
                home_season = self.get_team_season_stats(team_stats, home_team, season)
                
                # Away team features
                away_form = self.calculate_form_features(match_results, away_team, match_date)
                away_season = self.get_team_season_stats(team_stats, away_team, season)
                
                # Head-to-head features
                h2h = self.calculate_head_to_head(match_results, home_team, away_team, match_date)
                
                # Combined features
                match_features = {
                    'date': match_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'target_result': match['result'],
                    'target_home_score': match['home_score'],
                    'target_away_score': match['away_score'],
                    
                    # Home team features
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
                    
                    # Away team features
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
                    
                    # Head-to-head features
                    'h2h_home_wins': h2h['h2h_home_wins'],
                    'h2h_away_wins': h2h['h2h_away_wins'],
                    'h2h_draws': h2h['h2h_draws'],
                    'h2h_home_goals_avg': h2h['h2h_home_goals_avg'],
                    'h2h_away_goals_avg': h2h['h2h_away_goals_avg'],
                    
                    # Derived features
                    'form_difference': home_form['recent_form_points'] - away_form['recent_form_points'],
                    'points_per_game_difference': home_season['points_per_game'] - away_season['points_per_game'],
                    'goal_difference_difference': home_season['goal_difference_per_game'] - away_season['goal_difference_per_game'],
                    'possession_difference': home_season['avg_possession'] - away_season['avg_possession'],
                    'home_advantage': 1  # Home advantage factor
                }
                
                features.append(match_features)
            
            return pd.DataFrame(features)
        
        else:
            # Create features for upcoming matches
            # Similar logic but without target variables
            pass
    
    def process_all_features(self):
        """Process all data and create feature sets"""
        print("Loading data...")
        match_results = pd.read_csv(f'{self.data_dir}/all_match_results.csv')
        team_stats = pd.read_csv(f'{self.data_dir}/all_team_stats.csv')
        
        print("Creating features...")
        features_df = self.create_match_features(match_results, team_stats)
        
        print("Saving features...")
        features_df.to_csv(f'{self.data_dir}/match_features.csv', index=False)
        
        print(f"Created {len(features_df)} feature sets")
        print(f"Features saved to {self.data_dir}/match_features.csv")
        
        return features_df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.process_all_features()