import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

class FootballEloSystem:
    """
    Football ELO Rating System inspired by tennis predictions achieving 85% accuracy.
    
    Key features:
    - Overall ELO rating for each team
    - Competition-specific ELO (Premier League, Champions League, etc.)
    - Home/Away specific ELO ratings
    - Goal difference consideration
    - K-factor adjustment based on match importance
    """
    
    def __init__(self):
        self.default_elo = 1500
        self.k_factor_base = 32
        self.home_advantage = 100  # ELO points advantage for home team
        
        # Competition importance multipliers (like surface importance in tennis)
        self.competition_weights = {
            'world_cup': 60,
            'euros': 50, 
            'champions_league': 40,
            'premier_league': 32,
            'la_liga': 32,
            'bundesliga': 32,
            'serie_a': 32,
            'ligue_1': 30,
            'fa_cup': 25,
            'friendly': 15
        }
        
        # Team ELO ratings storage
        self.team_elo = {}  # Overall ELO
        self.competition_elo = {}  # Competition-specific ELO
        self.home_elo = {}  # Home performance ELO
        self.away_elo = {}  # Away performance ELO
        self.elo_history = {}  # ELO progression over time
        
    def initialize_team(self, team_name):
        """Initialize ELO ratings for a new team"""
        if team_name not in self.team_elo:
            self.team_elo[team_name] = self.default_elo
            self.competition_elo[team_name] = {}
            self.home_elo[team_name] = self.default_elo
            self.away_elo[team_name] = self.default_elo
            self.elo_history[team_name] = []
    
    def get_k_factor(self, competition='premier_league', goal_difference=1):
        """
        Calculate K-factor based on match importance and result margin
        Similar to tennis where different tournaments have different importance
        """
        base_k = self.competition_weights.get(competition.lower(), 32)
        
        # Adjust K-factor based on goal difference (bigger wins = more rating change)
        if goal_difference >= 3:
            k_multiplier = 1.5
        elif goal_difference >= 2:
            k_multiplier = 1.2
        else:
            k_multiplier = 1.0
            
        return base_k * k_multiplier
    
    def expected_score(self, team_a_elo, team_b_elo, home_advantage=0):
        """
        Calculate expected score (win probability) for team A
        """
        rating_diff = team_a_elo - team_b_elo + home_advantage
        expected = 1 / (1 + 10 ** (-rating_diff / 400))
        return expected
    
    def update_elo_ratings(self, home_team, away_team, home_score, away_score, 
                          competition='premier_league', match_date=None):
        """
        Update ELO ratings after a match - inspired by tennis ELO system
        """
        # Initialize teams if needed
        self.initialize_team(home_team)
        self.initialize_team(away_team)
        
        if match_date is None:
            match_date = datetime.now()
        
        # Get current ELO ratings
        home_elo = self.team_elo[home_team]
        away_elo = self.team_elo[away_team]
        
        # Calculate match result (1 = home win, 0.5 = draw, 0 = away win)
        if home_score > away_score:
            actual_score = 1.0
        elif home_score < away_score:
            actual_score = 0.0
        else:
            actual_score = 0.5
        
        # Calculate expected scores
        home_expected = self.expected_score(home_elo, away_elo, self.home_advantage)
        away_expected = 1 - home_expected
        
        # Get K-factor
        goal_difference = abs(home_score - away_score)
        k_factor = self.get_k_factor(competition, goal_difference)
        
        # Update overall ELO
        home_elo_change = k_factor * (actual_score - home_expected)
        away_elo_change = k_factor * ((1 - actual_score) - away_expected)
        
        self.team_elo[home_team] += home_elo_change
        self.team_elo[away_team] += away_elo_change
        
        # Update competition-specific ELO
        comp = competition.lower()
        if comp not in self.competition_elo[home_team]:
            self.competition_elo[home_team][comp] = self.default_elo
        if comp not in self.competition_elo[away_team]:
            self.competition_elo[away_team][comp] = self.default_elo
            
        self.competition_elo[home_team][comp] += home_elo_change
        self.competition_elo[away_team][comp] += away_elo_change
        
        # Update home/away specific ELO
        self.home_elo[home_team] += home_elo_change * 1.2  # Home performance matters more for home ELO
        self.away_elo[away_team] += away_elo_change * 1.2  # Away performance matters more for away ELO
        
        # Store ELO history
        self.elo_history[home_team].append({
            'date': match_date,
            'elo': self.team_elo[home_team],
            'change': home_elo_change,
            'opponent': away_team,
            'result': actual_score,
            'competition': competition
        })
        
        self.elo_history[away_team].append({
            'date': match_date,
            'elo': self.team_elo[away_team], 
            'change': away_elo_change,
            'opponent': home_team,
            'result': 1 - actual_score,
            'competition': competition
        })
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_elo_change': home_elo_change,
            'away_elo_change': away_elo_change,
            'home_new_elo': self.team_elo[home_team],
            'away_new_elo': self.team_elo[away_team]
        }
    
    def get_elo_difference(self, home_team, away_team, competition=None, venue='neutral'):
        """
        Get ELO difference between two teams (key feature from tennis model)
        """
        self.initialize_team(home_team)
        self.initialize_team(away_team)
        
        if competition and competition.lower() in self.competition_elo[home_team]:
            home_elo = self.competition_elo[home_team][competition.lower()]
            away_elo = self.competition_elo[away_team].get(competition.lower(), self.default_elo)
        else:
            home_elo = self.team_elo[home_team]
            away_elo = self.team_elo[away_team]
        
        # Adjust for venue
        if venue == 'home':
            home_elo += self.home_advantage
        elif venue == 'away':
            away_elo += self.home_advantage
            
        return home_elo - away_elo
    
    def get_win_probability(self, home_team, away_team, competition=None, venue='home'):
        """
        Get win probability based on ELO (crucial for prediction confidence)
        """
        elo_diff = self.get_elo_difference(home_team, away_team, competition, venue)
        
        # Convert ELO difference to probabilities
        home_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        draw_prob = 0.25  # Approximate draw probability in football
        away_prob = 1 - home_prob
        
        # Adjust for draws
        home_prob *= (1 - draw_prob)
        away_prob *= (1 - draw_prob)
        
        return {
            'home_win': home_prob,
            'draw': draw_prob, 
            'away_win': away_prob
        }
    
    def get_team_elo_features(self, team_name, competition=None):
        """
        Get comprehensive ELO features for a team (inspired by tennis features)
        """
        self.initialize_team(team_name)
        
        features = {
            'overall_elo': self.team_elo[team_name],
            'home_elo': self.home_elo[team_name],
            'away_elo': self.away_elo[team_name],
        }
        
        # Competition-specific ELO
        if competition:
            comp = competition.lower()
            features[f'{comp}_elo'] = self.competition_elo[team_name].get(comp, self.default_elo)
        
        # Recent form (ELO changes in last 5 matches)
        recent_matches = sorted(self.elo_history[team_name], key=lambda x: x['date'], reverse=True)[:5]
        if recent_matches:
            features['recent_elo_change'] = sum([match['change'] for match in recent_matches])
            features['recent_form_trend'] = np.mean([match['change'] for match in recent_matches])
            features['recent_win_rate'] = np.mean([match['result'] for match in recent_matches])
        else:
            features['recent_elo_change'] = 0
            features['recent_form_trend'] = 0
            features['recent_win_rate'] = 0.5
            
        return features
    
    def build_from_match_data(self, match_data):
        """
        Build ELO system from historical match data
        """
        print("Building ELO ratings from historical matches...")
        
        # Sort matches by date
        matches = match_data.copy()
        matches['date'] = pd.to_datetime(matches['date'])
        matches = matches.sort_values('date')
        
        updates = []
        for _, match in matches.iterrows():
            competition = match.get('competition', 'premier_league')
            
            update = self.update_elo_ratings(
                home_team=match['home_team'],
                away_team=match['away_team'],
                home_score=match['home_score'],
                away_score=match['away_score'],
                competition=competition,
                match_date=match['date']
            )
            updates.append(update)
        
        print(f"Processed {len(updates)} matches")
        print(f"ELO ratings calculated for {len(self.team_elo)} teams")
        return updates
    
    def get_top_teams(self, n=20, competition=None):
        """Get top teams by ELO rating"""
        if competition:
            # Get competition-specific rankings
            comp_ratings = {}
            for team, comps in self.competition_elo.items():
                if competition.lower() in comps:
                    comp_ratings[team] = comps[competition.lower()]
            ratings = comp_ratings
        else:
            ratings = self.team_elo
            
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_teams[:n]
    
    def plot_elo_progression(self, teams, save_path=None):
        """Plot ELO progression for specified teams (like tennis ELO plots)"""
        plt.figure(figsize=(15, 10))
        
        for team in teams:
            if team in self.elo_history:
                history = sorted(self.elo_history[team], key=lambda x: x['date'])
                dates = [h['date'] for h in history]
                elos = [h['elo'] for h in history]
                
                plt.plot(dates, elos, label=team, linewidth=2, marker='o', markersize=1)
        
        plt.title('Team ELO Progression Over Time', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('ELO Rating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_elo_data(self, filepath):
        """Save ELO data to file"""
        data = {
            'team_elo': self.team_elo,
            'competition_elo': self.competition_elo,
            'home_elo': self.home_elo,
            'away_elo': self.away_elo,
            'elo_history': {team: [
                {**h, 'date': h['date'].isoformat() if isinstance(h['date'], datetime) else str(h['date'])}
                for h in history
            ] for team, history in self.elo_history.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"ELO data saved to {filepath}")
    
    def load_elo_data(self, filepath):
        """Load ELO data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.team_elo = data['team_elo']
        self.competition_elo = data['competition_elo']
        self.home_elo = data['home_elo']
        self.away_elo = data['away_elo']
        
        # Convert date strings back to datetime objects
        self.elo_history = {}
        for team, history in data['elo_history'].items():
            self.elo_history[team] = [
                {**h, 'date': pd.to_datetime(h['date'])}
                for h in history
            ]
        print(f"ELO data loaded from {filepath}")

def main():
    # Example usage
    elo_system = FootballEloSystem()
    
    # Example matches
    example_matches = [
        {'home_team': 'Manchester City', 'away_team': 'Arsenal', 'home_score': 3, 'away_score': 1, 'date': '2024-01-15', 'competition': 'premier_league'},
        {'home_team': 'Liverpool', 'away_team': 'Chelsea', 'home_score': 2, 'away_score': 0, 'date': '2024-01-16', 'competition': 'premier_league'},
        {'home_team': 'Real Madrid', 'away_team': 'Barcelona', 'home_score': 2, 'away_score': 1, 'date': '2024-01-20', 'competition': 'la_liga'},
    ]
    
    # Build ELO from matches
    match_df = pd.DataFrame(example_matches)
    elo_system.build_from_match_data(match_df)
    
    # Show top teams
    print("\nTop Teams by ELO:")
    top_teams = elo_system.get_top_teams(10)
    for i, (team, elo) in enumerate(top_teams, 1):
        print(f"{i:2d}. {team:<20} {elo:4.0f}")
    
    # Get match prediction
    probs = elo_system.get_win_probability('Manchester City', 'Liverpool', venue='home')
    print(f"\nManchester City vs Liverpool (at home):")
    print(f"Home win: {probs['home_win']:.1%}")
    print(f"Draw:     {probs['draw']:.1%}")  
    print(f"Away win: {probs['away_win']:.1%}")

if __name__ == "__main__":
    main()