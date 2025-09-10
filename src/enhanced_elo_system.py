import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from elo_system import FootballEloSystem
from player_elo_system import PlayerEloSystem

class EnhancedEloSystem(FootballEloSystem):
    """
    Enhanced ELO System combining Team + Player ELO
    
    Tennis Inspiration:
    - Team ELO = Base team strength (like tennis doubles team)
    - Player ELO = Individual player strength (like tennis singles)
    - Combined rating = More accurate prediction power
    
    This system targets the 85% accuracy by adding granular player-level data
    to the team-level ELO calculations.
    """
    
    def __init__(self):
        super().__init__()
        self.player_system = PlayerEloSystem()
        self.lineup_history = {}  # Track starting XIs
        self.key_player_impacts = {}  # Track impact of star players
        
        # Enhanced weights for combined system
        self.team_elo_weight = 0.65      # Team chemistry/system (65%)
        self.player_elo_weight = 0.25    # Starting XI quality (25%)
        self.situational_weight = 0.10   # Injuries/suspensions/form (10%)
        
    def get_combined_team_strength(self, team_name, starting_xi=None, key_player_statuses=None, 
                                 competition='premier_league', venue='home', match_date=None):
        """
        Calculate combined team strength using both team and player ELO
        
        This is the core enhancement - combining multiple ELO sources like
        tennis combines different ranking systems for more accuracy
        """
        if match_date is None:
            match_date = datetime.now()
        
        # 1. Get base team ELO (existing system)
        base_team_elo = self.team_elo.get(team_name, self.default_elo)
        
        # Add competition-specific adjustment
        comp_elo = self.competition_elo.get(team_name, {}).get(competition, base_team_elo)
        
        # Add home/away adjustment  
        if venue == 'home':
            venue_elo = self.home_elo.get(team_name, base_team_elo)
        else:
            venue_elo = self.away_elo.get(team_name, base_team_elo)
        
        # Weight the different team ELO components
        weighted_team_elo = (base_team_elo * 0.5 + comp_elo * 0.3 + venue_elo * 0.2)
        
        # 2. Get player-based adjustments (if starting XI available)
        player_adjustment = 0
        player_analysis = None
        
        if starting_xi:
            player_analysis = self.player_system.get_enhanced_team_elo(
                weighted_team_elo, starting_xi, key_player_statuses
            )
            player_adjustment = player_analysis['starting_xi_adjustment'] + player_analysis['availability_adjustment']
        
        # 3. Calculate situational adjustments
        situational_adjustment = 0
        
        # Recent form adjustment (last 5 matches)
        recent_form = self.get_recent_team_form(team_name, matches=5)
        form_adjustment = (recent_form - weighted_team_elo) * 0.1
        situational_adjustment += form_adjustment
        
        # Home advantage for venue
        if venue == 'home':
            situational_adjustment += self.home_advantage * 0.8  # Reduced since already in venue_elo
        
        # 4. Combine all components
        final_strength = (
            weighted_team_elo * self.team_elo_weight +
            player_adjustment * self.player_elo_weight +
            situational_adjustment * self.situational_weight
        )
        
        return {
            'final_strength': final_strength,
            'base_team_elo': base_team_elo,
            'weighted_team_elo': weighted_team_elo,
            'player_adjustment': player_adjustment,
            'situational_adjustment': situational_adjustment,
            'player_analysis': player_analysis,
            'components': {
                'team_contribution': weighted_team_elo * self.team_elo_weight,
                'player_contribution': player_adjustment * self.player_elo_weight,
                'situational_contribution': situational_adjustment * self.situational_weight
            }
        }
    
    def get_recent_team_form(self, team_name, matches=5):
        """Calculate recent team form from ELO history"""
        if team_name not in self.elo_history or not self.elo_history[team_name]:
            return self.team_elo.get(team_name, self.default_elo)
        
        recent_matches = sorted(self.elo_history[team_name], 
                              key=lambda x: x['date'], reverse=True)[:matches]
        
        if not recent_matches:
            return self.team_elo.get(team_name, self.default_elo)
        
        # Weight recent matches more heavily
        weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(recent_matches)]
        weighted_elo = sum(match['elo'] * weight for match, weight in zip(recent_matches, weights))
        
        return weighted_elo / sum(weights)
    
    def predict_match_enhanced(self, home_team, away_team, home_xi=None, away_xi=None,
                             home_injuries=None, away_injuries=None, competition='premier_league',
                             match_date=None):
        """
        Enhanced match prediction using combined team + player ELO system
        
        This is where we target the 85% accuracy by using all available information
        """
        if match_date is None:
            match_date = datetime.now()
        
        print(f"\nEnhanced Match Prediction: {home_team} vs {away_team}")
        print(f"Competition: {competition.upper()}")
        print(f"Date: {match_date.strftime('%Y-%m-%d') if isinstance(match_date, datetime) else match_date}")
        print("-" * 60)
        
        # Calculate enhanced team strengths
        home_strength = self.get_combined_team_strength(
            home_team, home_xi, home_injuries, competition, 'home', match_date
        )
        
        away_strength = self.get_combined_team_strength(
            away_team, away_xi, away_injuries, competition, 'away', match_date
        )
        
        # Calculate match probabilities
        strength_diff = home_strength['final_strength'] - away_strength['final_strength']
        home_win_prob = 1 / (1 + 10 ** (-strength_diff / 400))
        
        # Adjust for draws (football-specific)
        draw_prob = 0.27  # Base draw probability in football
        home_win_prob *= (1 - draw_prob)
        away_win_prob = (1 - home_win_prob - draw_prob)
        
        # Determine prediction
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            prediction = 'H'
            confidence = home_win_prob
        elif away_win_prob > draw_prob:
            prediction = 'A' 
            confidence = away_win_prob
        else:
            prediction = 'D'
            confidence = draw_prob
        
        # Create detailed results
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'H': home_win_prob,
                'D': draw_prob,
                'A': away_win_prob
            },
            'home_team': home_team,
            'away_team': away_team,
            'home_analysis': home_strength,
            'away_analysis': away_strength,
            'strength_difference': strength_diff,
            'competition': competition,
            'match_date': match_date,
            'enhanced_features_used': {
                'starting_xi': home_xi is not None or away_xi is not None,
                'injury_data': home_injuries is not None or away_injuries is not None,
                'player_elo': True,
                'team_elo': True,
                'form_analysis': True
            }
        }
        
        # Print detailed analysis
        self._print_enhanced_analysis(result)
        
        return result
    
    def _print_enhanced_analysis(self, result):
        """Print detailed match analysis"""
        print(f"ENHANCED PREDICTION RESULTS")
        print("=" * 60)
        
        # Basic prediction
        pred_text = {
            'H': f"{result['home_team']} WIN",
            'D': "DRAW", 
            'A': f"{result['away_team']} WIN"
        }
        
        print(f"Prediction: {pred_text[result['prediction']]}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        # Probabilities
        print(f"\nMatch Probabilities:")
        print(f"  {result['home_team']} Win: {result['probabilities']['H']:.1%}")
        print(f"  Draw:               {result['probabilities']['D']:.1%}")  
        print(f"  {result['away_team']} Win: {result['probabilities']['A']:.1%}")
        
        # Team strength breakdown
        print(f"\nTeam Strength Analysis:")
        print(f"  {result['home_team']}:")
        home = result['home_analysis']
        print(f"    Final Strength:     {home['final_strength']:.0f}")
        print(f"    Base Team ELO:      {home['base_team_elo']:.0f}")
        print(f"    Player Adjustment:  {home['player_adjustment']:+.0f}")
        print(f"    Situation Adjust:   {home['situational_adjustment']:+.0f}")
        
        print(f"  {result['away_team']}:")
        away = result['away_analysis']
        print(f"    Final Strength:     {away['final_strength']:.0f}")
        print(f"    Base Team ELO:      {away['base_team_elo']:.0f}")
        print(f"    Player Adjustment:  {away['player_adjustment']:+.0f}")
        print(f"    Situation Adjust:   {away['situational_adjustment']:+.0f}")
        
        # Enhanced features used
        features = result['enhanced_features_used']
        print(f"\nEnhanced Features Used:")
        print(f"  Starting XI Data:   {'Yes' if features['starting_xi'] else 'No'}")
        print(f"  Injury/Suspension:  {'Yes' if features['injury_data'] else 'No'}")
        print(f"  Player ELO System:  {'Yes' if features['player_elo'] else 'No'}")
        print(f"  Team ELO System:    {'Yes' if features['team_elo'] else 'No'}")
        print(f"  Form Analysis:      {'Yes' if features['form_analysis'] else 'No'}")
        
        # Player analysis details (if available)
        if home['player_analysis']:
            print(f"\nPlayer Analysis ({result['home_team']}):")
            pa = home['player_analysis']
            print(f"  Average Player ELO: {pa['xi_analysis']['avg_player_elo']:.0f}")
            print(f"  Star Players:       {len(pa['xi_analysis']['star_players'])}")
            if pa['xi_analysis']['star_players']:
                print(f"  Stars: {', '.join(pa['xi_analysis']['star_players'][:3])}")
        
        if away['player_analysis']:
            print(f"\nPlayer Analysis ({result['away_team']}):")
            pa = away['player_analysis']
            print(f"  Average Player ELO: {pa['xi_analysis']['avg_player_elo']:.0f}")
            print(f"  Star Players:       {len(pa['xi_analysis']['star_players'])}")
            if pa['xi_analysis']['star_players']:
                print(f"  Stars: {', '.join(pa['xi_analysis']['star_players'][:3])}")
        
        # Confidence assessment
        conf = result['confidence']
        if conf >= 0.70:
            conf_level = "VERY HIGH (Tennis-level reliability)"
        elif conf >= 0.60:
            conf_level = "HIGH (Strong prediction)"
        elif conf >= 0.50:
            conf_level = "MEDIUM (Reasonable confidence)"
        else:
            conf_level = "LOW (Uncertain outcome)"
        
        print(f"\nConfidence Assessment: {conf_level}")
    
    def initialize_with_sample_data(self):
        """Initialize both team and player systems with sample data"""
        print("Initializing Enhanced ELO System with sample data...")
        
        # Initialize player system
        self.player_system.create_sample_player_database()
        
        # Add some sample team ELO data
        sample_teams = {
            'Manchester City': 1580,
            'Arsenal': 1520, 
            'Liverpool': 1510,
            'Chelsea': 1480,
            'Manchester United': 1460,
            'Newcastle': 1420,
            'Tottenham': 1440,
            'Brighton': 1380,
            'Real Madrid': 1550,
            'Barcelona': 1530,
            'PSG': 1540,
            'Bayern Munich': 1560
        }
        
        for team, elo in sample_teams.items():
            self.team_elo[team] = elo
            self.home_elo[team] = elo + np.random.normal(0, 30)
            self.away_elo[team] = elo + np.random.normal(0, 30)
            self.elo_history[team] = []
        
        print(f"Initialized {len(sample_teams)} teams with ELO ratings")

def main():
    print("Enhanced ELO System - Tennis-Inspired Football Prediction")
    print("Targeting 85% Accuracy with Combined Team + Player ELO")
    print("=" * 70)
    
    # Initialize system
    enhanced_elo = EnhancedEloSystem()
    enhanced_elo.initialize_with_sample_data()
    
    # Example match with full lineups
    print("\nExample 1: Basic prediction (Team ELO only)")
    result1 = enhanced_elo.predict_match_enhanced(
        home_team='Manchester City',
        away_team='Arsenal',
        competition='premier_league'
    )
    
    print("\n" + "="*70)
    print("\nExample 2: Enhanced prediction with Starting XIs")
    
    man_city_xi = {
        'GK': 'Ederson',
        'CB': ['Ruben Dias', 'John Stones'],
        'LB': 'Josko Gvardiol', 
        'RB': 'Kyle Walker',
        'CDM': 'Rodri',
        'CM': ['Kevin De Bruyne', 'Bernardo Silva'],
        'LW': 'Jack Grealish',
        'RW': 'Riyad Mahrez', 
        'ST': 'Erling Haaland'
    }
    
    arsenal_xi = {
        'GK': 'Aaron Ramsdale',
        'CB': ['William Saliba', 'Gabriel'],
        'LB': 'Oleksandr Zinchenko',
        'RB': 'Ben White', 
        'CDM': 'Thomas Partey',
        'CM': ['Martin Odegaard', 'Granit Xhaka'],
        'LW': 'Gabriel Martinelli',
        'RW': 'Bukayo Saka',
        'ST': 'Gabriel Jesus'
    }
    
    result2 = enhanced_elo.predict_match_enhanced(
        home_team='Manchester City',
        away_team='Arsenal',
        home_xi=man_city_xi,
        away_xi=arsenal_xi,
        competition='premier_league'
    )
    
    print("\n" + "="*70)
    print("\nAccuracy Comparison:")
    print(f"Basic Team ELO Confidence:    {result1['confidence']:.1%}")
    print(f"Enhanced ELO Confidence:      {result2['confidence']:.1%}")
    print(f"Confidence Improvement:       {(result2['confidence'] - result1['confidence']):.1%}")
    print(f"\nTarget: 85% accuracy through enhanced player-level data")
    
    # Save enhanced system
    os.makedirs('../models', exist_ok=True)
    enhanced_elo.save_elo_data('../models/enhanced_elo_system.json') 
    enhanced_elo.player_system.save_player_data('../models/player_elo_system.json')

if __name__ == "__main__":
    main()