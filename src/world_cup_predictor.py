import pandas as pd
import numpy as np
from predict import MatchPredictor
from datetime import datetime, timedelta
import json

class WorldCupPredictor(MatchPredictor):
    def __init__(self):
        super().__init__()
        self.world_cup_teams = [
            'Argentina', 'Brazil', 'France', 'Spain', 'England', 'Germany',
            'Italy', 'Netherlands', 'Portugal', 'Belgium', 'Croatia', 'Morocco',
            'Poland', 'Mexico', 'Uruguay', 'Switzerland', 'Denmark', 'Japan',
            'South Korea', 'Senegal', 'Australia', 'Canada', 'USA', 'Wales',
            'Iran', 'Saudi Arabia', 'Tunisia', 'Cameroon', 'Ghana', 'Ecuador',
            'Qatar', 'Serbia'
        ]
        
    def create_group_stage_fixtures(self):
        """Create World Cup group stage fixtures"""
        # Example groups (you can update these based on actual World Cup draw)
        groups = {
            'A': ['Argentina', 'Poland', 'Mexico', 'Saudi Arabia'],
            'B': ['England', 'Iran', 'USA', 'Wales'],
            'C': ['France', 'Australia', 'Denmark', 'Tunisia'],
            'D': ['Spain', 'Germany', 'Japan', 'Cameroon'],
            'E': ['Brazil', 'Switzerland', 'Serbia', 'South Korea'],
            'F': ['Belgium', 'Croatia', 'Morocco', 'Canada'],
            'G': ['Portugal', 'Uruguay', 'Ghana', 'Ecuador'],
            'H': ['Netherlands', 'Senegal', 'Qatar', 'Italy']
        }
        
        fixtures = []
        base_date = datetime(2024, 11, 20)  # Example World Cup start date
        
        for group_name, teams in groups.items():
            # Each team plays every other team once
            group_fixtures = []
            for i, team1 in enumerate(teams):
                for j, team2 in enumerate(teams[i+1:], i+1):
                    fixture = {
                        'group': group_name,
                        'home_team': team1,
                        'away_team': team2,
                        'date': (base_date + timedelta(days=len(fixtures))).strftime('%Y-%m-%d'),
                        'stage': 'group'
                    }
                    fixtures.append(fixture)
                    group_fixtures.append(fixture)
        
        return fixtures, groups
    
    def predict_group_stage(self):
        """Predict all group stage matches and determine group winners"""
        print(" WORLD CUP GROUP STAGE PREDICTIONS ")
        
        fixtures, groups = self.create_group_stage_fixtures()
        
        # Predict all group matches
        predictions = []
        for fixture in fixtures:
            result = self.predict_match(
                fixture['home_team'], 
                fixture['away_team'], 
                fixture['date']
            )
            if result:
                result.update(fixture)
                predictions.append(result)
        
        # Calculate group standings
        group_standings = self.calculate_group_standings(predictions, groups)
        
        return predictions, group_standings
    
    def calculate_group_standings(self, predictions, groups):
        """Calculate group standings based on predictions"""
        standings = {}
        
        for group_name, teams in groups.items():
            team_stats = {team: {'points': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'played': 0} for team in teams}
            
            # Process group matches
            group_matches = [p for p in predictions if p.get('group') == group_name]
            
            for match in group_matches:
                home_team = match['home_team']
                away_team = match['away_team']
                prediction = match['prediction']
                
                team_stats[home_team]['played'] += 1
                team_stats[away_team]['played'] += 1
                
                # Estimate scores based on prediction and team strengths
                if prediction == 'H':
                    home_score, away_score = 2, 1  # Home win
                    team_stats[home_team]['points'] += 3
                elif prediction == 'A':
                    home_score, away_score = 1, 2  # Away win
                    team_stats[away_team]['points'] += 3
                else:
                    home_score, away_score = 1, 1  # Draw
                    team_stats[home_team]['points'] += 1
                    team_stats[away_team]['points'] += 1
                
                team_stats[home_team]['gf'] += home_score
                team_stats[home_team]['ga'] += away_score
                team_stats[away_team]['gf'] += away_score
                team_stats[away_team]['ga'] += home_score
            
            # Calculate goal difference
            for team in teams:
                team_stats[team]['gd'] = team_stats[team]['gf'] - team_stats[team]['ga']
            
            # Sort teams by points, then goal difference, then goals for
            sorted_teams = sorted(teams, key=lambda t: (
                team_stats[t]['points'],
                team_stats[t]['gd'], 
                team_stats[t]['gf']
            ), reverse=True)
            
            standings[group_name] = {
                'teams': sorted_teams,
                'stats': team_stats,
                'qualifiers': sorted_teams[:2]  # Top 2 teams qualify
            }
        
        self.print_group_standings(standings)
        return standings
    
    def print_group_standings(self, standings):
        """Print formatted group standings"""
        print(f"\n{'='*50}")
        print("GROUP STAGE STANDINGS")
        print(f"{'='*50}")
        
        for group_name, group_data in standings.items():
            print(f"\nGROUP {group_name}:")
            print(f"{'Team':<15} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
            print("-" * 50)
            
            for i, team in enumerate(group_data['teams']):
                stats = group_data['stats'][team]
                # Calculate W/D/L from points (simplified)
                wins = stats['points'] // 3
                draws = stats['points'] % 3
                losses = stats['played'] - wins - draws
                
                qualifier = "" if team in group_data['qualifiers'] else ""
                
                print(f"{team:<15} {stats['played']:>2} {wins:>2} {draws:>2} {losses:>2} "
                      f"{stats['gf']:>3} {stats['ga']:>3} {stats['gd']:>+4} {stats['points']:>4} {qualifier}")
        
        # List all qualifiers
        print(f"\n{'='*50}")
        print("ROUND OF 16 QUALIFIERS:")
        print(f"{'='*50}")
        
        all_qualifiers = []
        for group_data in standings.values():
            all_qualifiers.extend(group_data['qualifiers'])
        
        for i, team in enumerate(all_qualifiers, 1):
            print(f"{i:2d}. {team}")
    
    def predict_knockout_stage(self, qualifiers):
        """Predict knockout stage matches"""
        print(f"\n{'='*50}")
        print("KNOCKOUT STAGE PREDICTIONS")
        print(f"{'='*50}")
        
        # Round of 16 (simplified bracket)
        round_16_matches = [
            (qualifiers[0], qualifiers[15]),  # Group A winner vs Group H runner-up
            (qualifiers[2], qualifiers[13]),  # Group B winner vs Group G runner-up
            (qualifiers[4], qualifiers[11]),  # Group C winner vs Group F runner-up
            (qualifiers[6], qualifiers[9]),   # Group D winner vs Group E runner-up
            (qualifiers[1], qualifiers[14]),  # Group A runner-up vs Group H winner
            (qualifiers[3], qualifiers[12]),  # Group B runner-up vs Group G winner
            (qualifiers[5], qualifiers[10]),  # Group C runner-up vs Group F winner
            (qualifiers[7], qualifiers[8])    # Group D runner-up vs Group E winner
        ]
        
        print("\nROUND OF 16:")
        quarter_finalists = []
        
        for i, (team1, team2) in enumerate(round_16_matches, 1):
            result = self.predict_match(team1, team2, '2024-12-10')
            if result:
                winner = team1 if result['prediction'] in ['H', 'D'] else team2
                # In knockouts, assume extra time/penalties resolve draws
                if result['prediction'] == 'D':
                    winner = team1 if result['probabilities']['H'] > result['probabilities']['A'] else team2
                
                quarter_finalists.append(winner)
                print(f"Match {i}: {team1} vs {team2} → {winner} wins")
        
        # Quarter-finals
        print("\nQUARTER-FINALS:")
        semi_finalists = []
        qf_matches = [
            (quarter_finalists[0], quarter_finalists[1]),
            (quarter_finalists[2], quarter_finalists[3]),
            (quarter_finalists[4], quarter_finalists[5]),
            (quarter_finalists[6], quarter_finalists[7])
        ]
        
        for i, (team1, team2) in enumerate(qf_matches, 1):
            result = self.predict_match(team1, team2, '2024-12-14')
            if result:
                winner = team1 if result['prediction'] in ['H', 'D'] else team2
                if result['prediction'] == 'D':
                    winner = team1 if result['probabilities']['H'] > result['probabilities']['A'] else team2
                
                semi_finalists.append(winner)
                print(f"QF{i}: {team1} vs {team2} → {winner} wins")
        
        # Semi-finals
        print("\nSEMI-FINALS:")
        finalists = []
        sf_matches = [
            (semi_finalists[0], semi_finalists[1]),
            (semi_finalists[2], semi_finalists[3])
        ]
        
        for i, (team1, team2) in enumerate(sf_matches, 1):
            result = self.predict_match(team1, team2, '2024-12-18')
            if result:
                winner = team1 if result['prediction'] in ['H', 'D'] else team2
                if result['prediction'] == 'D':
                    winner = team1 if result['probabilities']['H'] > result['probabilities']['A'] else team2
                
                finalists.append(winner)
                print(f"SF{i}: {team1} vs {team2} → {winner} wins")
        
        # Final
        print(f"\n{'='*50}")
        print(" WORLD CUP FINAL ")
        print(f"{'='*50}")
        
        if len(finalists) >= 2:
            final_result = self.predict_match(finalists[0], finalists[1], '2024-12-22')
            if final_result:
                champion = finalists[0] if final_result['prediction'] in ['H', 'D'] else finalists[1]
                if final_result['prediction'] == 'D':
                    champion = finalists[0] if final_result['probabilities']['H'] > final_result['probabilities']['A'] else finalists[1]
                
                print(f"FINAL: {finalists[0]} vs {finalists[1]}")
                print(f" WORLD CUP WINNER: {champion}")
                print(f"Confidence: {final_result['confidence']:.1%}")
                
                return {
                    'champion': champion,
                    'runner_up': finalists[1] if champion == finalists[0] else finalists[0],
                    'semi_finalists': semi_finalists,
                    'quarter_finalists': quarter_finalists,
                    'confidence': final_result['confidence']
                }
        
        return None
    
    def predict_full_tournament(self):
        """Predict the entire World Cup tournament"""
        print(" COMPLETE WORLD CUP PREDICTION ")
        print("Inspired by 85% accuracy AI predictions")
        
        # Group stage
        group_predictions, standings = self.predict_group_stage()
        
        # Get qualifiers
        all_qualifiers = []
        for group_data in standings.values():
            all_qualifiers.extend(group_data['qualifiers'])
        
        # Knockout stage
        tournament_result = self.predict_knockout_stage(all_qualifiers)
        
        if tournament_result:
            print(f"\n{'='*60}")
            print(" FINAL TOURNAMENT PREDICTION ")
            print(f"{'='*60}")
            print(f"Champion: {tournament_result['champion']}")
            print(f"Runner-up: {tournament_result['runner_up']}")
            print(f"Overall Confidence: {tournament_result['confidence']:.1%}")
            print(f"{'='*60}")
        
        return tournament_result
    
    def save_predictions(self, predictions, filename='world_cup_predictions.json'):
        """Save predictions to file"""
        with open(f'../results/{filename}', 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        print(f"Predictions saved to ../results/{filename}")

def main():
    print(" World Cup Prediction System ")
    print("Targeting 85%+ accuracy like tennis predictions")
    
    predictor = WorldCupPredictor()
    
    # Full tournament prediction
    result = predictor.predict_full_tournament()
    
    if result:
        # Save predictions
        predictor.save_predictions(result)

if __name__ == "__main__":
    main()