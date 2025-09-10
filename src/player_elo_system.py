import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class PlayerEloSystem:
    """
    Player-Level ELO System for Football - Tennis Inspired
    
    Just like tennis has individual player rankings, this system creates
    ELO ratings for each player to enhance team prediction accuracy.
    
    Key Features:
    - Individual player ELO ratings
    - Position-specific adjustments  
    - Starting XI impact on team ELO
    - Performance-based ELO updates
    - Age and form weighting
    """
    
    def __init__(self):
        self.default_player_elo = 1200  # Lower than team ELO (1500)
        self.player_elo = {}  # Individual player ratings
        self.player_history = {}  # Player ELO progression
        self.player_info = {}  # Player metadata (position, age, etc.)
        
        # Position-specific ELO ranges (like tennis court surfaces)
        self.position_ranges = {
            'GK': {'min': 1000, 'max': 1600, 'avg': 1200},  # Goalkeepers
            'CB': {'min': 1000, 'max': 1550, 'avg': 1180},  # Center backs
            'LB': {'min': 1000, 'max': 1500, 'avg': 1160},  # Left backs
            'RB': {'min': 1000, 'max': 1500, 'avg': 1160},  # Right backs
            'CDM': {'min': 1050, 'max': 1600, 'avg': 1220}, # Defensive midfielders
            'CM': {'min': 1100, 'max': 1700, 'avg': 1280},  # Central midfielders
            'CAM': {'min': 1150, 'max': 1750, 'avg': 1320}, # Attacking midfielders
            'LW': {'min': 1100, 'max': 1800, 'avg': 1350},  # Left wingers
            'RW': {'min': 1100, 'max': 1800, 'avg': 1350},  # Right wingers
            'ST': {'min': 1200, 'max': 1900, 'avg': 1400},  # Strikers
        }
        
        # Performance impact weights (tennis-inspired)
        self.performance_weights = {
            'goal': 15,      # Goals (like tennis winners)
            'assist': 10,    # Assists (like tennis assists)
            'clean_sheet': 8, # Clean sheets (defensive success)
            'yellow_card': -5, # Discipline (like unforced errors)
            'red_card': -20,  # Major errors
            'own_goal': -15,  # Costly mistakes
            'penalty_save': 20, # Exceptional plays
            'penalty_miss': -12, # Missed opportunities
            'motm': 25,      # Man of the match (outstanding performance)
            'poor_rating': -8 # Poor match rating (<6.0)
        }
        
        # Age impact curve (players peak around 27-29)
        self.age_multipliers = {
            17: 1.2, 18: 1.15, 19: 1.1, 20: 1.05, 21: 1.02, 22: 1.01,
            23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0,
            30: 0.99, 31: 0.98, 32: 0.97, 33: 0.95, 34: 0.93, 35: 0.90,
            36: 0.87, 37: 0.84, 38: 0.80, 39: 0.75, 40: 0.70
        }
    
    def initialize_player(self, player_name, position, age=25, team=None, initial_elo=None):
        """Initialize a player with position-specific ELO"""
        if player_name not in self.player_elo:
            # Set initial ELO based on position
            if initial_elo:
                starting_elo = initial_elo
            else:
                pos_info = self.position_ranges.get(position, self.position_ranges['CM'])
                starting_elo = pos_info['avg'] + np.random.normal(0, 50)
                starting_elo = max(pos_info['min'], min(pos_info['max'], starting_elo))
            
            self.player_elo[player_name] = starting_elo
            self.player_info[player_name] = {
                'position': position,
                'age': age,
                'team': team,
                'matches_played': 0,
                'last_updated': datetime.now()
            }
            self.player_history[player_name] = []
            
            print(f"Initialized {player_name} ({position}, age {age}): {starting_elo:.0f} ELO")
    
    def get_position_k_factor(self, position, performance_type):
        """Get K-factor based on position and performance type"""
        base_k = {
            'GK': 25,   # Goalkeepers (fewer chances to impact)
            'CB': 20,   # Center backs
            'LB': 22,   'RB': 22,  # Full backs
            'CDM': 18,  # Defensive midfielders
            'CM': 16,   # Central midfielders  
            'CAM': 18,  # Attacking midfielders
            'LW': 20,   'RW': 20,  # Wingers
            'ST': 22    # Strikers (high impact)
        }.get(position, 18)
        
        # Adjust based on performance type
        if performance_type in ['goal', 'assist', 'motm']:
            return base_k * 1.2  # Positive performances get higher impact
        elif performance_type in ['red_card', 'own_goal']:
            return base_k * 1.5  # Major errors get higher impact
        else:
            return base_k
    
    def update_player_elo(self, player_name, performance_data, opponent_strength=1500, match_date=None):
        """
        Update player ELO based on match performance
        Similar to how tennis players gain/lose points after matches
        """
        if player_name not in self.player_elo:
            print(f"Warning: Player {player_name} not initialized")
            return
        
        if match_date is None:
            match_date = datetime.now()
        
        current_elo = self.player_elo[player_name]
        player_info = self.player_info[player_name]
        
        # Calculate total ELO change
        total_elo_change = 0
        
        # Process each performance event
        for event, count in performance_data.items():
            if event in self.performance_weights:
                base_change = self.performance_weights[event] * count
                
                # Get position-specific K-factor
                k_factor = self.get_position_k_factor(player_info['position'], event)
                
                # Opponent strength adjustment (harder opponents = more ELO gain/loss)
                opponent_factor = opponent_strength / 1500  # Normalize to 1.0
                
                # Age adjustment
                age = player_info['age']
                age_factor = self.age_multipliers.get(age, 0.95)
                
                # Calculate final change
                elo_change = base_change * (k_factor / 20) * opponent_factor * age_factor
                total_elo_change += elo_change
        
        # Apply form factor (recent performance weighting)
        recent_form = self.get_recent_form(player_name)
        form_factor = 1.0 + (recent_form - 1200) / 2000  # Boost for good form
        total_elo_change *= form_factor
        
        # Update player ELO
        new_elo = current_elo + total_elo_change
        
        # Apply position limits
        pos_limits = self.position_ranges.get(player_info['position'], self.position_ranges['CM'])
        new_elo = max(pos_limits['min'], min(pos_limits['max'], new_elo))
        
        self.player_elo[player_name] = new_elo
        
        # Update player info
        self.player_info[player_name]['matches_played'] += 1
        self.player_info[player_name]['last_updated'] = match_date
        
        # Store history
        self.player_history[player_name].append({
            'date': match_date,
            'elo': new_elo,
            'change': total_elo_change,
            'performance': performance_data.copy(),
            'opponent_strength': opponent_strength
        })
        
        return {
            'player': player_name,
            'old_elo': current_elo,
            'new_elo': new_elo,
            'change': total_elo_change,
            'performance': performance_data
        }
    
    def get_recent_form(self, player_name, matches=5):
        """Get player's recent form (average ELO over last N matches)"""
        if player_name not in self.player_history:
            return self.default_player_elo
        
        recent_matches = sorted(self.player_history[player_name], 
                              key=lambda x: x['date'], reverse=True)[:matches]
        
        if not recent_matches:
            return self.player_elo.get(player_name, self.default_player_elo)
        
        return np.mean([match['elo'] for match in recent_matches])
    
    def calculate_starting_xi_elo(self, starting_xi_data):
        """
        Calculate team ELO bonus/penalty based on starting XI
        
        starting_xi_data format:
        {
            'GK': 'Alisson',
            'CB': ['Van Dijk', 'Matip'], 
            'LB': 'Robertson',
            'RB': 'Alexander-Arnold',
            'CDM': 'Fabinho',
            'CM': ['Henderson', 'Thiago'],
            'LW': 'Mane',
            'RW': 'Salah',
            'ST': 'Firmino'
        }
        """
        total_player_elo = 0
        player_count = 0
        position_bonuses = 0
        
        for position, players in starting_xi_data.items():
            if isinstance(players, str):
                players = [players]
            elif isinstance(players, list):
                pass
            else:
                continue
                
            for player in players:
                if player in self.player_elo:
                    player_elo = self.player_elo[player]
                    total_player_elo += player_elo
                    player_count += 1
                    
                    # Position-specific bonuses
                    if position in ['ST', 'CAM'] and player_elo > 1600:
                        position_bonuses += 20  # Star attackers
                    elif position == 'GK' and player_elo > 1400:
                        position_bonuses += 15  # World-class goalkeeper
                    elif position in ['CB'] and player_elo > 1450:
                        position_bonuses += 12  # Elite defender
                
        if player_count == 0:
            return 0
        
        # Calculate average player ELO
        avg_player_elo = total_player_elo / player_count
        
        # Convert to team ELO adjustment (scale appropriately)
        baseline_player_elo = 1200
        elo_adjustment = (avg_player_elo - baseline_player_elo) * 0.3  # Scale down for team impact
        elo_adjustment += position_bonuses
        
        return {
            'elo_adjustment': elo_adjustment,
            'avg_player_elo': avg_player_elo,
            'player_count': player_count,
            'position_bonuses': position_bonuses,
            'star_players': [p for p in starting_xi_data.values() if isinstance(p, str) and p in self.player_elo and self.player_elo[p] > 1600] + 
                           [p for players in starting_xi_data.values() if isinstance(players, list) for p in players if p in self.player_elo and self.player_elo[p] > 1600]
        }
    
    def get_enhanced_team_elo(self, base_team_elo, starting_xi_data, key_player_statuses=None):
        """
        Calculate enhanced team ELO considering starting XI and key player availability
        
        key_player_statuses format:
        {
            'Messi': 'available',
            'Neymar': 'injured', 
            'Mbappe': 'suspended'
        }
        """
        # Get starting XI adjustment
        xi_analysis = self.calculate_starting_xi_elo(starting_xi_data)
        starting_xi_adjustment = xi_analysis['elo_adjustment']
        
        # Key player availability adjustments
        availability_adjustment = 0
        if key_player_statuses:
            for player, status in key_player_statuses.items():
                if player in self.player_elo:
                    player_elo = self.player_elo[player]
                    player_impact = (player_elo - 1200) * 0.2  # Scale player impact
                    
                    if status == 'injured':
                        availability_adjustment -= player_impact * 1.2  # Injury hurts more
                    elif status == 'suspended':
                        availability_adjustment -= player_impact
                    elif status == 'doubtful':
                        availability_adjustment -= player_impact * 0.5
        
        # Calculate final enhanced ELO
        enhanced_elo = base_team_elo + starting_xi_adjustment + availability_adjustment
        
        return {
            'enhanced_elo': enhanced_elo,
            'base_elo': base_team_elo,
            'starting_xi_adjustment': starting_xi_adjustment,
            'availability_adjustment': availability_adjustment,
            'xi_analysis': xi_analysis
        }
    
    def create_sample_player_database(self):
        """Create sample player database with realistic ELO ratings"""
        print("Creating sample player database...")
        
        # Premier League sample players
        sample_players = {
            # Manchester City
            'Erling Haaland': {'position': 'ST', 'age': 23, 'team': 'Manchester City', 'elo': 1750},
            'Kevin De Bruyne': {'position': 'CAM', 'age': 32, 'team': 'Manchester City', 'elo': 1720},
            'Ederson': {'position': 'GK', 'age': 30, 'team': 'Manchester City', 'elo': 1480},
            'Ruben Dias': {'position': 'CB', 'age': 26, 'team': 'Manchester City', 'elo': 1520},
            
            # Arsenal  
            'Bukayo Saka': {'position': 'RW', 'age': 22, 'team': 'Arsenal', 'elo': 1650},
            'Martin Odegaard': {'position': 'CAM', 'age': 25, 'team': 'Arsenal', 'elo': 1620},
            'William Saliba': {'position': 'CB', 'age': 23, 'team': 'Arsenal', 'elo': 1480},
            'Aaron Ramsdale': {'position': 'GK', 'age': 26, 'team': 'Arsenal', 'elo': 1420},
            
            # Liverpool
            'Mohamed Salah': {'position': 'RW', 'age': 31, 'team': 'Liverpool', 'elo': 1700},
            'Virgil van Dijk': {'position': 'CB', 'age': 32, 'team': 'Liverpool', 'elo': 1550},
            'Alisson': {'position': 'GK', 'age': 30, 'team': 'Liverpool', 'elo': 1520},
            'Sadio Mane': {'position': 'LW', 'age': 31, 'team': 'Liverpool', 'elo': 1680},
            
            # Chelsea
            'Reece James': {'position': 'RB', 'age': 24, 'team': 'Chelsea', 'elo': 1480},
            'Thiago Silva': {'position': 'CB', 'age': 39, 'team': 'Chelsea', 'elo': 1420},
            'Enzo Fernandez': {'position': 'CM', 'age': 23, 'team': 'Chelsea', 'elo': 1450},
            
            # International Stars
            'Kylian Mbappe': {'position': 'ST', 'age': 25, 'team': 'PSG', 'elo': 1800},
            'Lionel Messi': {'position': 'CAM', 'age': 36, 'team': 'Inter Miami', 'elo': 1650},  # Age adjustment
            'Cristiano Ronaldo': {'position': 'ST', 'age': 39, 'team': 'Al Nassr', 'elo': 1580},  # Age adjustment
            'Vinicius Jr': {'position': 'LW', 'age': 23, 'team': 'Real Madrid', 'elo': 1720},
            'Pedri': {'position': 'CM', 'age': 21, 'team': 'Barcelona', 'elo': 1580},
            'Gavi': {'position': 'CM', 'age': 19, 'team': 'Barcelona', 'elo': 1520},
        }
        
        # Initialize all sample players
        for player_name, info in sample_players.items():
            self.initialize_player(
                player_name=player_name,
                position=info['position'],
                age=info['age'],
                team=info['team'],
                initial_elo=info['elo']
            )
        
        print(f"Initialized {len(sample_players)} players in database")
        return len(sample_players)
    
    def get_top_players_by_position(self, position=None, n=10):
        """Get top players overall or by position"""
        filtered_players = []
        
        for player_name, elo in self.player_elo.items():
            player_info = self.player_info[player_name]
            if position is None or player_info['position'] == position:
                filtered_players.append((player_name, elo, player_info))
        
        # Sort by ELO
        filtered_players.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_players[:n]
    
    def simulate_match_with_players(self, home_xi, away_xi, home_base_elo=1500, away_base_elo=1500):
        """
        Simulate a match with specific starting XIs
        """
        print(f"\nSimulating match with player-enhanced ELO...")
        
        # Calculate enhanced team ELO
        home_enhanced = self.get_enhanced_team_elo(home_base_elo, home_xi)
        away_enhanced = self.get_enhanced_team_elo(away_base_elo, away_xi)
        
        home_elo = home_enhanced['enhanced_elo']
        away_elo = away_enhanced['enhanced_elo']
        
        # Add home advantage
        home_elo += 100
        
        # Calculate win probability
        elo_diff = home_elo - away_elo
        home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        print(f"Home Enhanced ELO: {home_elo:.0f} (base: {home_base_elo}, adjustment: {home_enhanced['starting_xi_adjustment']:+.0f})")
        print(f"Away Enhanced ELO: {away_elo:.0f} (base: {away_base_elo}, adjustment: {away_enhanced['starting_xi_adjustment']:+.0f})")
        print(f"Home Win Probability: {home_win_prob:.1%}")
        print(f"Draw Probability: 25%")  
        print(f"Away Win Probability: {1-home_win_prob:.1%}")
        
        return {
            'home_enhanced_elo': home_elo,
            'away_enhanced_elo': away_elo,
            'home_win_probability': home_win_prob,
            'away_win_probability': 1 - home_win_prob,
            'home_analysis': home_enhanced,
            'away_analysis': away_enhanced
        }
    
    def save_player_data(self, filepath):
        """Save player ELO data"""
        data = {
            'player_elo': self.player_elo,
            'player_info': {
                player: {
                    **info,
                    'last_updated': info['last_updated'].isoformat() if isinstance(info['last_updated'], datetime) else str(info['last_updated'])
                }
                for player, info in self.player_info.items()
            },
            'player_history': {
                player: [
                    {
                        **record,
                        'date': record['date'].isoformat() if isinstance(record['date'], datetime) else str(record['date'])
                    }
                    for record in history
                ]
                for player, history in self.player_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Player ELO data saved to {filepath}")

def main():
    # Demo the player ELO system
    player_elo = PlayerEloSystem()
    
    print("Tennis-Inspired Player ELO System for Football")
    print("=" * 60)
    
    # Create sample database
    player_elo.create_sample_player_database()
    
    # Show top players overall
    print(f"\nTop 10 Players Overall:")
    top_players = player_elo.get_top_players_by_position()
    for i, (player, elo, info) in enumerate(top_players, 1):
        print(f"{i:2d}. {player:<20} {elo:4.0f} ({info['position']}, {info['team']})")
    
    # Show top strikers
    print(f"\nTop 5 Strikers:")
    top_strikers = player_elo.get_top_players_by_position('ST', 5)
    for i, (player, elo, info) in enumerate(top_strikers, 1):
        print(f"{i:2d}. {player:<20} {elo:4.0f} ({info['team']})")
    
    # Example starting XIs
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
    
    # Simulate match
    print(f"\nMatch Simulation: Manchester City vs Arsenal")
    print("-" * 50)
    result = player_elo.simulate_match_with_players(
        home_xi=man_city_xi,
        away_xi=arsenal_xi, 
        home_base_elo=1580,
        away_base_elo=1520
    )
    
    # Save data
    os.makedirs('../models', exist_ok=True)
    player_elo.save_player_data('../models/player_elo_system.json')

if __name__ == "__main__":
    main()