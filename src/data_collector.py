import requests
import pandas as pd
import json
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os

class FootballDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data_dir = '../data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_team_stats(self, league='premier-league', season='2023-24'):
        """Collect team statistics from football data sources"""
        print(f"Collecting team stats for {league} {season}...")
        
        # Sample team stats structure - replace with actual API calls
        teams_data = []
        
        # Example team stats (you'll replace this with real data sources)
        sample_teams = ['Arsenal', 'Manchester City', 'Liverpool', 'Chelsea']
        
        for team in sample_teams:
            team_stats = {
                'team_name': team,
                'season': season,
                'matches_played': 38,
                'wins': 25,
                'draws': 8,
                'losses': 5,
                'goals_for': 75,
                'goals_against': 35,
                'goal_difference': 40,
                'points': 83,
                'avg_possession': 58.5,
                'shots_per_game': 15.2,
                'shots_on_target_per_game': 5.8,
                'pass_accuracy': 88.2,
                'tackles_per_game': 18.5,
                'fouls_per_game': 11.2,
                'cards_per_game': 2.1,
                'form_last_5': 'WWDWW'
            }
            teams_data.append(team_stats)
        
        df = pd.DataFrame(teams_data)
        df.to_csv(f'{self.data_dir}/team_stats_{season}.csv', index=False)
        print(f"Saved team stats to {self.data_dir}/team_stats_{season}.csv")
        return df
    
    def collect_match_results(self, league='premier-league', season='2023-24'):
        """Collect historical match results"""
        print(f"Collecting match results for {league} {season}...")
        
        # Sample match data - replace with actual API calls
        matches_data = []
        
        # Example matches
        sample_matches = [
            {'home_team': 'Arsenal', 'away_team': 'Manchester City', 'home_score': 2, 'away_score': 1, 'date': '2024-01-15'},
            {'home_team': 'Liverpool', 'away_team': 'Chelsea', 'home_score': 3, 'away_score': 0, 'date': '2024-01-16'},
            {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'home_score': 1, 'away_score': 2, 'date': '2024-01-20'},
        ]
        
        for match in sample_matches:
            match_data = {
                'date': match['date'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'home_score': match['home_score'],
                'away_score': match['away_score'],
                'result': 'H' if match['home_score'] > match['away_score'] else 'A' if match['away_score'] > match['home_score'] else 'D',
                'total_goals': match['home_score'] + match['away_score']
            }
            matches_data.append(match_data)
        
        df = pd.DataFrame(matches_data)
        df.to_csv(f'{self.data_dir}/match_results_{season}.csv', index=False)
        print(f"Saved match results to {self.data_dir}/match_results_{season}.csv")
        return df
    
    def collect_player_stats(self, league='premier-league', season='2023-24'):
        """Collect player statistics"""
        print(f"Collecting player stats for {league} {season}...")
        
        # Sample player data
        players_data = []
        
        sample_players = [
            {'name': 'Erling Haaland', 'team': 'Manchester City', 'position': 'Forward', 'goals': 27, 'assists': 5},
            {'name': 'Harry Kane', 'team': 'Tottenham', 'position': 'Forward', 'goals': 25, 'assists': 3},
            {'name': 'Mohamed Salah', 'team': 'Liverpool', 'position': 'Forward', 'goals': 22, 'assists': 8},
        ]
        
        for player in sample_players:
            player_stats = {
                'player_name': player['name'],
                'team': player['team'],
                'position': player['position'],
                'goals': player['goals'],
                'assists': player['assists'],
                'minutes_played': 2800,
                'shots_per_game': 4.2,
                'pass_accuracy': 82.5,
                'season': season
            }
            players_data.append(player_stats)
        
        df = pd.DataFrame(players_data)
        df.to_csv(f'{self.data_dir}/player_stats_{season}.csv', index=False)
        print(f"Saved player stats to {self.data_dir}/player_stats_{season}.csv")
        return df
    
    def collect_all_data(self, seasons=['2022-23', '2023-24']):
        """Collect all required data for multiple seasons"""
        all_team_stats = []
        all_match_results = []
        all_player_stats = []
        
        for season in seasons:
            print(f"\n=== Collecting data for {season} ===")
            
            # Collect data for each season
            team_stats = self.collect_team_stats(season=season)
            match_results = self.collect_match_results(season=season)
            player_stats = self.collect_player_stats(season=season)
            
            all_team_stats.append(team_stats)
            all_match_results.append(match_results)
            all_player_stats.append(player_stats)
            
            # Rate limiting
            time.sleep(2)
        
        # Combine all seasons
        combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
        combined_match_results = pd.concat(all_match_results, ignore_index=True)
        combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
        
        # Save combined data
        combined_team_stats.to_csv(f'{self.data_dir}/all_team_stats.csv', index=False)
        combined_match_results.to_csv(f'{self.data_dir}/all_match_results.csv', index=False)
        combined_player_stats.to_csv(f'{self.data_dir}/all_player_stats.csv', index=False)
        
        print(f"\n=== Data collection complete ===")
        print(f"Team stats: {len(combined_team_stats)} records")
        print(f"Match results: {len(combined_match_results)} records")
        print(f"Player stats: {len(combined_player_stats)} records")

if __name__ == "__main__":
    collector = FootballDataCollector()
    collector.collect_all_data()