"""Plot Generator Module

Generates matplotlib/seaborn visualizations for the FPL report.
Replicates styles from https://alpscode.com/blog/hindsight-optimization/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import squarify
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")

class PlotGenerator:
    """Generates visualizations for FPL analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FPL Colors
        self.colors = {
            'purple': '#37003c',
            'green': '#00ff87', 
            'pink': '#ff2f82',
            'cyan': '#00ffff',
            'blue': '#0057ff', 
            'white': '#ffffff',
            'gray': '#808080',
            'bench': '#f8d7da',  # Light red for bench
            'captain': '#d4edda', # Light green for captain
            'vice': '#fff3cd',    # Light yellow for VC
            'lineup': '#d1ecf1'   # Light blue for lineup
        }
        
        # Chip symbols and abbreviations
        self.chip_symbols = {
            'wildcard': ('*', 'WC'),
            'freehit': ('s', 'FH'),
            'bboost': ('D', 'BB'),
            '3xc': ('^', 'TC')
        }
        
        # Colors for different teams in competitive plots
        self.team_colors = [
            '#37003c', '#00ff87', '#ff2f82', '#0057ff', '#ff9500',
            '#00c8ff', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12'
        ]

    def _save_plot(self, filename: str):
        """Save the current plot to file."""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_contributing_points(self, season_history: List[Dict]) -> Dict[str, Dict]:
        """Calculate points only when players were in starting XI.
        
        Only counts points from gameweeks where the player was in the starting XI
        (position_in_squad <= 11), excluding benched players' points.
        Applies captain multiplier (2x for captain, 3x for triple captain).
        
        Args:
            season_history: List of gameweek entries with squad information.
            
        Returns:
            Dict mapping player names to their contribution data:
            {name: {'points': int, 'position': str}}
        """
        player_contributions = {}
        
        for gw_entry in season_history:
            squad = gw_entry.get('squad', [])
            for player in squad:
                # Only count if in starting XI (position_in_squad <= 11)
                position_in_squad = player.get('position_in_squad', 0)
                if position_in_squad <= 11:
                    name = player.get('name', 'Unknown')
                    base_points = player.get('stats', {}).get('event_points', 0) or 0
                    position = player.get('position', 'UNK')
                    
                    # Apply captain multiplier (2 for captain, 3 for triple captain)
                    multiplier = player.get('multiplier', 1)
                    points = base_points * multiplier
                    
                    if name not in player_contributions:
                        player_contributions[name] = {'points': 0, 'position': position}
                    
                    player_contributions[name]['points'] += points
        
        return player_contributions

    def generate_points_per_gw(self, history: List[Dict], chips: List[Dict]):
        """Generate Points per GW bar chart showing gross points with hit deductions."""
        if not history:
            return

        df = pd.DataFrame(history)
        # Check if 'points' and 'event_transfers_cost' exist
        if 'points' not in df.columns:
            return
        
        df['hits'] = df.get('event_transfers_cost', 0)
        df['gross_points'] = df['points']
        df['net_points'] = df['points'] - df['hits']
        df['gw'] = df['event']

        plt.figure(figsize=(12, 6))
        
        # Stacked bar chart: net points (green) + hits (red overlay at bottom)
        # We'll show gross points as main bars, with hit portion in red at bottom
        bars_net = plt.bar(df['gw'], df['net_points'], color='#66c2a5', width=0.8, label='Net Points')
        bars_hits = plt.bar(df['gw'], df['hits'], bottom=df['net_points'], 
                           color='#e74c3c', width=0.8, alpha=0.8, label='Hits')
        
        # Add labels showing net points
        for i, (gw, net_pts, hits) in enumerate(zip(df['gw'], df['net_points'], df['hits'])):
            # Net points label
            plt.text(gw, net_pts + hits + 2, f'{int(net_pts)}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Hit label (if there was a hit)
            if hits > 0:
                plt.text(gw, net_pts + hits/2, f'-{int(hits)}', 
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

        # Annotate chips with symbols
        chip_map = {c['event']: c['name'] for c in chips}
        
        for gw, name in chip_map.items():
            if gw in df['gw'].values:
                # Find the bar height for this GW
                row = df[df['gw'] == gw].iloc[0]
                height = row['gross_points']
                marker, abbr = self.chip_symbols.get(name, ('o', name[:2].upper()))
                
                # Plot symbol above the bar
                plt.plot(gw, height + 8, marker=marker, markersize=12, 
                        color='purple', markeredgecolor='darkviolet', markeredgewidth=1.5)
                plt.text(gw, height + 12, abbr, ha='center', va='bottom', 
                         color='purple', fontweight='bold', fontsize=9)

        plt.title('Points per Gameweek', fontsize=14, fontweight='bold')
        plt.xlabel('Gameweeks')
        plt.ylabel('Points')
        plt.xticks(df['gw'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        self._save_plot('points_per_gw.png')

    def generate_contribution_heatmap(self, squad_history: List[Dict]):
        """Generate player contribution heatmap/Gantt chart.
        
        Shows when players were in lineup, captained, or benched, and their points.
        """
        if not squad_history:
            return

        # Flatten data for dataframe
        data = []
        all_players = set()
        
        for gw_entry in squad_history:
            gw = gw_entry['gameweek']
            for player in gw_entry['squad']:
                pid = player['id']
                name = player['name']
                all_players.add(name)
                
                status = 'Lineup'
                if player['position_in_squad'] > 11:
                    status = 'Bench'
                
                if player['is_captain']:
                    status = 'Captain'
                
                points = player['stats'].get('event_points', 0)
                # If player didn't play (0 points usually, but check minutes if possible),
                # sometimes 0 points is valid play. 
                # For simplicity, we just show points.
                
                data.append({
                    'Player': name,
                    'GW': gw,
                    'Points': points,
                    'Status': status
                })

        df = pd.DataFrame(data)
        
        if df.empty:
            return

        # Filter to top 20 contributors (by total points) to keep chart readable
        player_totals = df.groupby('Player')['Points'].sum().sort_values(ascending=False)
        top_players = player_totals.head(20).index.tolist()
        df = df[df['Player'].isin(top_players)]

        # Pivot for heatmap
        pivot_points = df.pivot(index='Player', columns='GW', values='Points').fillna(0)
        # Reorder rows by total points
        pivot_points = pivot_points.reindex(top_players)
        
        # Pivot for status to color code
        pivot_status = df.pivot(index='Player', columns='GW', values='Status')
        pivot_status = pivot_status.reindex(top_players)

        # Plot setup
        plt.figure(figsize=(16, max(8, len(top_players) * 0.5)))
        
        # Create a numeric matrix for coloring
        # 0: Not owned, 1: Bench, 2: Lineup, 3: Captain
        status_map = {'Bench': 1, 'Lineup': 2, 'Captain': 3}
        color_matrix = pivot_status.replace(status_map).fillna(0)
        
        # Custom colormap
        # 0: white, 1: pink (bench), 2: light blue (lineup), 3: green (captain)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white', '#fce8e6', '#e3f2fd', '#d4edda'])
        
        ax = sns.heatmap(color_matrix, cmap=cmap, cbar=False, linewidths=1, linecolor='white')
        
        # Annotate with points
        for i in range(len(pivot_points.index)):
            for j in range(len(pivot_points.columns)):
                val = pivot_points.iloc[i, j]
                gw = pivot_points.columns[j]
                player = pivot_points.index[i]
                
                # Check if player was owned
                status = pivot_status.loc[player, gw]
                if pd.notna(status):
                    text = f"{int(val)}"
                    weight = 'bold' if status == 'Captain' else 'normal'
                    color = 'black'
                    ax.text(j + 0.5, i + 0.5, text, 
                           ha='center', va='center', color=color, fontweight=weight)

        plt.title('Players with most contributions', fontsize=16)
        plt.xlabel('Gameweeks')
        plt.ylabel('')
        
        # Create legend
        legend_patches = [
            mpatches.Patch(color='#d1ecf1', label='Lineup'),
            mpatches.Patch(color='#d4edda', label='Captain'),
            mpatches.Patch(color='#f8d7da', label='Bench')
        ]
        plt.legend(handles=legend_patches, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        self._save_plot('player_contribution.png')

    def generate_treemap(self, season_history: List[Dict]):
        """Generate treemap of contributing points by position/player.
        
        Only counts points from gameweeks where the player was in the starting XI,
        excluding benched players' points. Applies captain multiplier.
        
        Args:
            season_history: List of gameweek entries with squad information.
        """
        if not season_history:
            return

        # Determine GW range from season_history
        gameweeks = [gw.get('gameweek', 0) for gw in season_history]
        min_gw = min(gameweeks) if gameweeks else 1
        max_gw = max(gameweeks) if gameweeks else 1

        # Calculate contributing points (only when in starting XI)
        player_contributions = self._calculate_contributing_points(season_history)
        
        if not player_contributions:
            return
        
        # Prepare data for treemap
        data = []
        for name, contrib_data in player_contributions.items():
            data.append({
                'name': name,
                'points': contrib_data['points'],
                'position': contrib_data['position']
            })
        
        df = pd.DataFrame(data)
        df = df[df['points'] > 0]  # Filter 0 points to avoid errors
        df = df.sort_values('points', ascending=False)
        
        if df.empty:
            return
        
        # Define position colors
        pos_colors = {
            'GKP': '#ff9999',  # Red-ish
            'DEF': '#66b3ff',  # Blue-ish
            'MID': '#99ff99',  # Green-ish
            'FWD': '#ffcc99'   # Orange-ish
        }
        
        # Assign colors based on position
        colors = [pos_colors.get(r['position'], '#cccccc') for _, r in df.iterrows()]
        
        plt.figure(figsize=(14, 8))
        
        # Labels with name and points
        labels = [f"{r['name']}\n({r['points']})" for _, r in df.iterrows()]
        
        squarify.plot(sizes=df['points'], label=labels, color=colors, alpha=0.8, 
                      text_kwargs={'fontsize': 9, 'wrap': True})
        
        # Calculate total contributing points
        total_pts = df['points'].sum()
        plt.title(f'Points per Player Distribution (GW{min_gw}-GW{max_gw})\nTotal Contributing Points: {total_pts}', fontsize=16)
        plt.axis('off')
        
        # Add legend for positions
        patches = [mpatches.Patch(color=c, label=p) for p, c in pos_colors.items()]
        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        self._save_plot('points_treemap.png')

    def generate_transfer_matrix(self, transfers: List[Dict]):
        """Generate matrix of transfer counts by category and position."""
        if not transfers:
            return
            
        # Define price categories
        def get_price_cat(pos, price):
            if pos == 'GKP':
                if price < 5.0: return 'Budget'
                if price <= 5.5: return 'Mid-price'
                return 'Premium'
            elif pos == 'DEF':
                if price < 5.0: return 'Budget'
                if price <= 6.0: return 'Mid-price'
                return 'Premium'
            elif pos == 'MID':
                if price < 7.5: return 'Budget'
                if price <= 10.0: return 'Mid-price'
                return 'Premium'
            elif pos == 'FWD':
                if price < 7.5: return 'Budget'
                if price <= 10.0: return 'Mid-price'
                return 'Premium'
            return 'Budget'

        data = []
        for t in transfers:
            # We care about transfers IN
            pos = t.get('element_in_position')
            cost = t.get('element_in_cost_m')
            
            if pos and cost:
                cat = get_price_cat(pos, cost)
                data.append({'Position': pos, 'Category': cat})
                
        df = pd.DataFrame(data)
        
        if df.empty:
            return
            
        # Create pivot table for heatmap
        # Categories: Budget, Mid-price, Premium
        # Positions: GKP, DEF, MID, FWD
        
        pivot = df.pivot_table(index='Category', columns='Position', aggfunc='size', fill_value=0)
        
        # Reorder rows and columns
        cats = ['Budget', 'Mid-price', 'Premium']
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        
        # Ensure all exist
        pivot = pivot.reindex(index=cats, columns=positions, fill_value=0)
        
        # Add total column and row
        pivot['Total'] = pivot.sum(axis=1)
        pivot.loc['Total'] = pivot.sum()
        
        # Plotting - Use a custom heatmap-like grid with text
        # Since the example is specific, we'll try to replicate the table look with matplotlib table or heatmap
        
        plt.figure(figsize=(10, 6))
        
        # We'll use a heatmap but mask the totals for different coloring or just include them
        # Let's do a heatmap of the main data and text for totals
        
        main_data = pivot.iloc[:-1, :-1]
        
        # Heatmap
        sns.heatmap(main_data, annot=True, fmt='d', cmap='Oranges', cbar=False,
                   annot_kws={'size': 14, 'weight': 'bold'})
                   
        plt.title('Transfer Counts by Category and Position', fontsize=16)
        plt.yticks(rotation=0)
        
        self._save_plot('transfer_matrix.png')

    def generate_competitive_points_per_gw(self, competitive_data: List[Dict], filename: str = 'competitive_points_per_gw.png'):
        """Generate grouped bar chart comparing points per GW for all teams.
        
        Shows bars side-by-side for each team per gameweek with chip markers and hits.
        
        Args:
            competitive_data: List of dicts with team_info, gw_history, and chips_used.
            filename: Output filename for the plot.
        """
        if not competitive_data:
            return
        
        # Determine max gameweek across all teams
        max_gw = 0
        for entry in competitive_data:
            gw_history = entry.get('gw_history', [])
            if gw_history:
                max_gw = max(max_gw, max(gw.get('event', 0) for gw in gw_history))
        
        if max_gw == 0:
            return
        
        # Prepare data structure: {gw: {team_name: {'net': net_points, 'hits': hits}}}
        gw_data = {}
        team_chips = {}  # {team_name: {gw: chip_name}}
        team_names = []
        
        for idx, entry in enumerate(competitive_data):
            team_info = entry.get('team_info', {})
            gw_history = entry.get('gw_history', [])
            chips_used = entry.get('chips_used', [])
            
            team_name = team_info.get('team_name', f"Team {entry.get('entry_id', idx)}")
            team_names.append(team_name)
            
            # Build chip map for this team
            chip_map = {c.get('event'): c.get('name') for c in chips_used if c.get('event')}
            team_chips[team_name] = chip_map
            
            # Extract net points and hits per GW
            for gw_entry in gw_history:
                gw = gw_entry.get('event', 0)
                points = gw_entry.get('points', 0)
                transfers_cost = gw_entry.get('event_transfers_cost', 0)
                net_points = points - transfers_cost
                
                if gw not in gw_data:
                    gw_data[gw] = {}
                gw_data[gw][team_name] = {'net': net_points, 'hits': transfers_cost}
        
        # Create plot
        gameweeks = sorted(gw_data.keys())
        n_teams = len(team_names)
        bar_width = 0.8 / n_teams
        
        fig, ax = plt.subplots(figsize=(max(14, max_gw * 0.8), 8))
        
        # Plot stacked bars for each team (net points + hits)
        for idx, team_name in enumerate(team_names):
            positions = []
            net_heights = []
            hit_heights = []
            
            for gw in gameweeks:
                data = gw_data[gw].get(team_name, {'net': 0, 'hits': 0})
                positions.append(gw + (idx - n_teams/2) * bar_width + bar_width/2)
                net_heights.append(data['net'])
                hit_heights.append(data['hits'])
            
            color = self.team_colors[idx % len(self.team_colors)]
            
            # Plot net points
            bars_net = ax.bar(positions, net_heights, bar_width * 0.9, 
                             label=team_name, color=color, alpha=0.85)
            
            # Plot hits as red portion on top
            bars_hits = ax.bar(positions, hit_heights, bar_width * 0.9,
                              bottom=net_heights, color='#e74c3c', alpha=0.6)
            
            # Add chip markers
            for i, (pos, net_h, hit_h, gw) in enumerate(zip(positions, net_heights, hit_heights, gameweeks)):
                chip_name = team_chips[team_name].get(gw)
                total_h = net_h + hit_h
                if chip_name and total_h > 0:
                    marker, abbr = self.chip_symbols.get(chip_name, ('o', chip_name[:2].upper()))
                    # Plot symbol above the bar
                    ax.plot(pos, total_h + 3, marker=marker, markersize=8,
                           color='black', markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        
        ax.set_title('Points per GW Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Gameweek', fontsize=12)
        ax.set_ylabel('Points', fontsize=12)
        ax.set_xticks(gameweeks)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create combined legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        # Team legend elements
        team_legend_elements = [
            Patch(facecolor=self.team_colors[idx % len(self.team_colors)], 
                  alpha=0.85, label=team_name)
            for idx, team_name in enumerate(team_names)
        ]
        
        # Hits legend element
        hits_element = Patch(facecolor='#e74c3c', alpha=0.6, label='Hits')
        
        # Chip legend elements
        chip_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='black',
                   markeredgecolor='white', markersize=8, label=abbr)
            for chip_name, (marker, abbr) in self.chip_symbols.items()
        ]
        
        # Add team legend
        team_legend = ax.legend(handles=team_legend_elements + [hits_element], 
                               loc='upper left', fontsize=9, ncol=min(3, n_teams + 1))
        ax.add_artist(team_legend)
        
        # Add chip legend
        chip_legend = ax.legend(handles=chip_legend_elements, loc='upper right',
                               fontsize=9, title='Chips', framealpha=0.9)
        ax.add_artist(chip_legend)
        
        self._save_plot(filename)

    def generate_competitive_points_progression(self, competitive_data: List[Dict], filename: str = 'competitive_points_progression.png'):
        """Generate multi-team total points progression line plot with hit markers.

        Args:
            competitive_data: List of dicts with team_info and gw_history.
            filename: Output filename for the plot.
        """
        if not competitive_data:
            return

        plt.figure(figsize=(14, 8))

        for idx, entry in enumerate(competitive_data):
            team_info = entry.get('team_info', {})
            gw_history = entry.get('gw_history', [])
            chips_used = entry.get('chips_used', [])

            if not gw_history:
                continue

            team_name = team_info.get('team_name', f"Team {entry.get('entry_id', idx)}")

            # Sort by event/gameweek
            sorted_history = sorted(gw_history, key=lambda x: x.get('event', 0))

            gameweeks = [gw.get('event', 0) for gw in sorted_history]
            total_points = [gw.get('total_points', 0) for gw in sorted_history]

            color = self.team_colors[idx % len(self.team_colors)]
            plt.plot(gameweeks, total_points, marker='o', linewidth=2,
                     markersize=6, label=team_name, color=color)
            
            # Add chip markers
            chip_map = {c.get('event'): c.get('name') for c in chips_used if c.get('event')}
            for gw_entry in sorted_history:
                gw = gw_entry.get('event', 0)
                pts = gw_entry.get('total_points', 0)
                
                # Mark chips with special symbols
                if gw in chip_map:
                    chip_name = chip_map[gw]
                    marker, abbr = self.chip_symbols.get(chip_name, ('o', chip_name[:2].upper()))
                    plt.plot(gw, pts, marker=marker, markersize=12, 
                            color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                
                # Mark hits with X symbol
                hits = gw_entry.get('event_transfers_cost', 0)
                if hits > 0:
                    plt.plot(gw, pts, marker='x', markersize=10,
                            color='red', markeredgecolor='darkred', markeredgewidth=2, zorder=4)

        plt.title('Points Progression Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Total Points', fontsize=12)
        
        # Team legend
        plt.legend(loc='upper left', fontsize=10)
        
        # Add chip and hit legends
        from matplotlib.lines import Line2D
        chip_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label=abbr)
            for chip_name, (marker, abbr) in self.chip_symbols.items()
        ]
        # Add hit marker to legend
        hit_element = Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                            markeredgecolor='darkred', markersize=10, markeredgewidth=2, label='Hit')
        chip_legend_elements.append(hit_element)
        
        chip_legend = plt.legend(handles=chip_legend_elements, loc='lower right',
                                fontsize=9, title='Chips & Hits', framealpha=0.9)
        plt.gca().add_artist(chip_legend)
        
        # Restore team legend
        plt.legend(loc='upper left', fontsize=10)
        plt.gca().add_artist(chip_legend)
        
        plt.grid(True, linestyle='--', alpha=0.7)

        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        self._save_plot(filename)

    def generate_competitive_rank_progression(self, competitive_data: List[Dict], filename: str = 'competitive_rank_progression.png'):
        """Generate multi-team rank progression line plot with inverted y-axis and hit markers.

        Args:
            competitive_data: List of dicts with team_info and gw_history.
            filename: Output filename for the plot.
        """
        if not competitive_data:
            return

        plt.figure(figsize=(14, 8))

        for idx, entry in enumerate(competitive_data):
            team_info = entry.get('team_info', {})
            gw_history = entry.get('gw_history', [])
            chips_used = entry.get('chips_used', [])

            if not gw_history:
                continue

            team_name = team_info.get('team_name', f"Team {entry.get('entry_id', idx)}")

            # Sort by event/gameweek
            sorted_history = sorted(gw_history, key=lambda x: x.get('event', 0))

            gameweeks = [gw.get('event', 0) for gw in sorted_history]
            ranks = [gw.get('overall_rank', 0) for gw in sorted_history]

            color = self.team_colors[idx % len(self.team_colors)]
            plt.plot(gameweeks, ranks, marker='o', linewidth=2,
                     markersize=6, label=team_name, color=color)
            
            # Add chip markers
            chip_map = {c.get('event'): c.get('name') for c in chips_used if c.get('event')}
            for gw_entry in sorted_history:
                gw = gw_entry.get('event', 0)
                rank = gw_entry.get('overall_rank', 0)
                
                # Mark chips with special symbols
                if gw in chip_map:
                    chip_name = chip_map[gw]
                    marker, abbr = self.chip_symbols.get(chip_name, ('o', chip_name[:2].upper()))
                    plt.plot(gw, rank, marker=marker, markersize=12, 
                            color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                
                # Mark hits with X symbol
                hits = gw_entry.get('event_transfers_cost', 0)
                if hits > 0:
                    plt.plot(gw, rank, marker='x', markersize=10,
                            color='red', markeredgecolor='darkred', markeredgewidth=2, zorder=4)

        plt.title('Rank Progression Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Gameweek', fontsize=12)
        plt.ylabel('Overall Rank', fontsize=12)

        # Invert y-axis so better rank (lower number) is higher on chart
        plt.gca().invert_yaxis()

        # Team legend
        plt.legend(loc='upper right', fontsize=10)
        
        # Add chip and hit legends
        from matplotlib.lines import Line2D
        chip_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=10, label=abbr)
            for chip_name, (marker, abbr) in self.chip_symbols.items()
        ]
        # Add hit marker to legend
        hit_element = Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                            markeredgecolor='darkred', markersize=10, markeredgewidth=2, label='Hit')
        chip_legend_elements.append(hit_element)
        
        chip_legend = plt.legend(handles=chip_legend_elements, loc='lower right',
                                fontsize=9, title='Chips & Hits', framealpha=0.9)
        plt.gca().add_artist(chip_legend)
        
        # Restore team legend
        plt.legend(loc='upper right', fontsize=10)
        plt.gca().add_artist(chip_legend)
        
        plt.grid(True, linestyle='--', alpha=0.7)

        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Format y-axis with comma separators for large numbers
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ','))
        )

        self._save_plot(filename)

    def generate_competitive_treemaps(self, competitive_data: List[Dict], prefix: str = '') -> List[str]:
        """Generate player contribution treemaps for each competitor.
        
        Creates one treemap per team showing how each player contributed
        to their total points. Only counts points from gameweeks where the
        player was in the starting XI (excluding benched players).
        Applies captain multiplier.

        Args:
            competitive_data: List of dicts with team_info and season_history.
            prefix: Optional prefix for filenames (e.g., 'global_' for global analysis).
            
        Returns:
            List of generated filenames (one per team).
        """
        if not competitive_data:
            return []

        generated_files = []
        
        # Position colors (consistent with main treemap)
        pos_colors = {
            'GKP': '#ff9999',  # Red-ish
            'DEF': '#66b3ff',  # Blue-ish
            'MID': '#99ff99',  # Green-ish
            'FWD': '#ffcc99'   # Orange-ish
        }

        for entry in competitive_data:
            team_info = entry.get('team_info', {})
            season_history = entry.get('season_history', [])
            entry_id = entry.get('entry_id', 0)
            
            if not season_history:
                continue
            
            team_name = team_info.get('team_name', f'Team {entry_id}')
            
            # Determine GW range from season_history
            gameweeks = [gw.get('gameweek', 0) for gw in season_history]
            min_gw = min(gameweeks) if gameweeks else 1
            max_gw = max(gameweeks) if gameweeks else 1
            
            # Calculate contributing points (only when in starting XI)
            player_contributions = self._calculate_contributing_points(season_history)
            
            if not player_contributions:
                continue
            
            # Prepare data for treemap
            data = []
            for name, contrib_data in player_contributions.items():
                data.append({
                    'name': name,
                    'points': contrib_data['points'],
                    'position': contrib_data['position']
                })
            
            df = pd.DataFrame(data)
            df = df[df['points'] > 0]  # Filter 0 points
            
            if df.empty:
                continue
                
            df = df.sort_values('points', ascending=False)
            
            # Assign colors based on position
            colors = [pos_colors.get(r['position'], '#cccccc') for _, r in df.iterrows()]
            
            plt.figure(figsize=(12, 7))
            
            # Labels with name and points
            labels = [f"{r['name']}\n({r['points']})" for _, r in df.iterrows()]
            
            squarify.plot(sizes=df['points'], label=labels, color=colors, alpha=0.8,
                          text_kwargs={'fontsize': 9, 'wrap': True})
            
            # Truncate team name if too long
            display_name = team_name[:25] + '...' if len(team_name) > 25 else team_name
            # Calculate total contributing points from the data
            total_contributing_pts = df['points'].sum()
            plt.title(f'{display_name} (GW{min_gw}-GW{max_gw})\nTotal: {total_contributing_pts} pts', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Add legend for positions
            patches = [mpatches.Patch(color=c, label=p) for p, c in pos_colors.items()]
            plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.1, 1))
            
            # Save with unique filename
            filename = f'{prefix}treemap_team_{entry_id}.png'
            self._save_plot(filename)
            generated_files.append(filename)
        
        return generated_files

    def generate_hindsight_fixture_analysis(self, season_history: List[Dict], 
                                           fixtures_df, teams_data: List[Dict],
                                           start_gw: int = None, end_gw: int = None):
        """Generate hindsight analysis showing fixture difficulty vs actual points.
        
        Creates a dual heatmap showing:
        1. Fixture difficulty for each player per gameweek
        2. Actual points scored by each player per gameweek
        
        Args:
            season_history: Full season history with squad data per GW.
            fixtures_df: DataFrame of fixtures with difficulty ratings.
            teams_data: List of team dictionaries from bootstrap data.
            start_gw: Starting gameweek for analysis (default: earliest available).
            end_gw: Ending gameweek for analysis (default: latest available).
        """
        if not season_history:
            return
        
        # Extract gameweek range
        gameweeks = [gw.get('gameweek', 0) for gw in season_history]
        min_gw = start_gw or min(gameweeks)
        max_gw = end_gw or max(gameweeks)
        
        # Filter to requested range
        history_subset = [gw for gw in season_history 
                         if min_gw <= gw.get('gameweek', 0) <= max_gw]
        
        if not history_subset:
            return
        
        # Build team name lookup
        team_names = {t['id']: t['short_name'] for t in teams_data}
        
        # Collect all unique players who played in starting XI
        player_data = {}  # {player_name: {'position': str, 'gws': {gw: {'difficulty': int, 'points': int, 'opponent': str}}}}
        
        for gw_entry in history_subset:
            gw = gw_entry.get('gameweek', 0)
            squad = gw_entry.get('squad', [])
            
            for player in squad:
                # Only include players in starting XI
                if player.get('position_in_squad', 99) > 11:
                    continue
                
                name = player.get('name', 'Unknown')
                position = player.get('position', 'UNK')
                
                # Get team_id from player data
                team_id = player.get('team_id', 0)
                
                base_points = player.get('stats', {}).get('event_points', 0) or 0
                multiplier = player.get('multiplier', 1)
                points = base_points * multiplier  # Apply captain multiplier
                
                # Get fixture difficulty for this GW
                difficulty = 3  # Default
                opponent = '?'
                
                if team_id:
                    # Find fixture for this team and gameweek
                    # Check home games
                    home_fix = fixtures_df[(fixtures_df['event'] == gw) & 
                                          (fixtures_df['team_h'] == team_id)]
                    if not home_fix.empty:
                        row = home_fix.iloc[0]
                        difficulty = int(row['team_h_difficulty'])
                        opponent_id = int(row['team_a'])
                        opponent = team_names.get(opponent_id, '?') + '(H)'
                    else:
                        # Check away games
                        away_fix = fixtures_df[(fixtures_df['event'] == gw) & 
                                              (fixtures_df['team_a'] == team_id)]
                        if not away_fix.empty:
                            row = away_fix.iloc[0]
                            difficulty = int(row['team_a_difficulty'])
                            opponent_id = int(row['team_h'])
                            opponent = team_names.get(opponent_id, '?') + '(A)'
                
                # Initialize player if not seen before
                if name not in player_data:
                    player_data[name] = {
                        'position': position,
                        'gws': {}
                    }
                
                player_data[name]['gws'][gw] = {
                    'difficulty': difficulty,
                    'points': points,
                    'opponent': opponent
                }
        
        # Filter to players with at least 3 starts in the range
        active_players = {name: data for name, data in player_data.items() 
                         if len(data['gws']) >= 3}
        
        if not active_players:
            return
        
        # Sort players by position and total points
        player_list = []
        for name, data in active_players.items():
            total_pts = sum(gw_data['points'] for gw_data in data['gws'].values())
            player_list.append((name, data['position'], total_pts, data['gws']))
        
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        player_list.sort(key=lambda x: (position_order.get(x[1], 4), -x[2]))
        
        # Limit to top 15 for readability
        player_list = player_list[:15]
        
        gw_range = list(range(min_gw, max_gw + 1))
        
        # Create subplots: one for difficulty, one for points
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(player_list) * 0.6)))
        
        # Prepare matrices
        difficulty_matrix = []
        points_matrix = []
        player_labels = []
        
        for name, pos, total_pts, gw_data in player_list:
            difficulty_row = []
            points_row = []
            
            for gw in gw_range:
                if gw in gw_data:
                    difficulty_row.append(gw_data[gw]['difficulty'])
                    points_row.append(gw_data[gw]['points'])
                else:
                    difficulty_row.append(0)  # Not owned/not in XI
                    points_row.append(0)
            
            difficulty_matrix.append(difficulty_row)
            points_matrix.append(points_row)
            player_labels.append(f"{name} ({pos})")
        
        # Plot 1: Fixture Difficulty
        difficulty_matrix = np.array(difficulty_matrix)
        
        # Custom colormap for FPL difficulty scale (0=not in XI, 2=easy, 3=medium, 4=hard, 5=very hard)
        # FPL uses difficulty 2-5, no difficulty 1 exists
        from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
        
        # Define colors for each difficulty level
        # Index 0: White (not in XI/no data)
        # Index 1: Not used (FPL doesn't have difficulty 1)
        # Index 2: Green (Easy)
        # Index 3: Yellow (Medium)
        # Index 4: Orange/Red (Hard)
        # Index 5: Red/Purple (Very Hard)
        colors_diff = ['#FFFFFF', '#CCCCCC', '#00ff87', '#FFD700', '#FF6B35', '#DC143C']
        cmap_diff = ListedColormap(colors_diff)
        
        # Define boundaries: [0, 1), [1, 2), [2, 3), [3, 4), [4, 5), [5, 6)
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = BoundaryNorm(bounds, cmap_diff.N)
        
        im1 = ax1.imshow(difficulty_matrix, cmap=cmap_diff, norm=norm, aspect='auto',
                        interpolation='nearest')
        
        # Annotate with opponent names
        for i in range(len(player_list)):
            name, pos, total_pts, gw_data = player_list[i]
            for j, gw in enumerate(gw_range):
                if gw in gw_data:
                    opponent = gw_data[gw]['opponent']
                    diff = gw_data[gw]['difficulty']
                    # Choose text color based on difficulty
                    text_color = 'white' if diff >= 4 else 'black'
                    ax1.text(j, i, opponent, ha='center', va='center', 
                            fontsize=7, color=text_color, weight='bold')
        
        ax1.set_title('Fixture Difficulty (with opponents)', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Gameweek', fontsize=11)
        ax1.set_yticks(range(len(player_labels)))
        ax1.set_yticklabels(player_labels, fontsize=9)
        ax1.set_xticks(range(len(gw_range)))
        ax1.set_xticklabels(gw_range, fontsize=9)
        
        # Add colorbar for difficulty
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Difficulty', fontsize=10)
        cbar1.set_ticks([0.5, 2.5, 3.5, 4.5, 5.5])
        cbar1.set_ticklabels(['Not in XI', 'Easy (2)', 'Medium (3)', 'Hard (4)', 'Very Hard (5)'])
        
        # Plot 2: Actual Points
        points_matrix = np.array(points_matrix)
        
        # Custom colormap for points: white (0) -> light green -> green (high)
        colors_pts = ['#FFFFFF', '#E8F5E9', '#66BB6A', '#2E7D32', '#1B5E20']
        cmap_pts = LinearSegmentedColormap.from_list('points', colors_pts, N=100)
        
        max_pts = max(10, points_matrix.max())
        im2 = ax2.imshow(points_matrix, cmap=cmap_pts, aspect='auto',
                        vmin=0, vmax=max_pts, interpolation='nearest')
        
        # Annotate with point values
        for i in range(len(player_list)):
            name, pos, total_pts, gw_data = player_list[i]
            for j, gw in enumerate(gw_range):
                if gw in gw_data:
                    pts = gw_data[gw]['points']
                    if pts > 0:
                        text_color = 'white' if pts >= max_pts * 0.6 else 'black'
                        ax2.text(j, i, str(int(pts)), ha='center', va='center',
                                fontsize=9, color=text_color, weight='bold')
        
        ax2.set_title('Actual Points Scored', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xlabel('Gameweek', fontsize=11)
        ax2.set_yticks(range(len(player_labels)))
        ax2.set_yticklabels([''] * len(player_labels))  # No labels on right side
        ax2.set_xticks(range(len(gw_range)))
        ax2.set_xticklabels(gw_range, fontsize=9)
        
        # Add colorbar for points
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Points', fontsize=10)
        
        plt.suptitle(f'Hindsight Fixture Analysis (GW{min_gw}-GW{max_gw})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        self._save_plot('hindsight_fixture_analysis.png')

    def generate_free_hit_gw_comparison(self, gw_data: Dict) -> str:
        """Generate a line chart comparing current squad vs Free Hit xP over gameweeks.
        
        Args:
            gw_data: Dict containing:
                - current_gw: Current gameweek number
                - gameweeks: List of GW numbers [17, 18, 19, 20, 21]
                - current_squad_xp: List of xP per GW for current squad
                - free_hit_xp: List of xP per GW for optimal Free Hit squad
                - best_gw: The GW with highest differential (best to use FH)
        
        Returns:
            Path to the saved plot.
        """
        plt.figure(figsize=(10, 6))
        
        gameweeks = gw_data.get('gameweeks', [])
        current_xp = gw_data.get('current_squad_xp', [])
        fh_xp = gw_data.get('free_hit_xp', [])
        best_gw = gw_data.get('best_gw', None)
        current_gw = gw_data.get('current_gw', gameweeks[0] - 1 if gameweeks else 16)
        
        if not gameweeks or not current_xp or not fh_xp:
            plt.close()
            return None
        
        # Create the plot
        ax = plt.gca()
        
        # Plot lines
        ax.plot(gameweeks, current_xp, 'o-', color=self.colors['purple'], 
                linewidth=2.5, markersize=10, label='Current Squad', zorder=3)
        ax.plot(gameweeks, fh_xp, 's-', color=self.colors['green'], 
                linewidth=2.5, markersize=10, label='Free Hit Squad', zorder=3)
        
        # Fill between to show the gain area
        ax.fill_between(gameweeks, current_xp, fh_xp, 
                        where=[fh > curr for fh, curr in zip(fh_xp, current_xp)],
                        alpha=0.3, color=self.colors['green'], label='Potential Gain')
        
        # Highlight the best GW
        if best_gw and best_gw in gameweeks:
            idx = gameweeks.index(best_gw)
            best_gain = fh_xp[idx] - current_xp[idx]
            ax.axvline(x=best_gw, color=self.colors['pink'], linestyle='--', 
                      linewidth=2, alpha=0.7, zorder=2)
            ax.annotate(f'Best GW: +{best_gain:.1f}', 
                       xy=(best_gw, fh_xp[idx]), 
                       xytext=(best_gw + 0.3, fh_xp[idx] + 3),
                       fontsize=11, fontweight='bold', color=self.colors['pink'],
                       arrowprops=dict(arrowstyle='->', color=self.colors['pink']))
        
        # Add value labels
        for i, gw in enumerate(gameweeks):
            # Current squad label
            ax.annotate(f'{current_xp[i]:.1f}', 
                       xy=(gw, current_xp[i]), 
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', fontsize=9, color=self.colors['purple'])
            # Free Hit label
            ax.annotate(f'{fh_xp[i]:.1f}', 
                       xy=(gw, fh_xp[i]), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='#006644', fontweight='bold')
        
        # Styling
        ax.set_xlabel('Gameweek', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Points (Model)', fontsize=12, fontweight='bold')
        ax.set_title('Free Hit Analysis: When to Play Your Chip', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.text(0.5, 1.02, '(Uniform ML predictor across all gameweeks)', 
               transform=ax.transAxes, fontsize=9, ha='center', 
               color='gray', style='italic')
        
        ax.set_xticks(gameweeks)
        ax.set_xticklabels([f'GW{gw}' for gw in gameweeks], fontsize=10)
        
        # Set y-axis limits with padding
        all_values = current_xp + fh_xp
        y_min = min(all_values) - 5
        y_max = max(all_values) + 8
        ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add summary text box
        total_current = sum(current_xp)
        total_fh = sum(fh_xp)
        total_gain = total_fh - total_current
        
        summary_text = f'5-GW Totals:\nCurrent: {total_current:.1f}\nFree Hit: {total_fh:.1f}\nMax Gain: +{max(f - c for f, c in zip(fh_xp, current_xp)):.1f}'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'free_hit_gw_analysis.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)

