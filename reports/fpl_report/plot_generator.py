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
import json
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
            'green': '#2ecc71', 
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
            '#37003c', '#2ecc71', '#ff2f82', '#0057ff', '#ff9500',
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

    # Unified position color scheme used throughout the report
    POSITION_COLORS = {
        'GKP': '#F8B4B4',  # Soft coral/pink - Goalkeepers
        'DEF': '#93C5FD',  # Soft blue - Defenders
        'MID': '#86EFAC',  # Soft green - Midfielders
        'FWD': '#FDBA74'   # Soft orange/peach - Forwards
    }

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
        
        # Assign colors based on position using unified scheme
        colors = [self.POSITION_COLORS.get(r['position'], '#E5E7EB') for _, r in df.iterrows()]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Labels with name and points
        labels = [f"{r['name']}\n({r['points']})" for _, r in df.iterrows()]
        
        # Plot with thin white borders for clean separation
        squarify.plot(sizes=df['points'], label=labels, color=colors, alpha=0.9,
                      ax=ax,
                      edgecolor='white', linewidth=0.8,
                      text_kwargs={'fontsize': 10, 'fontweight': 'medium', 'color': '#1F2937'})
        
        ax.axis('off')
        
        # Add legend outside the plot area
        patches = [mpatches.Patch(facecolor=c, label=p, edgecolor='#D1D5DB', linewidth=0.5) 
                   for p, c in self.POSITION_COLORS.items()]
        legend = ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                          frameon=True, fancybox=False, shadow=False,
                          edgecolor='#D1D5DB', facecolor='white',
                          fontsize=9, title='Position', title_fontsize=10)
        legend.get_title().set_fontweight('bold')
        
        plt.subplots_adjust(right=0.88)
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
            
            # Assign colors based on position using unified scheme
            colors = [self.POSITION_COLORS.get(r['position'], '#E5E7EB') for _, r in df.iterrows()]
            
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Labels with name and points
            labels = [f"{r['name']}\n({r['points']})" for _, r in df.iterrows()]
            
            # Plot with thin white borders for clean separation
            squarify.plot(sizes=df['points'], label=labels, color=colors, alpha=0.9,
                          ax=ax,
                          edgecolor='white', linewidth=0.8,
                          text_kwargs={'fontsize': 9, 'fontweight': 'medium', 'color': '#1F2937'})
            
            ax.axis('off')
            
            # Add legend outside the plot area
            patches = [mpatches.Patch(facecolor=c, label=p, edgecolor='#D1D5DB', linewidth=0.5) 
                       for p, c in self.POSITION_COLORS.items()]
            legend = ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                              frameon=True, fancybox=False, shadow=False,
                              edgecolor='#D1D5DB', facecolor='white',
                              fontsize=9, title='Position', title_fontsize=10)
            legend.get_title().set_fontweight('bold')
            
            plt.subplots_adjust(right=0.86)
            
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
        colors_diff = ['#FFFFFF', '#CCCCCC', '#2ecc71', '#FFD700', '#FF6B35', '#DC143C']
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
        """Generate a single-GW bar comparison: Current Squad vs Free Hit Squad.
        
        Free Hit is optimized for ONE specific gameweek, so comparing across
        multiple GWs doesn't make sense (squad would be different for each).
        
        Args:
            gw_data: Dict containing:
                - target_gw: The gameweek Free Hit is optimized for
                - current_squad_xp: xP for current squad in target GW
                - free_hit_xp: xP for Free Hit squad in target GW
        
        Returns:
            Path to the saved plot.
        """
        target_gw = gw_data.get('target_gw', 18)
        current_xp = gw_data.get('current_squad_xp', 0)
        fh_xp = gw_data.get('free_hit_xp', 0)
        
        if current_xp == 0 and fh_xp == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Data for bars
        categories = ['Current Squad', 'Free Hit Squad']
        values = [current_xp, fh_xp]
        colors_list = [self.colors['purple'], self.colors['green']]
        
        # Create horizontal bar chart
        bars = ax.barh(categories, values, color=colors_list, height=0.5, edgecolor='white', linewidth=2)
        
        # Add value labels on the bars
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.annotate(f'{val:.1f} xP',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(8, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=14, fontweight='bold',
                       color=bar.get_facecolor())
        
        # Calculate and display the gain
        gain = fh_xp - current_xp
        gain_color = self.colors['green'] if gain > 0 else self.colors['pink']
        gain_sign = '+' if gain > 0 else ''
        
        # Add gain annotation
        ax.annotate(f'{gain_sign}{gain:.1f} pts',
                   xy=(max(values) * 0.95, 0.5),
                   fontsize=18, fontweight='bold', color=gain_color,
                   ha='right', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=gain_color, linewidth=2, alpha=0.9))
        
        # Styling
        ax.set_xlabel('Predicted Points (xP)', fontsize=12, fontweight='bold')
        ax.set_title(f'Free Hit Analysis: GW{target_gw}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.text(0.5, 1.02, f'Squad optimized specifically for Gameweek {target_gw} fixtures', 
               transform=ax.transAxes, fontsize=9, ha='center', 
               color='gray', style='italic')
        
        # Set x-axis limits with padding
        ax.set_xlim(0, max(values) * 1.25)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add gridlines
        ax.xaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Style y-axis labels
        ax.tick_params(axis='y', labelsize=12)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'free_hit_gw_analysis.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(filepath)

    def _aggregate_fplcore_player_stats(self, fpl_core_season_data: Dict, 
                                       fpl_core_gw_data: Dict, 
                                       current_gw: int,
                                       min_minutes: int = 450,
                                       squad_ids: List[int] = None) -> Dict:
        """Aggregate FPL Core Insights data for player performance analysis.
        
        Processes playermatchstats and player_gameweek_stats to compute:
        - Season cumulative stats (Goals, Assists, xG, xA, Shots, Box Touches per 90)
        - Single GW stats for current gameweek
        - Filters to players with minimum minutes threshold
        
        Args:
            fpl_core_season_data: Season-level FPL Core data dict
            fpl_core_gw_data: Current gameweek FPL Core data dict
            current_gw: Current gameweek number
            min_minutes: Minimum minutes played for inclusion (default: 450)
            squad_ids: Optional list of player IDs to filter to (for squad-only plots)
            
        Returns:
            Dict with 'season' and 'gameweek' keys containing player stats DataFrames
        """
        import pandas as pd
        
        result = {'season': None, 'gameweek': None}
        
        # Process season-level stats
        if fpl_core_season_data and 'playerstats' in fpl_core_season_data:
            season_df = fpl_core_season_data['playerstats'].copy()
            
            if not season_df.empty:
                # Filter to only include data up to current_gw (no future GW leakage)
                if 'gw' in season_df.columns:
                    season_df = season_df[season_df['gw'] <= current_gw]
                # Deduplicate - keep only the latest (most recent) row for each player
                # playerstats has historical snapshots, we want current season cumulative
                season_df = season_df.sort_values('gw', ascending=False).drop_duplicates(subset=['id'], keep='first').copy()
                
                # Merge position information from players dataset
                if 'players' in fpl_core_season_data:
                    players_info = fpl_core_season_data['players'][['player_id', 'position']].copy()
                    season_df = season_df.merge(players_info, left_on='id', right_on='player_id', how='left')
                
                # Filter to squad players first if specified (for squad-only plots)
                if squad_ids is not None:
                    season_df = season_df[season_df['id'].isin(squad_ids)].copy()
                
                # Filter to players with sufficient minutes
                season_df = season_df[season_df['minutes'] >= min_minutes].copy()
                
                # Calculate per-90 stats
                season_df['minutes_90'] = season_df['minutes'] / 90.0
                season_df['goals'] = season_df['goals_scored'].fillna(0)
                season_df['xG'] = season_df['expected_goals'].fillna(0)
                season_df['xA'] = season_df['expected_assists'].fillna(0)
                season_df['xGI'] = season_df['xG'] + season_df['xA']
                season_df['goal_diff'] = season_df['goals'] - season_df['xG']
                season_df['assist_diff'] = season_df['assists'] - season_df['xA']
                season_df['total_output'] = season_df['goals'] + season_df['assists']
                
                # Keep relevant columns (include position for goalkeeper filtering)
                result['season'] = season_df[[
                    'id', 'web_name', 'first_name', 'second_name', 'position',
                    'minutes', 'minutes_90', 'goals', 'assists', 
                    'xG', 'xA', 'xGI', 'goal_diff', 'assist_diff', 'total_output'
                ]].copy()
        
        # Process gameweek-level stats  
        if fpl_core_gw_data and 'player_gameweek_stats' in fpl_core_gw_data:
            gw_df = fpl_core_gw_data['player_gameweek_stats']
            
            if not gw_df.empty:
                # Filter to current GW
                gw_df = gw_df[gw_df['gw'] == current_gw].copy()
                
                # Merge position information from players dataset
                if 'players' in fpl_core_gw_data:
                    players_info = fpl_core_gw_data['players'][['player_id', 'position']].copy()
                    gw_df = gw_df.merge(players_info, left_on='id', right_on='player_id', how='left')
                elif fpl_core_season_data and 'players' in fpl_core_season_data:
                    players_info = fpl_core_season_data['players'][['player_id', 'position']].copy()
                    gw_df = gw_df.merge(players_info, left_on='id', right_on='player_id', how='left')
                
                # Filter to squad players first if specified (for squad-only plots)
                if squad_ids is not None:
                    gw_df = gw_df[gw_df['id'].isin(squad_ids)].copy()
                
                # Calculate stats
                gw_df['goals'] = gw_df['goals_scored'].fillna(0)
                gw_df['xG'] = gw_df['expected_goals'].fillna(0)
                gw_df['xA'] = gw_df['expected_assists'].fillna(0)
                gw_df['xGI'] = gw_df['xG'] + gw_df['xA']
                gw_df['goal_diff'] = gw_df['goals'] - gw_df['xG']
                gw_df['assist_diff'] = gw_df['assists'] - gw_df['xA']
                gw_df['total_output'] = gw_df['goals'] + gw_df['assists']
                
                # Keep relevant columns (include position for goalkeeper filtering)
                result['gameweek'] = gw_df[[
                    'id', 'web_name', 'first_name', 'second_name', 'position',
                    'minutes', 'goals', 'assists', 
                    'xG', 'xA', 'xGI', 'goal_diff', 'assist_diff', 'total_output'
                ]].copy()
        
        return result

    def _select_gameweeks(self,
                          all_gw_data: Dict[int, Dict],
                          start_gw: int = None,
                          end_gw: int = None,
                          last_n_gw: int = None) -> List[int]:
        """Return sorted list of gameweeks respecting the requested window."""
        if not all_gw_data:
            return []

        gws = []
        for gw, gw_data in all_gw_data.items():
            gw_int = int(gw)
            if isinstance(gw_data, dict):
                pms = gw_data.get('playermatchstats')
                if pms is None:
                    continue
                if hasattr(pms, 'empty') and pms.empty:
                    continue
            gws.append(gw_int)

        gws = sorted(gws)

        if start_gw is not None:
            gws = [gw for gw in gws if gw >= start_gw]
        if end_gw is not None:
            gws = [gw for gw in gws if gw <= end_gw]
        if last_n_gw is not None and last_n_gw > 0:
            gws = gws[-last_n_gw:]

        return gws

    def _format_gw_range_label(self, gws: List[int]) -> str:
        """Format a friendly GW range label such as GW12-17."""
        if not gws:
            return "No GW data"

        start, end = min(gws), max(gws)
        return f"GW{start}" if start == end else f"GW{start}-{end}"

    def _write_usage_summary(self,
                             summary_key: str,
                             range_label: str,
                             df: pd.DataFrame,
                             out_path: Path,
                             x_median: float = None,
                             y_median: float = None) -> None:
        """Persist summary tables and quick insights for Usage vs Output."""
        if df.empty:
            return

        pos_map = {
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'FWD',
            'Goalkeeper': 'GKP'
        }

        subset = df.copy()
        subset = subset.sort_values(['total_output', 'usage_per_90', 'xGI'], ascending=False)

        rows = []
        for _, row in subset.head(12).iterrows():
            rows.append({
                'name': row.get('web_name', 'Unknown'),
                'pos': pos_map.get(row.get('position'), row.get('position', 'UNK')),
                'usage_per_90': round(float(row.get('usage_per_90', 0)), 2),
                'ga': int(row.get('total_output', 0)),
                'xgi': round(float(row.get('xGI', 0)), 2),
                'pts': int(row.get('total_points', 0))
            })

        def _top_by(col):
            if subset.empty or col not in subset.columns:
                return None
            return subset.loc[subset[col].idxmax()]

        top_use = _top_by('usage_per_90')
        top_ga = _top_by('total_output')
        top_xgi = _top_by('xGI')

        insights = []
        if top_use is not None:
            insights.append(f"Focal point: {top_use.get('web_name', 'Unknown')} leads usage at {top_use.get('usage_per_90', 0):.1f}/90 ({range_label}).")
        if top_ga is not None:
            insights.append(f"End product: {top_ga.get('web_name', 'Unknown')} tops G+A with {int(top_ga.get('total_output', 0))} ({range_label}).")
        if top_xgi is not None:
            insights.append(f"Underlying strength: {top_xgi.get('web_name', 'Unknown')} leads xGI at {top_xgi.get('xGI', 0):.2f} ({range_label}).")

        categories = {}
        if x_median is not None and y_median is not None:
            def add_cat(key, frame):
                if frame.empty:
                    return
                categories[key] = [{
                    'name': r.get('web_name', 'Unknown'),
                    'pos': pos_map.get(r.get('position'), r.get('position', 'UNK')),
                    'usage_per_90': round(float(r.get('usage_per_90', 0)), 2),
                    'ga': int(r.get('total_output', 0)),
                    'xgi': round(float(r.get('xGI', 0)), 2),
                    'pts': int(r.get('total_points', 0))
                } for _, r in frame.head(8).iterrows()]

            elite = subset[(subset['usage_per_90'] >= x_median) & (subset['total_output'] >= y_median)]
            volume = subset[(subset['usage_per_90'] >= x_median) & (subset['total_output'] < y_median)]
            clinical = subset[(subset['usage_per_90'] < x_median) & (subset['total_output'] >= y_median)]
            avoid = subset[(subset['usage_per_90'] < x_median) & (subset['total_output'] < y_median)]

            add_cat('elite', elite)
            add_cat('volume', volume)
            add_cat('clinical', clinical)
            add_cat('avoid', avoid)

        payload = {
            'range_label': range_label,
            'rows': rows,
            'insights': insights,
            'categories': categories
        }

        existing = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except Exception:
                existing = {}

        existing[summary_key] = payload
        out_path.write_text(json.dumps(existing, indent=2))

    def _write_defensive_summary(self,
                                 summary_key: str,
                                 range_label: str,
                                 df: pd.DataFrame,
                                 out_path: Path,
                                 x_median: float = None,
                                 y_median: float = None) -> None:
        """Persist summary tables and quick insights for Defensive Value charts."""
        if df.empty:
            return

        subset = df.copy()
        subset = subset.sort_values(['total_points', 'def_actions_per_90'], ascending=False)

        # Top defenders table
        rows = []
        for _, row in subset.head(12).iterrows():
            rows.append({
                'name': row.get('web_name', 'Unknown'),
                'team': '',  # Could add team if available
                'price': round(float(row.get('price', 5.0)), 1),
                'def_per_90': round(float(row.get('def_actions_per_90', 0)), 1),
                'tackles': int(row.get('tackles_won', 0)),
                'interceptions': int(row.get('interceptions', 0)),
                'clearances': int(row.get('clearances', 0)),
                'blocks': int(row.get('blocks', 0)),
                'cs': int(row.get('clean_sheets', 0)),
                'cs_pct': round(float(row.get('cs_pct', 0)), 1),
                'pts': int(row.get('total_points', 0)),
                'pts_per_m': round(float(row.get('pts_per_m', 0)), 2)
            })

        # Insights
        def _top_by(col):
            if subset.empty or col not in subset.columns:
                return None
            return subset.loc[subset[col].idxmax()]

        top_def = _top_by('def_actions_per_90')
        top_pts = _top_by('total_points')
        top_cs = _top_by('clean_sheets')

        insights = []
        if top_def is not None:
            insights.append(f"Defensive anchor: {top_def.get('web_name', 'Unknown')} leads defensive actions at {top_def.get('def_actions_per_90', 0):.1f}/90 ({range_label}).")
        if top_pts is not None:
            insights.append(f"Top scorer: {top_pts.get('web_name', 'Unknown')} tops points with {int(top_pts.get('total_points', 0))} ({range_label}).")
        if top_cs is not None:
            insights.append(f"Clean sheet king: {top_cs.get('web_name', 'Unknown')} has {int(top_cs.get('clean_sheets', 0))} clean sheets ({range_label}).")

        # Categories based on quadrants
        categories = {}
        if x_median is not None and y_median is not None:
            def add_cat(key, frame):
                if frame.empty:
                    return
                categories[key] = [{
                    'name': r.get('web_name', 'Unknown'),
                    'def_per_90': round(float(r.get('def_actions_per_90', 0)), 1),
                    'cs': int(r.get('clean_sheets', 0)),
                    'cs_pct': round(float(r.get('cs_pct', 0)), 1),
                    'pts': int(r.get('total_points', 0))
                } for _, r in frame.head(8).iterrows()]

            # ELITE: High defensive actions AND high points
            elite = subset[(subset['def_actions_per_90'] >= x_median) & (subset['total_points'] >= y_median)]
            # VOLUME: High defensive actions but low points
            volume = subset[(subset['def_actions_per_90'] >= x_median) & (subset['total_points'] < y_median)]
            # CS MERCHANTS: Low defensive actions but high points
            cs_merchants = subset[(subset['def_actions_per_90'] < x_median) & (subset['total_points'] >= y_median)]
            # AVOID: Low defensive actions and low points
            avoid = subset[(subset['def_actions_per_90'] < x_median) & (subset['total_points'] < y_median)]

            add_cat('elite', elite)
            add_cat('volume', volume)
            add_cat('cs_merchants', cs_merchants)
            add_cat('avoid', avoid)

        payload = {
            'range_label': range_label,
            'rows': rows,
            'insights': insights,
            'categories': categories
        }

        existing = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except Exception:
                existing = {}

        existing[summary_key] = payload
        out_path.write_text(json.dumps(existing, indent=2))

    def _aggregate_fplcore_usage_stats(self, all_gw_data: Dict,
                                      fpl_core_season_data: Dict,
                                      min_minutes: int = 450,
                                      start_gw: int = None,
                                      end_gw: int = None,
                                      last_n_gw: int = None) -> pd.DataFrame:
        """Aggregate usage statistics from FPL Core playermatchstats across all gameweeks.
        
        Computes shots + box touches per 90 for usage analysis.
        
        Args:
            all_gw_data: Dict of gameweek data {gw_num: {playermatchstats: df, ...}}
            fpl_core_season_data: Season-level FPL Core data dict (for player info)
            min_minutes: Minimum minutes for inclusion
            start_gw: Optional start gameweek (inclusive)
            end_gw: Optional end gameweek (inclusive)
            last_n_gw: If provided, limit to the most recent N gameweeks
            
        Returns:
            DataFrame with player usage statistics
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(
            all_gw_data,
            start_gw=start_gw,
            end_gw=end_gw,
            last_n_gw=last_n_gw
        )

        if not selected_gws:
            return pd.DataFrame()
        
        # Collect playermatchstats from selected gameweeks
        all_match_stats = []
        all_gw_points = []
        for gw_num in selected_gws:
            gw_data = all_gw_data.get(gw_num, {})
            if 'playermatchstats' in gw_data and gw_data['playermatchstats'] is not None and not gw_data['playermatchstats'].empty:
                match_df = gw_data['playermatchstats'].copy()
                match_df['gw'] = gw_num
                all_match_stats.append(match_df)
            
            # Collect event_points from player_gameweek_stats
            if 'player_gameweek_stats' in gw_data and gw_data['player_gameweek_stats'] is not None and not gw_data['player_gameweek_stats'].empty:
                gw_stats = gw_data['player_gameweek_stats'][['id', 'event_points']].copy()
                gw_stats = gw_stats.rename(columns={'id': 'player_id'})
                gw_stats['event_points'] = pd.to_numeric(gw_stats['event_points'], errors='coerce').fillna(0)
                all_gw_points.append(gw_stats)
        
        if not all_match_stats:
            return pd.DataFrame()
        
        # Concatenate all gameweek match stats
        match_stats = pd.concat(all_match_stats, ignore_index=True)
        
        # Aggregate by player
        player_agg = match_stats.groupby('player_id').agg({
            'minutes_played': 'sum',
            'goals': 'sum',
            'assists': 'sum',
            'total_shots': 'sum',
            'touches_opposition_box': 'sum',
            'xg': 'sum',
            'xa': 'sum'
        }).reset_index()
        
        # Aggregate FPL points for the window
        if all_gw_points:
            points_df = pd.concat(all_gw_points, ignore_index=True)
            points_agg = points_df.groupby('player_id')['event_points'].sum().reset_index()
            points_agg = points_agg.rename(columns={'event_points': 'total_points'})
            player_agg = player_agg.merge(points_agg, on='player_id', how='left')
            player_agg['total_points'] = player_agg['total_points'].fillna(0).astype(int)
        else:
            player_agg['total_points'] = 0
        
        # Filter by minimum minutes
        player_agg = player_agg[player_agg['minutes_played'] >= min_minutes].copy()
        
        # Calculate per-90 stats
        player_agg['minutes_90'] = player_agg['minutes_played'] / 90.0
        player_agg['shots_per_90'] = player_agg['total_shots'] / player_agg['minutes_90']
        player_agg['box_touches_per_90'] = player_agg['touches_opposition_box'] / player_agg['minutes_90']
        player_agg['usage_per_90'] = player_agg['shots_per_90'] + player_agg['box_touches_per_90']
        player_agg['total_output'] = player_agg['goals'] + player_agg['assists']
        player_agg['xGI'] = player_agg['xg'] + player_agg['xa']
        
        # Merge with player info from playerstats and position from players dataset
        # Use the latest GW data up to our analysis window (max of selected_gws)
        max_gw = max(selected_gws) if selected_gws else 17
        if fpl_core_season_data:
            if 'playerstats' in fpl_core_season_data:
                ps = fpl_core_season_data['playerstats'].copy()
                # Filter to only include data up to the analysis window (no future GW leakage)
                if 'gw' in ps.columns:
                    ps = ps[ps['gw'] <= max_gw]
                    # Sort by gw descending to get the latest GW data for each player
                    ps = ps.sort_values('gw', ascending=False)
                player_info = ps[['id', 'web_name', 'first_name', 'second_name']].drop_duplicates(subset=['id'], keep='first')
                player_agg = player_agg.merge(player_info, left_on='player_id', right_on='id', how='left')
            
            # Merge position information from players dataset
            if 'players' in fpl_core_season_data:
                players_pos = fpl_core_season_data['players'][['player_id', 'position']].copy()
                player_agg = player_agg.merge(players_pos, on='player_id', how='left')
        
        return player_agg

    def _aggregate_fplcore_defensive_stats(self, all_gw_data: Dict,
                                          fpl_core_season_data: Dict,
                                          min_minutes: int = 450,
                                          start_gw: int = None,
                                          end_gw: int = None,
                                          last_n_gw: int = None) -> pd.DataFrame:
        """Aggregate defensive statistics from FPL Core playermatchstats.
        
        Computes defensive actions (tackles + interceptions + clearances + blocks) per 90.
        
        Args:
            all_gw_data: Dict of gameweek data {gw_num: {playermatchstats: df, ...}}
            fpl_core_season_data: Season-level FPL Core data dict (for player info)
            min_minutes: Minimum minutes for inclusion
            start_gw: Optional start gameweek (inclusive)
            end_gw: Optional end gameweek (inclusive)
            last_n_gw: If provided, limit to the most recent N gameweeks
            
        Returns:
            DataFrame with player defensive statistics
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(
            all_gw_data,
            start_gw=start_gw,
            end_gw=end_gw,
            last_n_gw=last_n_gw
        )

        if not selected_gws:
            return pd.DataFrame()
        
        # Collect playermatchstats and player_gameweek_stats from selected gameweeks
        all_match_stats = []
        all_gw_points = []
        all_gw_cs = []
        
        for gw_num in selected_gws:
            gw_data = all_gw_data.get(gw_num, {})
            if 'playermatchstats' in gw_data and gw_data['playermatchstats'] is not None and not gw_data['playermatchstats'].empty:
                match_df = gw_data['playermatchstats'].copy()
                match_df['gw'] = gw_num
                all_match_stats.append(match_df)
            
            # Collect event_points and clean_sheets from player_gameweek_stats
            if 'player_gameweek_stats' in gw_data and gw_data['player_gameweek_stats'] is not None and not gw_data['player_gameweek_stats'].empty:
                gw_stats = gw_data['player_gameweek_stats'][['id', 'event_points', 'clean_sheets']].copy()
                gw_stats = gw_stats.rename(columns={'id': 'player_id'})
                gw_stats['event_points'] = pd.to_numeric(gw_stats['event_points'], errors='coerce').fillna(0)
                gw_stats['clean_sheets'] = pd.to_numeric(gw_stats['clean_sheets'], errors='coerce').fillna(0)
                all_gw_points.append(gw_stats)
        
        if not all_match_stats:
            return pd.DataFrame()
        
        # Concatenate all gameweek match stats
        match_stats = pd.concat(all_match_stats, ignore_index=True)
        
        # Fill missing defensive columns with 0
        defensive_cols = ['tackles_won', 'interceptions', 'clearances', 'blocks', 'tackles']
        for col in defensive_cols:
            if col not in match_stats.columns:
                match_stats[col] = 0
            else:
                match_stats[col] = pd.to_numeric(match_stats[col], errors='coerce').fillna(0)
        
        # Use 'tackles' if 'tackles_won' is mostly zero
        if match_stats['tackles_won'].sum() == 0 and match_stats['tackles'].sum() > 0:
            match_stats['tackles_won'] = match_stats['tackles']
        
        # Aggregate by player
        player_agg = match_stats.groupby('player_id').agg({
            'minutes_played': 'sum',
            'tackles_won': 'sum',
            'interceptions': 'sum',
            'clearances': 'sum',
            'blocks': 'sum',
            'goals': 'sum',
            'assists': 'sum'
        }).reset_index()
        
        # Aggregate FPL points and clean sheets for the window
        if all_gw_points:
            points_df = pd.concat(all_gw_points, ignore_index=True)
            points_agg = points_df.groupby('player_id').agg({
                'event_points': 'sum',
                'clean_sheets': 'sum'
            }).reset_index()
            points_agg = points_agg.rename(columns={'event_points': 'total_points'})
            player_agg = player_agg.merge(points_agg, on='player_id', how='left')
            player_agg['total_points'] = player_agg['total_points'].fillna(0).astype(int)
            player_agg['clean_sheets'] = player_agg['clean_sheets'].fillna(0).astype(int)
        else:
            player_agg['total_points'] = 0
            player_agg['clean_sheets'] = 0
        
        # Filter by minimum minutes
        player_agg = player_agg[player_agg['minutes_played'] >= min_minutes].copy()
        
        # Calculate per-90 stats and defensive actions
        player_agg['minutes_90'] = player_agg['minutes_played'] / 90.0
        player_agg['defensive_actions'] = (
            player_agg['tackles_won'] + 
            player_agg['interceptions'] + 
            player_agg['clearances'] + 
            player_agg['blocks']
        )
        player_agg['def_actions_per_90'] = player_agg['defensive_actions'] / player_agg['minutes_90']
        
        # Calculate games played (approx) and clean sheet percentage
        num_gws = len(selected_gws)
        player_agg['games_played'] = (player_agg['minutes_played'] / 90).round().astype(int).clip(lower=1)
        player_agg['cs_pct'] = (player_agg['clean_sheets'] / player_agg['games_played'] * 100).round(1)
        
        # Merge with player info from playerstats and position from players dataset
        # Use the latest GW data up to our analysis window (max of selected_gws)
        max_gw = max(selected_gws) if selected_gws else 17
        if fpl_core_season_data:
            if 'playerstats' in fpl_core_season_data:
                ps = fpl_core_season_data['playerstats'].copy()
                # Filter to only include data up to the analysis window (no future GW leakage)
                if 'gw' in ps.columns:
                    ps = ps[ps['gw'] <= max_gw]
                    # Sort by gw descending to get the latest GW data for each player
                    ps = ps.sort_values('gw', ascending=False)
                player_info = ps[['id', 'web_name', 'first_name', 'second_name']].drop_duplicates(subset=['id'], keep='first')
                
                # Get now_cost if available (from the same latest GW)
                if 'now_cost' in ps.columns:
                    cost_info = ps[['id', 'now_cost']].drop_duplicates(subset=['id'], keep='first')
                    player_info = player_info.merge(cost_info, on='id', how='left')
                
                player_agg = player_agg.merge(player_info, left_on='player_id', right_on='id', how='left')
            
            # Merge position information from players dataset
            if 'players' in fpl_core_season_data:
                players_pos = fpl_core_season_data['players'][['player_id', 'position']].copy()
                player_agg = player_agg.merge(players_pos, on='player_id', how='left')
        
        # Calculate price efficiency (pts per million)
        # Try to fetch live prices from FPL API, fall back to FPL Core data
        try:
            from scraping.fpl_api import get_data
            bootstrap = get_data()
            elements = bootstrap.get('elements', [])
            if elements:
                # Build price lookup from live API (now_cost is in tenths: 55 = £5.5m)
                live_prices = pd.DataFrame([
                    {'id': el['id'], 'price': el['now_cost'] / 10.0}
                    for el in elements
                ])
                player_agg = player_agg.merge(live_prices, left_on='player_id', right_on='id',
                                              how='left', suffixes=('', '_live'))
                if 'id_live' in player_agg.columns:
                    player_agg = player_agg.drop(columns=['id_live'])
        except Exception as e:
            pass  # Fall back to FPL Core prices if live API unavailable
        
        # If price wasn't set from live API, fall back to FPL Core or default
        if 'price' not in player_agg.columns:
            if 'now_cost' in player_agg.columns:
                player_agg['now_cost'] = pd.to_numeric(player_agg['now_cost'], errors='coerce').fillna(5.0)
                player_agg['price'] = player_agg['now_cost']
            else:
                player_agg['price'] = 5.0
        
        player_agg['price'] = player_agg['price'].fillna(5.0)
        player_agg['pts_per_m'] = player_agg['total_points'] / player_agg['price'].clip(lower=0.1)
        
        return player_agg

    def generate_defensive_value_scatter(self, all_gw_data: Dict,
                                        fpl_core_season_data: Dict,
                                        squad_ids: List[int],
                                        top_n: int = 25,
                                        last_n_gw: int = None,
                                        filename: str = 'defensive_value_scatter.png',
                                        title_suffix: str = None,
                                        min_minutes: int = None) -> str:
        """Generate Defensive Value bubble scatter plot for defenders.
        
        Shows relationship between defensive actions per 90 and total FPL points,
        with bubble size indicating price efficiency and color indicating CS%.
        
        Args:
            all_gw_data: Dict of all gameweek data
            fpl_core_season_data: Season-level FPL Core data
            squad_ids: List of player IDs in user's squad
            top_n: Number of top defenders by points to include
            last_n_gw: Limit to most recent N gameweeks (None = full season)
            filename: Output filename
            title_suffix: Optional label for title
            min_minutes: Optional minimum minutes filter
            
        Returns:
            Filename of saved plot
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(all_gw_data, last_n_gw=last_n_gw)
        window_len = len(selected_gws)

        if min_minutes is None:
            if last_n_gw is None:
                min_minutes = 450
            else:
                min_minutes = max(90, 45 * window_len) if window_len else 90

        def_df = self._aggregate_fplcore_defensive_stats(
            all_gw_data,
            fpl_core_season_data,
            min_minutes=min_minutes,
            last_n_gw=last_n_gw
        )
        
        if def_df.empty:
            return None
        
        # Filter to defenders only
        if 'position' in def_df.columns:
            def_df = def_df[def_df['position'] == 'Defender'].copy()
        
        if def_df.empty:
            return None
        
        # Sort by total points and get top performers
        def_df = def_df.sort_values('total_points', ascending=False)
        
        # Get top N players and always include squad defenders
        top_players = def_df.head(top_n)
        squad_players = def_df[def_df['player_id'].isin(squad_ids)]
        
        # Combine and deduplicate
        combined = pd.concat([top_players, squad_players]).drop_duplicates(subset=['player_id'])
        
        if combined.empty:
            return None
        
        range_label = self._format_gw_range_label(selected_gws)
        computed_title_suffix = title_suffix or ('Season' if last_n_gw is None else f'Last {max(window_len, 1)} Games')
        chart_title = f"Top {min(top_n, len(combined))} Defenders - {computed_title_suffix} ({range_label})"
        
        # Color by clean sheet percentage
        cs_colors = {
            'high': '#4ADE80',    # Green - 40%+ CS
            'medium': '#60A5FA',  # Blue - 20-40% CS
            'low': '#FBBF24'      # Amber - <20% CS
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate quadrant dividers (median values)
        x_median = combined['def_actions_per_90'].median()
        y_median = combined['total_points'].median()
        
        # Create scatter plot
        for _, row in combined.iterrows():
            cs_pct = row.get('cs_pct', 0)
            if cs_pct >= 40:
                base_color = cs_colors['high']
                cs_cat = '40%+ CS'
            elif cs_pct >= 20:
                base_color = cs_colors['medium']
                cs_cat = '20-40% CS'
            else:
                base_color = cs_colors['low']
                cs_cat = '<20% CS'
            
            is_squad = row['player_id'] in squad_ids
            edge_color = '#0e0e0e' if is_squad else 'white'
            alpha = 0.95 if is_squad else 0.55
            
            # Bubble size based on price efficiency (scaled to match attacker charts)
            pts_per_m = row.get('pts_per_m', 1)
            size = max(pts_per_m * 15, 40)  # Smaller multiplier to match attacker bubble sizes
            
            ax.scatter(
                row['def_actions_per_90'],
                row['total_points'],
                s=size,
                c=base_color,
                edgecolors=edge_color,
                linewidths=2 if is_squad else 0.5,
                alpha=alpha,
                zorder=5 if is_squad else 3
            )
            
            # Label logic
            name = row.get('web_name', 'Unknown')
            x_val = row['def_actions_per_90']
            y_val = row['total_points']
            
            # Determine label position
            if x_val > x_median:
                ha = 'left'
                x_offset = 0.15
            else:
                ha = 'right'
                x_offset = -0.15
            
            fontweight = 'bold' if is_squad else 'normal'
            fontsize = 9 if is_squad else 8
            
            # Add box around squad player labels
            if is_squad:
                ax.annotate(
                    name, (x_val + x_offset, y_val),
                    fontsize=fontsize, fontweight=fontweight, ha=ha, va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8),
                    zorder=10
                )
            else:
                ax.annotate(
                    name, (x_val + x_offset, y_val),
                    fontsize=fontsize, fontweight=fontweight, ha=ha, va='center',
                    alpha=0.9, zorder=4
                )
        
        # Draw quadrant lines
        ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Get axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Quadrant labels
        # Top-right: ELITE (high activity, high points)
        ax.text(x_max - 0.02*(x_max-x_min), y_max - 0.02*(y_max-y_min), 
                'ELITE\n(BPS Magnets)', ha='right', va='top',
                fontsize=11, fontweight='bold', color='#DC2626', alpha=0.8)
        
        # Top-left: CS MERCHANTS (low activity, high points)
        ax.text(x_min + 0.02*(x_max-x_min), y_max - 0.02*(y_max-y_min),
                'CS MERCHANTS\n(Fixture Dependent)', ha='left', va='top',
                fontsize=11, fontweight='bold', color='#7C3AED', alpha=0.8)
        
        # Bottom-right: VOLUME (high activity, low points)
        ax.text(x_max - 0.02*(x_max-x_min), y_min + 0.02*(y_max-y_min),
                'VOLUME\n(Buy Signal)', ha='right', va='bottom',
                fontsize=11, fontweight='bold', color='#F59E0B', alpha=0.8)
        
        # Bottom-left: AVOID
        ax.text(x_min + 0.02*(x_max-x_min), y_min + 0.02*(y_max-y_min),
                'AVOID', ha='left', va='bottom',
                fontsize=11, fontweight='bold', color='#9CA3AF', alpha=0.6)
        
        # Labels and title
        ax.set_xlabel('Defensive Actions (T+I+C+B) per 90', fontsize=12, fontweight='medium')
        ax.set_ylabel(f'L{last_n_gw} Points' if last_n_gw else 'Total Points', fontsize=12, fontweight='medium')
        ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=15)
        
        # Legend for CS percentage
        legend_elements = [
            plt.scatter([], [], c=cs_colors['high'], s=100, label='40%+ CS', edgecolors='white'),
            plt.scatter([], [], c=cs_colors['medium'], s=100, label='20-40% CS', edgecolors='white'),
            plt.scatter([], [], c=cs_colors['low'], s=100, label='<20% CS', edgecolors='white'),
            plt.scatter([], [], c='gray', s=100, label='Your squad (outlined)', edgecolors='black', linewidths=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', title='Clean Sheet Rate',
                 frameon=True, fancybox=True, shadow=False,
                 edgecolor='#E5E7EB', facecolor='white', fontsize=9)
        
        # Add size legend for price efficiency (positioned to avoid overlap with axis)
        # Create legend handles for bubble sizes
        legend_sizes = [4, 6, 8]
        legend_labels = [f'pts/£m: {s}' for s in legend_sizes]
        legend_handles = [plt.scatter([], [], s=s*15, color='#D1D5DB', alpha=0.7, edgecolors='white', linewidth=1)
                         for s in legend_sizes]
        
        size_legend = ax.legend(legend_handles, legend_labels, 
                              loc='lower center', title='Price Efficiency',
                              bbox_to_anchor=(0.5, -0.12),
                              framealpha=0.9, fontsize=8, ncol=3,
                              frameon=True, fancybox=True, edgecolor='#E5E7EB')
        ax.add_artist(size_legend)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the size legend
        self._save_plot(filename)
        
        # Write summary data for tables
        summary_key = 'season' if last_n_gw is None else f'last{last_n_gw}'
        summary_path = self.output_dir / 'defensive_value_summary.json'
        try:
            self._write_defensive_summary(summary_key, range_label, combined, summary_path, x_median, y_median)
        except Exception as exc:
            print(f"WARNING: Failed to write defensive summary ({summary_key}): {exc}")
        
        return filename

    def generate_defensive_value_scatter_recent(self, all_gw_data: Dict,
                                               fpl_core_season_data: Dict,
                                               squad_ids: List[int],
                                               last_n_gw: int = 5,
                                               top_n: int = 25) -> str:
        """Wrapper for recent-form defensive value plot (last N GWs)."""
        filename = f'defensive_value_scatter_last{last_n_gw}.png'
        return self.generate_defensive_value_scatter(
            all_gw_data=all_gw_data,
            fpl_core_season_data=fpl_core_season_data,
            squad_ids=squad_ids,
            top_n=top_n,
            last_n_gw=last_n_gw,
            filename=filename,
            title_suffix=None
        )

    def _aggregate_fplcore_goalkeeper_stats(self, all_gw_data: Dict,
                                           fpl_core_season_data: Dict,
                                           min_minutes: int = 450,
                                           start_gw: int = None,
                                           end_gw: int = None,
                                           last_n_gw: int = None) -> pd.DataFrame:
        """Aggregate goalkeeper shot-stopping statistics from FPL Core playermatchstats.
        
        Computes goals prevented (xGOT faced - goals conceded) and save% over a window,
        plus total FPL points and clean sheets from player_gameweek_stats.
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(
            all_gw_data,
            start_gw=start_gw,
            end_gw=end_gw,
            last_n_gw=last_n_gw
        )
        
        if not selected_gws:
            return pd.DataFrame()
        
        all_match_stats = []
        all_gw_points = []
        
        for gw_num in selected_gws:
            gw_data = all_gw_data.get(gw_num, {})
            
            pms = gw_data.get('playermatchstats')
            if pms is not None and hasattr(pms, 'empty') and not pms.empty:
                match_df = pms.copy()
                match_df['gw'] = gw_num
                all_match_stats.append(match_df)
            
            pgs = gw_data.get('player_gameweek_stats')
            if pgs is not None and hasattr(pgs, 'empty') and not pgs.empty:
                gw_stats = pgs[['id', 'event_points', 'clean_sheets']].copy()
                gw_stats = gw_stats.rename(columns={'id': 'player_id'})
                gw_stats['event_points'] = pd.to_numeric(gw_stats['event_points'], errors='coerce').fillna(0)
                gw_stats['clean_sheets'] = pd.to_numeric(gw_stats['clean_sheets'], errors='coerce').fillna(0)
                all_gw_points.append(gw_stats)
        
        if not all_match_stats:
            return pd.DataFrame()
        
        match_stats = pd.concat(all_match_stats, ignore_index=True)
        
        # Ensure expected goalkeeper columns exist and are numeric
        gk_cols = ['saves', 'goals_conceded', 'xgot_faced', 'goals_prevented']
        for col in gk_cols:
            if col not in match_stats.columns:
                match_stats[col] = 0
            match_stats[col] = pd.to_numeric(match_stats[col], errors='coerce').fillna(0)
        
        if 'minutes_played' not in match_stats.columns:
            return pd.DataFrame()
        match_stats['minutes_played'] = pd.to_numeric(match_stats['minutes_played'], errors='coerce').fillna(0)
        
        player_agg = match_stats.groupby('player_id').agg({
            'minutes_played': 'sum',
            'saves': 'sum',
            'goals_conceded': 'sum',
            'xgot_faced': 'sum',
            'goals_prevented': 'sum'
        }).reset_index()
        
        # Aggregate points + clean sheets for the window
        if all_gw_points:
            points_df = pd.concat(all_gw_points, ignore_index=True)
            points_agg = points_df.groupby('player_id').agg({
                'event_points': 'sum',
                'clean_sheets': 'sum'
            }).reset_index()
            points_agg = points_agg.rename(columns={'event_points': 'total_points'})
            player_agg = player_agg.merge(points_agg, on='player_id', how='left')
            player_agg['total_points'] = player_agg['total_points'].fillna(0).astype(int)
            player_agg['clean_sheets'] = player_agg['clean_sheets'].fillna(0).astype(int)
        else:
            player_agg['total_points'] = 0
            player_agg['clean_sheets'] = 0
        
        # Filter by minimum minutes
        player_agg = player_agg[player_agg['minutes_played'] >= min_minutes].copy()
        
        if player_agg.empty:
            return player_agg
        
        # Fallback goals_prevented if missing in source
        if player_agg['goals_prevented'].isna().all() and 'xgot_faced' in player_agg.columns:
            player_agg['goals_prevented'] = player_agg['xgot_faced'] - player_agg['goals_conceded']
        
        # Save percentage: saves / (saves + goals conceded)
        denom = (player_agg['saves'] + player_agg['goals_conceded']).replace(0, np.nan)
        player_agg['save_pct'] = (player_agg['saves'] / denom * 100.0).fillna(0.0).round(1)
        
        # Merge player identity from playerstats (latest row <= max GW in window)
        max_gw = max(selected_gws) if selected_gws else None
        if fpl_core_season_data and 'playerstats' in fpl_core_season_data:
            ps = fpl_core_season_data['playerstats'].copy()
            if 'gw' in ps.columns and max_gw is not None:
                ps = ps[ps['gw'] <= max_gw].copy()
                ps = ps.sort_values('gw', ascending=False)
            player_info = ps[['id', 'web_name', 'first_name', 'second_name']].drop_duplicates(subset=['id'], keep='first')
            player_agg = player_agg.merge(player_info, left_on='player_id', right_on='id', how='left')
        
        # Merge position from players dataset
        if fpl_core_season_data and 'players' in fpl_core_season_data:
            players_pos = fpl_core_season_data['players'][['player_id', 'position']].copy()
            player_agg = player_agg.merge(players_pos, on='player_id', how='left')
        
        return player_agg

    def _write_goalkeeper_summary(self,
                                  summary_key: str,
                                  range_label: str,
                                  df: pd.DataFrame,
                                  out_path: Path,
                                  x_split: float = 0.0,
                                  y_split: float = None) -> None:
        """Persist summary tables and quadrant categories for goalkeeper shot-stopping charts."""
        if df.empty:
            return

        subset = df.copy()
        subset = subset.sort_values(['total_points', 'goals_prevented'], ascending=False)

        # Top goalkeepers table
        rows = []
        for _, row in subset.head(12).iterrows():
            rows.append({
                'name': row.get('web_name', 'Unknown'),
                'gp': round(float(row.get('goals_prevented', 0)), 2),
                'save_pct': round(float(row.get('save_pct', 0)), 1),
                'cs': int(row.get('clean_sheets', 0)),
                'pts': int(row.get('total_points', 0))
            })

        # Insights (optional, mirror other summaries)
        def _top_by(col):
            if subset.empty or col not in subset.columns:
                return None
            return subset.loc[subset[col].idxmax()]

        top_gp = _top_by('goals_prevented')
        top_pts = _top_by('total_points')
        top_cs = _top_by('clean_sheets')
        top_save = _top_by('save_pct')

        insights = []
        if top_gp is not None:
            insights.append(f"Shot-stopper: {top_gp.get('web_name', 'Unknown')} leads goals prevented at {top_gp.get('goals_prevented', 0):.2f} ({range_label}).")
        if top_pts is not None:
            insights.append(f"Top scorer: {top_pts.get('web_name', 'Unknown')} tops points with {int(top_pts.get('total_points', 0))} ({range_label}).")
        if top_cs is not None:
            insights.append(f"Clean sheet king: {top_cs.get('web_name', 'Unknown')} has {int(top_cs.get('clean_sheets', 0))} clean sheets ({range_label}).")
        if top_save is not None:
            insights.append(f"Safe hands: {top_save.get('web_name', 'Unknown')} leads save% at {top_save.get('save_pct', 0):.1f}\\% ({range_label}).")

        # Quadrant categories: ELITE / PROTECTED / UNLUCKY / AVOID
        categories = {}
        if y_split is None:
            y_split = subset['total_points'].median() if not subset.empty else 0

        def add_cat(key, frame):
            if frame.empty:
                return
            categories[key] = [{
                'name': r.get('web_name', 'Unknown'),
                'gp': round(float(r.get('goals_prevented', 0)), 2),
                'save_pct': round(float(r.get('save_pct', 0)), 1),
                'cs': int(r.get('clean_sheets', 0)),
                'pts': int(r.get('total_points', 0))
            } for _, r in frame.head(8).iterrows()]

        elite = subset[(subset['goals_prevented'] >= x_split) & (subset['total_points'] >= y_split)]
        unlucky = subset[(subset['goals_prevented'] >= x_split) & (subset['total_points'] < y_split)]
        protected = subset[(subset['goals_prevented'] < x_split) & (subset['total_points'] >= y_split)]
        avoid = subset[(subset['goals_prevented'] < x_split) & (subset['total_points'] < y_split)]

        add_cat('elite', elite)
        add_cat('protected', protected)
        add_cat('unlucky', unlucky)
        add_cat('avoid', avoid)

        payload = {
            'range_label': range_label,
            'rows': rows,
            'insights': insights,
            'categories': categories
        }

        existing = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except Exception:
                existing = {}

        existing[summary_key] = payload
        out_path.write_text(json.dumps(existing, indent=2))

    def generate_goalkeeper_value_scatter(self, all_gw_data: Dict,
                                          fpl_core_season_data: Dict,
                                          squad_ids: List[int],
                                          top_n: int = 20,
                                          last_n_gw: int = None,
                                          filename: str = 'goalkeeper_value_scatter.png',
                                          title_suffix: str = None,
                                          min_minutes: int = None) -> str:
        """Generate Goalkeeper shot-stopping bubble scatter plot (Goals Prevented vs Points).
        
        X-axis: Goals prevented (xGOT faced - Goals conceded)
        Y-axis: FPL points over the window
        Bubble size: Clean sheets
        Color: Save%
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(all_gw_data, last_n_gw=last_n_gw)
        window_len = len(selected_gws)
        
        if min_minutes is None:
            if last_n_gw is None:
                min_minutes = 450
            else:
                min_minutes = max(90, 45 * window_len) if window_len else 90
        
        gk_df = self._aggregate_fplcore_goalkeeper_stats(
            all_gw_data=all_gw_data,
            fpl_core_season_data=fpl_core_season_data,
            min_minutes=min_minutes,
            last_n_gw=last_n_gw
        )
        
        if gk_df.empty:
            return None
        
        # Filter to goalkeepers only
        if 'position' in gk_df.columns:
            gk_df = gk_df[gk_df['position'] == 'Goalkeeper'].copy()
        
        if gk_df.empty:
            return None
        
        # Sort by points and select top players (always include squad goalkeepers)
        gk_df = gk_df.sort_values('total_points', ascending=False)
        top_players = gk_df.head(top_n)
        squad_players = gk_df[gk_df['player_id'].isin(squad_ids)]
        combined = pd.concat([top_players, squad_players]).drop_duplicates(subset=['player_id'])
        
        if combined.empty:
            return None
        
        range_label = self._format_gw_range_label(selected_gws)
        computed_title_suffix = title_suffix or ('Season' if last_n_gw is None else f'Last {max(window_len, 1)} Games')
        chart_title = f"Top {min(top_n, len(combined))} Goalkeepers - {computed_title_suffix} ({range_label})"
        
        save_colors = {
            'high': '#10B981',  # Green
            'mid': '#3B82F6',   # Blue
            'low': '#F59E0B'    # Amber
        }
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Quadrant split: x at 0 (outperforming vs underperforming), y at median points
        x_split = 0.0
        y_split = combined['total_points'].median()
        
        # Scatter
        for _, row in combined.iterrows():
            save_pct = float(row.get('save_pct', 0))
            if save_pct >= 75:
                base_color = save_colors['high']
            elif save_pct >= 65:
                base_color = save_colors['mid']
            else:
                base_color = save_colors['low']
            
            is_squad = row['player_id'] in squad_ids
            edge_color = '#0e0e0e' if is_squad else 'white'
            alpha = 0.95 if is_squad else 0.55
            
            cs = int(row.get('clean_sheets', 0))
            size = max(cs * 90, 50)
            
            x_val = float(row.get('goals_prevented', 0))
            y_val = float(row.get('total_points', 0))
            
            ax.scatter(
                x_val,
                y_val,
                s=size,
                c=base_color,
                edgecolors=edge_color,
                linewidths=2 if is_squad else 0.6,
                alpha=alpha,
                zorder=5 if is_squad else 3
            )
            
            name = row.get('web_name', 'Unknown')
            if x_val >= x_split:
                ha = 'left'
                x_offset = 0.08
            else:
                ha = 'right'
                x_offset = -0.08
            
            fontsize = 9 if is_squad else 8
            fontweight = 'bold' if is_squad else 'normal'
            
            if is_squad:
                ax.annotate(
                    name, (x_val + x_offset, y_val),
                    fontsize=fontsize, fontweight=fontweight, ha=ha, va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8),
                    zorder=10
                )
            else:
                ax.annotate(
                    name, (x_val + x_offset, y_val),
                    fontsize=fontsize, fontweight=fontweight, ha=ha, va='center',
                    alpha=0.9, zorder=4
                )
        
        # Quadrant lines
        ax.axvline(x=x_split, color='#EF4444', linestyle='-', alpha=0.35, linewidth=2)
        ax.axhline(y=y_split, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Quadrant labels
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        ax.text(x_max - 0.02*(x_max-x_min), y_max - 0.02*(y_max-y_min),
                'ELITE\n(Shot Stoppers)', ha='right', va='top',
                fontsize=11, fontweight='bold', color=save_colors['high'], alpha=0.8)
        
        ax.text(x_min + 0.02*(x_max-x_min), y_max - 0.02*(y_max-y_min),
                'PROTECTED\n(Good Defence)', ha='left', va='top',
                fontsize=11, fontweight='bold', color=save_colors['mid'], alpha=0.75)
        
        ax.text(x_max - 0.02*(x_max-x_min), y_min + 0.02*(y_max-y_min),
                'UNLUCKY\n(Due Points)', ha='right', va='bottom',
                fontsize=11, fontweight='bold', color=save_colors['low'], alpha=0.8)
        
        ax.text(x_min + 0.02*(x_max-x_min), y_min + 0.02*(y_max-y_min),
                'AVOID\n(Leaky)', ha='left', va='bottom',
                fontsize=11, fontweight='bold', color='#9CA3AF', alpha=0.6)
        
        ax.set_xlabel('Goals Prevented (xGOT - Goals Conceded)', fontsize=12, fontweight='medium')
        ax.set_ylabel(f'L{max(window_len, 1)} Points' if last_n_gw else 'Total Points', fontsize=12, fontweight='medium')
        ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=22)
        
        # Legend for save%
        legend_elements = [
            plt.scatter([], [], c=save_colors['high'], s=110, label='75%+ Save', edgecolors='white'),
            plt.scatter([], [], c=save_colors['mid'], s=110, label='65-75% Save', edgecolors='white'),
            plt.scatter([], [], c=save_colors['low'], s=110, label='<65% Save', edgecolors='white'),
        ]
        if squad_ids:
            legend_elements.append(
                plt.scatter([], [], c='gray', s=110, label='Your squad (outlined)', edgecolors='black', linewidths=2)
            )
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                  ncol=len(legend_elements), frameon=False, fontsize=9)
        
        # Bubble size legend for clean sheets
        legend_cs = [1, 3, 5]
        cs_labels = [f'CS: {v}' for v in legend_cs]
        cs_handles = [plt.scatter([], [], s=max(v * 90, 50), color='#D1D5DB', alpha=0.7, edgecolors='white', linewidth=1)
                      for v in legend_cs]
        
        size_legend = ax.legend(cs_handles, cs_labels,
                               loc='lower center', title='Clean Sheets',
                               bbox_to_anchor=(0.5, -0.12),
                               framealpha=0.9, fontsize=8, ncol=3,
                               frameon=True, fancybox=True, edgecolor='#E5E7EB')
        ax.add_artist(size_legend)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self._save_plot(filename)

        # Write summary data for tables
        summary_key = 'season' if last_n_gw is None else f'last{last_n_gw}'
        summary_path = self.output_dir / 'goalkeeper_value_summary.json'
        try:
            self._write_goalkeeper_summary(summary_key, range_label, combined, summary_path,
                                           x_split=x_split, y_split=y_split)
        except Exception as exc:
            print(f"WARNING: Failed to write goalkeeper summary ({summary_key}): {exc}")
        
        return filename

    def generate_goalkeeper_value_scatter_recent(self, all_gw_data: Dict,
                                                 fpl_core_season_data: Dict,
                                                 squad_ids: List[int],
                                                 last_n_gw: int = 5,
                                                 top_n: int = 20) -> str:
        """Wrapper for recent-form goalkeeper value plot (last N GWs)."""
        filename = f'goalkeeper_value_scatter_last{last_n_gw}.png'
        return self.generate_goalkeeper_value_scatter(
            all_gw_data=all_gw_data,
            fpl_core_season_data=fpl_core_season_data,
            squad_ids=squad_ids,
            top_n=top_n,
            last_n_gw=last_n_gw,
            filename=filename,
            title_suffix=None
        )

    def generate_clinical_wasteful_chart(self, fpl_core_season_data: Dict,
                                        fpl_core_gw_data: Dict,
                                        squad_ids: List[int],
                                        current_gw: int,
                                        top_n: int = 10) -> Tuple[str, str]:
        """Generate Clinical vs Wasteful (Goals) horizontal bar charts.
        
        Creates two charts comparing actual goals vs expected goals:
        - Single GW performance
        - Season cumulative performance
        
        Args:
            fpl_core_season_data: Season-level FPL Core data
            fpl_core_gw_data: Current gameweek FPL Core data
            squad_ids: List of player IDs in user's squad
            current_gw: Current gameweek number
            top_n: Number of top performers to show
            
        Returns:
            Tuple of (season_filename, gw_filename)
        """
        import pandas as pd
        
        aggregated = self._aggregate_fplcore_player_stats(
            fpl_core_season_data, fpl_core_gw_data, current_gw, min_minutes=450
        )
        
        filenames = []
        
        # Generate Season chart
        if aggregated['season'] is not None:
            season_df = aggregated['season'].copy()
            
            # Filter to players with goals or xG > 0.5
            season_df = season_df[(season_df['goals'] > 0) | (season_df['xG'] > 0.5)]
            
            # Sort by absolute differential
            season_df = season_df.sort_values('goal_diff', ascending=False)
            
            # Get top clinical and wasteful
            top_clinical = season_df[season_df['goal_diff'] > 0].head(top_n)
            top_wasteful = season_df[season_df['goal_diff'] < 0].tail(top_n).iloc[::-1]
            
            # Combine and include squad players
            squad_players = season_df[season_df['id'].isin(squad_ids)]
            combined = pd.concat([top_clinical, top_wasteful, squad_players]).drop_duplicates(subset=['id'])
            
            if not combined.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare data
                combined = combined.sort_values('goal_diff', ascending=True)
                y_pos = range(len(combined))
                
                # Color bars based on differential
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in combined['goal_diff']]
                
                # Create bars
                bars = ax.barh(y_pos, combined['goal_diff'], color=colors, alpha=0.8)
                
                # Labels with player names and stats
                labels = [
                    f"{row['web_name']} ({int(row['goals'])}G / {row['xG']:.2f}xG)"
                    for _, row in combined.iterrows()
                ]
                
                # Highlight squad players with bold
                label_weights = ['bold' if pid in squad_ids else 'normal' 
                               for pid in combined['id']]
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                for tick, weight in zip(ax.get_yticklabels(), label_weights):
                    tick.set_weight(weight)
                
                ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
                ax.set_xlabel('Goals - xG Differential', fontsize=11, fontweight='bold')
                ax.set_title(f'Clinical vs Wasteful: Season (GW 1-{current_gw})', 
                           fontsize=13, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                season_file = 'clinical_wasteful_season.png'
                self._save_plot(season_file)
                filenames.append(season_file)
        
        # Generate GW chart
        if aggregated['gameweek'] is not None:
            gw_df = aggregated['gameweek'].copy()
            
            # Filter to players with goals or xG in this GW
            gw_df = gw_df[(gw_df['goals'] > 0) | (gw_df['xG'] > 0.3)]
            
            # Sort by differential
            gw_df = gw_df.sort_values('goal_diff', ascending=False)
            
            # Get top performers
            top_clinical = gw_df[gw_df['goal_diff'] > 0].head(top_n)
            top_wasteful = gw_df[gw_df['goal_diff'] < 0].tail(top_n).iloc[::-1]
            
            # Include squad players
            squad_players = gw_df[gw_df['id'].isin(squad_ids)]
            combined = pd.concat([top_clinical, top_wasteful, squad_players]).drop_duplicates(subset=['id'])
            
            if not combined.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                combined = combined.sort_values('goal_diff', ascending=True)
                y_pos = range(len(combined))
                
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in combined['goal_diff']]
                bars = ax.barh(y_pos, combined['goal_diff'], color=colors, alpha=0.8)
                
                labels = [
                    f"{row['web_name']} ({int(row['goals'])}G / {row['xG']:.2f}xG)"
                    for _, row in combined.iterrows()
                ]
                
                label_weights = ['bold' if pid in squad_ids else 'normal' 
                               for pid in combined['id']]
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                for tick, weight in zip(ax.get_yticklabels(), label_weights):
                    tick.set_weight(weight)
                
                ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
                ax.set_xlabel('Goals - xG Differential', fontsize=11, fontweight='bold')
                ax.set_title(f'Clinical vs Wasteful: GW {current_gw}', 
                           fontsize=13, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                gw_file = 'clinical_wasteful_gw.png'
                self._save_plot(gw_file)
                filenames.append(gw_file)
        
        return tuple(filenames) if len(filenames) == 2 else (filenames[0] if filenames else None, None)

    def generate_clinical_wasteful_chart_squad_only(self, fpl_core_season_data: Dict,
                                                     fpl_core_gw_data: Dict,
                                                     squad_ids: List[int],
                                                     current_gw: int) -> Tuple[str, str]:
        """Generate squad-only Clinical vs Wasteful (Goals) horizontal bar charts.
        
        Creates two charts showing only the user's squad players:
        - Single GW performance
        - Season cumulative performance
        
        Args:
            fpl_core_season_data: Season-level FPL Core data
            fpl_core_gw_data: Current gameweek FPL Core data
            squad_ids: List of player IDs in user's squad
            current_gw: Current gameweek number
            
        Returns:
            Tuple of (season_filename, gw_filename)
        """
        import pandas as pd
        
        aggregated = self._aggregate_fplcore_player_stats(
            fpl_core_season_data, fpl_core_gw_data, current_gw, min_minutes=0, squad_ids=squad_ids
        )
        
        filenames = []
        
        # Generate Season chart
        if aggregated['season'] is not None:
            season_df = aggregated['season']
            # CRITICAL: Filter to ONLY squad players first
            squad_df = season_df[season_df['id'].isin(squad_ids)].copy()
            
            # Apply filters: exclude GKP, minimum minutes (45 * GW), and players with zero involvement
            min_season_minutes = 45 * current_gw
            squad_df = squad_df[
                (squad_df['position'] != 'Goalkeeper') &  # Exclude goalkeepers
                (squad_df['minutes'] >= min_season_minutes) &  # Minutes filter
                ((squad_df['goals'] > 0) | (squad_df['xG'] > 0))  # Exclude players with no goal involvement
            ].copy()
            
            # Safety check
            if len(squad_df) > 15:
                print(f"WARNING: Found {len(squad_df)} players in squad, expected max 15. Truncating.")
                squad_df = squad_df.head(15)
            
            if not squad_df.empty:
                squad_df = squad_df.sort_values('goal_diff', ascending=True)
                
                # Use fixed reasonable height for squad plots
                fig, ax = plt.subplots(figsize=(10, min(10, max(6, len(squad_df) * 0.6))))
                y_pos = range(len(squad_df))
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in squad_df['goal_diff']]
                
                ax.barh(y_pos, squad_df['goal_diff'], color=colors, alpha=0.8)
                
                # Fixed label formatting to prevent overlap
                labels = [f"{row['web_name']} ({int(row['goals'])}G, {row['xG']:.2f}xG)"
                         for _, row in squad_df.iterrows()]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                
                ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
                ax.set_xlabel('Goals - xG Differential', fontsize=12, fontweight='bold')
                ax.set_title(f'Your Squad: Clinical vs Wasteful (Season GW 1-{current_gw})', 
                           fontsize=14, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                season_file = 'clinical_wasteful_season_squad.png'
                self._save_plot(season_file)
                filenames.append(season_file)
        
        # Generate GW chart
        if aggregated['gameweek'] is not None:
            gw_df = aggregated['gameweek']
            # CRITICAL: Filter to ONLY squad players first
            squad_df = gw_df[gw_df['id'].isin(squad_ids)].copy()
            
            # Apply filters: exclude GKP, minimum 45 minutes, and players with zero involvement
            squad_df = squad_df[
                (squad_df['position'] != 'Goalkeeper') &  # Exclude goalkeepers
                (squad_df['minutes'] >= 45) &  # Minutes filter
                ((squad_df['goals'] > 0) | (squad_df['xG'] > 0))  # Exclude players with no goal involvement
            ].copy()
            
            # Safety check
            if len(squad_df) > 15:
                print(f"WARNING: Found {len(squad_df)} players in GW squad, expected max 15. Truncating.")
                squad_df = squad_df.head(15)
            
            if not squad_df.empty:
                squad_df = squad_df.sort_values('goal_diff', ascending=True)
                
                # Use fixed reasonable height for squad plots
                fig, ax = plt.subplots(figsize=(10, min(10, max(5, len(squad_df) * 0.6))))
                y_pos = range(len(squad_df))
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in squad_df['goal_diff']]
                
                ax.barh(y_pos, squad_df['goal_diff'], color=colors, alpha=0.8)
                
                # Fixed label formatting to prevent overlap
                labels = [f"{row['web_name']} ({int(row['goals'])}G, {row['xG']:.2f}xG)"
                         for _, row in squad_df.iterrows()]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                
                ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
                ax.set_xlabel('Goals - xG Differential', fontsize=12, fontweight='bold')
                ax.set_title(f'Your Squad: Clinical vs Wasteful (GW {current_gw})', 
                           fontsize=14, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                gw_file = 'clinical_wasteful_gw_squad.png'
                self._save_plot(gw_file)
                filenames.append(gw_file)
        
        return tuple(filenames) if len(filenames) == 2 else (filenames[0] if filenames else None, None)

    def generate_clutch_frustrated_chart(self, fpl_core_season_data: Dict,
                                        fpl_core_gw_data: Dict,
                                        squad_ids: List[int],
                                        current_gw: int,
                                        top_n: int = 10) -> Tuple[str, str]:
        """Generate Clutch vs Frustrated (Assists) horizontal bar charts.
        
        Creates two charts comparing actual assists vs expected assists:
        - Single GW performance
        - Season cumulative performance
        
        Args:
            fpl_core_season_data: Season-level FPL Core data
            fpl_core_gw_data: Current gameweek FPL Core data
            squad_ids: List of player IDs in user's squad
            current_gw: Current gameweek number
            top_n: Number of top performers to show
            
        Returns:
            Tuple of (season_filename, gw_filename)
        """
        import pandas as pd
        
        aggregated = self._aggregate_fplcore_player_stats(
            fpl_core_season_data, fpl_core_gw_data, current_gw, min_minutes=450
        )
        
        filenames = []
        
        # Generate Season chart
        if aggregated['season'] is not None:
            season_df = aggregated['season'].copy()
            
            # Filter to players with assists or xA > 0.5
            season_df = season_df[(season_df['assists'] > 0) | (season_df['xA'] > 0.5)]
            
            # Sort by absolute differential
            season_df = season_df.sort_values('assist_diff', ascending=False)
            
            # Get top clutch and frustrated
            top_clutch = season_df[season_df['assist_diff'] > 0].head(top_n)
            top_frustrated = season_df[season_df['assist_diff'] < 0].tail(top_n).iloc[::-1]
            
            # Combine and include squad players
            squad_players = season_df[season_df['id'].isin(squad_ids)]
            combined = pd.concat([top_clutch, top_frustrated, squad_players]).drop_duplicates(subset=['id'])
            
            if not combined.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare data
                combined = combined.sort_values('assist_diff', ascending=True)
                y_pos = range(len(combined))
                
                # Color bars based on differential
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in combined['assist_diff']]
                
                # Create bars
                bars = ax.barh(y_pos, combined['assist_diff'], color=colors, alpha=0.8)
                
                # Labels with player names and stats
                labels = [
                    f"{row['web_name']} ({int(row['assists'])}A / {row['xA']:.2f}xA)"
                    for _, row in combined.iterrows()
                ]
                
                # Highlight squad players with bold
                label_weights = ['bold' if pid in squad_ids else 'normal' 
                               for pid in combined['id']]
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                for tick, weight in zip(ax.get_yticklabels(), label_weights):
                    tick.set_weight(weight)
                
                ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
                ax.set_xlabel('Assists - xA Differential', fontsize=11, fontweight='bold')
                ax.set_title(f'Clutch vs Frustrated: Season (GW 1-{current_gw})', 
                           fontsize=13, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                season_file = 'clutch_frustrated_season.png'
                self._save_plot(season_file)
                filenames.append(season_file)
        
        # Generate GW chart
        if aggregated['gameweek'] is not None:
            gw_df = aggregated['gameweek'].copy()
            
            # Filter to players with assists or xA in this GW
            gw_df = gw_df[(gw_df['assists'] > 0) | (gw_df['xA'] > 0.3)]
            
            # Sort by differential
            gw_df = gw_df.sort_values('assist_diff', ascending=False)
            
            # Get top performers
            top_clutch = gw_df[gw_df['assist_diff'] > 0].head(top_n)
            top_frustrated = gw_df[gw_df['assist_diff'] < 0].tail(top_n).iloc[::-1]
            
            # Include squad players
            squad_players = gw_df[gw_df['id'].isin(squad_ids)]
            combined = pd.concat([top_clutch, top_frustrated, squad_players]).drop_duplicates(subset=['id'])
            
            if not combined.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                combined = combined.sort_values('assist_diff', ascending=True)
                y_pos = range(len(combined))
                
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in combined['assist_diff']]
                bars = ax.barh(y_pos, combined['assist_diff'], color=colors, alpha=0.8)
                
                labels = [
                    f"{row['web_name']} ({int(row['assists'])}A / {row['xA']:.2f}xA)"
                    for _, row in combined.iterrows()
                ]
                
                label_weights = ['bold' if pid in squad_ids else 'normal' 
                               for pid in combined['id']]
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                for tick, weight in zip(ax.get_yticklabels(), label_weights):
                    tick.set_weight(weight)
                
                ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
                ax.set_xlabel('Assists - xA Differential', fontsize=11, fontweight='bold')
                ax.set_title(f'Clutch vs Frustrated: GW {current_gw}', 
                           fontsize=13, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                gw_file = 'clutch_frustrated_gw.png'
                self._save_plot(gw_file)
                filenames.append(gw_file)
        
        return tuple(filenames) if len(filenames) == 2 else (filenames[0] if filenames else None, None)

    def generate_clutch_frustrated_chart_squad_only(self, fpl_core_season_data: Dict,
                                                     fpl_core_gw_data: Dict,
                                                     squad_ids: List[int],
                                                     current_gw: int) -> Tuple[str, str]:
        """Generate squad-only Clutch vs Frustrated (Assists) horizontal bar charts.
        
        Creates two charts showing only the user's squad players:
        - Single GW performance
        - Season cumulative performance
        
        Args:
            fpl_core_season_data: Season-level FPL Core data
            fpl_core_gw_data: Current gameweek FPL Core data
            squad_ids: List of player IDs in user's squad
            current_gw: Current gameweek number
            
        Returns:
            Tuple of (season_filename, gw_filename)
        """
        import pandas as pd
        
        aggregated = self._aggregate_fplcore_player_stats(
            fpl_core_season_data, fpl_core_gw_data, current_gw, min_minutes=0, squad_ids=squad_ids
        )
        
        filenames = []
        
        # Generate Season chart
        if aggregated['season'] is not None:
            season_df = aggregated['season']
            # CRITICAL: Filter to ONLY squad players first
            squad_df = season_df[season_df['id'].isin(squad_ids)].copy()
            
            # Apply filters: exclude GKP, minimum minutes (45 * GW), and players with zero involvement
            min_season_minutes = 45 * current_gw
            squad_df = squad_df[
                (squad_df['position'] != 'Goalkeeper') &  # Exclude goalkeepers
                (squad_df['minutes'] >= min_season_minutes) &  # Minutes filter
                ((squad_df['assists'] > 0) | (squad_df['xA'] > 0))  # Exclude players with no assist involvement
            ].copy()
            
            # Safety check
            if len(squad_df) > 15:
                print(f"WARNING: Found {len(squad_df)} players in squad (assists), expected max 15. Truncating.")
                squad_df = squad_df.head(15)
            
            if not squad_df.empty:
                squad_df = squad_df.sort_values('assist_diff', ascending=True)
                
                # Use fixed reasonable height for squad plots
                fig, ax = plt.subplots(figsize=(10, min(10, max(6, len(squad_df) * 0.6))))
                y_pos = range(len(squad_df))
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in squad_df['assist_diff']]
                
                ax.barh(y_pos, squad_df['assist_diff'], color=colors, alpha=0.8)
                
                # Fixed label formatting to prevent overlap
                labels = [f"{row['web_name']} ({int(row['assists'])}A, {row['xA']:.2f}xA)"
                         for _, row in squad_df.iterrows()]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                
                ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
                ax.set_xlabel('Assists - xA Differential', fontsize=12, fontweight='bold')
                ax.set_title(f'Your Squad: Clutch vs Frustrated (Season GW 1-{current_gw})', 
                           fontsize=14, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                season_file = 'clutch_frustrated_season_squad.png'
                self._save_plot(season_file)
                filenames.append(season_file)
        
        # Generate GW chart
        if aggregated['gameweek'] is not None:
            gw_df = aggregated['gameweek']
            # CRITICAL: Filter to ONLY squad players first
            squad_df = gw_df[gw_df['id'].isin(squad_ids)].copy()
            
            # Apply filters: exclude GKP, minimum 45 minutes, and players with zero involvement
            squad_df = squad_df[
                (squad_df['position'] != 'Goalkeeper') &  # Exclude goalkeepers
                (squad_df['minutes'] >= 45) &  # Minutes filter
                ((squad_df['assists'] > 0) | (squad_df['xA'] > 0))  # Exclude players with no assist involvement
            ].copy()
            
            # Safety check
            if len(squad_df) > 15:
                print(f"WARNING: Found {len(squad_df)} players in GW squad (assists), expected max 15. Truncating.")
                squad_df = squad_df.head(15)
            
            if not squad_df.empty:
                squad_df = squad_df.sort_values('assist_diff', ascending=True)
                
                # Use fixed reasonable height for squad plots
                fig, ax = plt.subplots(figsize=(10, min(10, max(5, len(squad_df) * 0.6))))
                y_pos = range(len(squad_df))
                colors = ['#2ecc71' if diff > 0 else '#e74c3c' for diff in squad_df['assist_diff']]
                
                ax.barh(y_pos, squad_df['assist_diff'], color=colors, alpha=0.8)
                
                # Fixed label formatting to prevent overlap
                labels = [f"{row['web_name']} ({int(row['assists'])}A, {row['xA']:.2f}xA)"
                         for _, row in squad_df.iterrows()]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                
                ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
                ax.set_xlabel('Assists - xA Differential', fontsize=12, fontweight='bold')
                ax.set_title(f'Your Squad: Clutch vs Frustrated (GW {current_gw})', 
                           fontsize=14, fontweight='bold', pad=15)
                
                
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plt.tight_layout()
                gw_file = 'clutch_frustrated_gw_squad.png'
                self._save_plot(gw_file)
                filenames.append(gw_file)
        
        return tuple(filenames) if len(filenames) == 2 else (filenames[0] if filenames else None, None)

    def generate_usage_output_scatter(self, all_gw_data: Dict,
                                     fpl_core_season_data: Dict,
                                     squad_ids: List[int],
                                     position_filter: List[str] = None,
                                     top_n: int = 25,
                                     last_n_gw: int = None,
                                     filename: str = 'usage_output_scatter.png',
                                     title_suffix: str = None,
                                     min_minutes: int = None) -> str:
        """Generate Usage vs Output bubble scatter plot.
        
        Shows relationship between player involvement (shots + box touches per 90)
        and actual output (goals + assists), with bubble size indicating xGI.
        
        Args:
            all_gw_data: Dict of all gameweek data
            fpl_core_season_data: Season-level FPL Core data
            squad_ids: List of player IDs in user's squad
            position_filter: List of positions to include (default: ['MID', 'FWD'])
            top_n: Number of top players by xGI to include
            last_n_gw: Limit to most recent N gameweeks (None = full season)
            filename: Output filename
            title_suffix: Optional label to append after "Top N Attackers - ..."
            min_minutes: Optional minimum minutes filter (auto if None)
            
        Returns:
            Filename of saved plot
        """
        import pandas as pd
        
        if position_filter is None:
            position_filter = ['MID', 'FWD']
        position_lookup = {
            'GKP': 'Goalkeeper', 'GK': 'Goalkeeper',
            'DEF': 'Defender', 'D': 'Defender',
            'MID': 'Midfielder', 'M': 'Midfielder',
            'FWD': 'Forward', 'F': 'Forward'
        }
        allowed_positions = [
            position_lookup.get(pos.upper(), pos) if isinstance(pos, str) else pos
            for pos in position_filter
        ]

        selected_gws = self._select_gameweeks(all_gw_data, last_n_gw=last_n_gw)
        window_len = len(selected_gws)

        if min_minutes is None:
            if last_n_gw is None:
                min_minutes = 450
            else:
                min_minutes = max(90, 45 * window_len) if window_len else 90

        usage_df = self._aggregate_fplcore_usage_stats(
            all_gw_data,
            fpl_core_season_data,
            min_minutes=min_minutes,
            last_n_gw=last_n_gw
        )
        
        if usage_df.empty:
            return None
        
        # Ensure position column exists and filter out Goalkeepers/other positions
        if 'position' not in usage_df.columns:
            usage_df['position'] = 'Unknown'
        usage_df['position'] = usage_df['position'].fillna('Unknown')
        usage_df = usage_df[usage_df['position'] != 'Goalkeeper']
        if allowed_positions:
            usage_df = usage_df[usage_df['position'].isin(allowed_positions)]
        
        if usage_df.empty:
            return None
        
        # Sort by xGI and get top performers
        usage_df = usage_df.sort_values('xGI', ascending=False)
        
        # Get top N players and always include squad
        top_players = usage_df.head(top_n)
        squad_players = usage_df[usage_df['player_id'].isin(squad_ids)]
        
        # Combine and deduplicate
        combined = pd.concat([top_players, squad_players]).drop_duplicates(subset=['player_id'])
        
        if combined.empty:
            return None
        
        range_label = self._format_gw_range_label(selected_gws)
        computed_title_suffix = title_suffix or ('Season' if last_n_gw is None else f'Last {max(window_len, 1)} GWs')
        chart_title = f"Top {min(top_n, len(combined))} Attackers - {computed_title_suffix} ({range_label})"
        summary_key = 'season' if last_n_gw is None else f'last{last_n_gw}'
        summary_path = self.output_dir / 'usage_output_summary.json'
        
        # Use unified position colors (slightly more saturated for scatter visibility)
        position_colors = {
            'Defender': '#60A5FA',   # Blue (matches DEF)
            'Midfielder': '#4ADE80', # Green (matches MID)
            'Forward': '#FB923C'     # Orange (matches FWD)
        }
        position_labels = {
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'FWD'
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate quadrant dividers (median values) - moved up for label logic
        x_median = combined['usage_per_90'].median()
        y_median = combined['total_output'].median()
        
        # Create scatter plot
        for _, row in combined.iterrows():
            pos = row['position']
            base_color = position_colors.get(pos, '#808080')
            is_squad = row['player_id'] in squad_ids
            edge_color = '#0e0e0e' if is_squad else 'white'
            alpha = 0.95 if is_squad else 0.55
            size = max(row['xGI'] * 50, 40)
            
            ax.scatter(row['usage_per_90'], row['total_output'], 
                      s=size, color=base_color, alpha=alpha, edgecolors=edge_color, linewidth=1.8 if is_squad else 1.0)
            
            # Label squad players and most big bubbles (top 50% by xGI)
            is_top_performer = row['xGI'] > combined['xGI'].quantile(0.50)
            
            if is_squad or is_top_performer:
                x_off = 5 if row['usage_per_90'] > x_median else -20
                y_off = 5 if row['total_output'] > y_median else -15
                fontweight = 'bold' if is_squad else 'normal'
                
                ax.annotate(row['web_name'], 
                           xy=(row['usage_per_90'], row['total_output']),
                           xytext=(x_off, y_off), textcoords='offset points',
                           fontsize=8, fontweight=fontweight,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=edge_color,
                                   alpha=0.75, linewidth=1.2))
        
        # Add quadrant lines
        ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Get plot limits for quadrant labels
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Add padding to limits (15%)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - x_range*0.15, x_max + x_range*0.15)
        ax.set_ylim(y_min - y_range*0.15, y_max + y_range*0.15)
        
        # Add quadrant labels (pegged to corners with transAxes)
        ax.text(0.98, 0.98, 
               'ELITE\n(Keep)', ha='right', va='top', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#2ecc71',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='#2ecc71', alpha=0.8, linewidth=2))
        
        ax.text(0.98, 0.02,
               'VOLUME\n(Buy Signal)', ha='right', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#0057ff',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#0057ff', alpha=0.8, linewidth=2))
        
        ax.text(0.02, 0.98,
               'CLINICAL\n(Sell Watch)', ha='left', va='top', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#ff9500',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#ff9500', alpha=0.8, linewidth=2))
        
        ax.text(0.02, 0.02,
               'AVOID', ha='left', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#e74c3c',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#e74c3c', alpha=0.8, linewidth=2))
        
        # Styling
        ax.set_xlabel('Usage: (Shots + Box Touches) per 90 mins', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Output: Goals + Assists', 
                     fontsize=12, fontweight='bold')
        
        ax.text(0.5, 1.02,
                f"{range_label} | Color = position | Outline + bold = your squad",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=10, fontweight='semibold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend for bubble sizes
        legend_sizes = [combined['xGI'].quantile(q) for q in [0.25, 0.5, 0.75]]
        legend_labels = [f'xGI: {size:.1f}' for size in legend_sizes]
        legend_handles = [plt.scatter([], [], s=max(size*50, 40), color='gray', alpha=0.6, edgecolors='white', linewidth=1)
                         for size in legend_sizes]
        
        legend1 = ax.legend(legend_handles, legend_labels, 
                          loc='lower center', title='Expected Goal Involvement',
                          framealpha=0.9, fontsize=9, ncol=3)
        ax.add_artist(legend1)
        
        # Position + squad legend
        position_handles = [
            plt.scatter([], [], s=120, color=color, edgecolors='white', linewidth=1.2,
                        label=position_labels.get(pos, pos))
            for pos, color in position_colors.items()
        ]
        squad_handle = plt.scatter([], [], s=120, facecolors='white', edgecolors='#0e0e0e',
                                  linewidth=1.8, label='Your squad (outlined)')
        
        ax.legend(handles=position_handles + [squad_handle], loc='upper right',
                 framealpha=0.9, fontsize=9, title='Attacker Type')
        
        plt.tight_layout()
        
        self._save_plot(filename)

        # Write summary table + insights
        try:
            self._write_usage_summary(summary_key, range_label, combined, summary_path, x_median, y_median)
        except Exception as exc:
            print(f"WARNING: Failed to write usage summary ({summary_key}): {exc}")
        
        return filename

    def generate_usage_output_scatter_recent(self, all_gw_data: Dict,
                                             fpl_core_season_data: Dict,
                                             squad_ids: List[int],
                                             last_n_gw: int = 5,
                                             top_n: int = 25) -> str:
        """Wrapper for recent-form usage vs output plot (last N GWs)."""
        filename = f'usage_output_scatter_last{last_n_gw}.png'
        return self.generate_usage_output_scatter(
            all_gw_data=all_gw_data,
            fpl_core_season_data=fpl_core_season_data,
            squad_ids=squad_ids,
            position_filter=['MID', 'FWD'],
            top_n=top_n,
            last_n_gw=last_n_gw,
            filename=filename,
            title_suffix=None
        )

    def generate_usage_output_scatter_squad_only(self, all_gw_data: Dict,
                                                 fpl_core_season_data: Dict,
                                                 squad_ids: List[int],
                                                 current_gw: int,
                                                 last_n_gw: int = None,
                                                 filename: str = 'usage_output_scatter_squad.png',
                                                 title_suffix: str = None) -> str:
        """Generate squad-only Usage vs Output bubble scatter plot.
        
        Shows relationship between player involvement and output for squad players only.
        Filters: >= 45*GW minutes, excludes GKP
        
        Args:
            all_gw_data: Dict of all gameweek data
            fpl_core_season_data: Season-level FPL Core data
            squad_ids: List of player IDs in user's squad
            current_gw: Current gameweek number
            last_n_gw: Limit to most recent N gameweeks (None = season to date)
            filename: Output filename
            title_suffix: Optional label override for the title
            
        Returns:
            Filename of saved plot
        """
        import pandas as pd
        
        selected_gws = self._select_gameweeks(
            all_gw_data,
            end_gw=current_gw,
            last_n_gw=last_n_gw
        )
        window_len = len(selected_gws)
        
        min_season_minutes = max(45 * window_len, 90) if window_len else 45 * current_gw
        usage_df = self._aggregate_fplcore_usage_stats(
            all_gw_data,
            fpl_core_season_data,
            min_minutes=min_season_minutes,
            end_gw=current_gw,
            last_n_gw=last_n_gw
        )
        
        # Filter to squad and exclude goalkeepers
        squad_usage = usage_df[
            (usage_df['player_id'].isin(squad_ids)) &
            (usage_df['position'] != 'Goalkeeper')  # Exclude goalkeepers
        ].copy()
        
        if squad_usage.empty:
            return None
        
        # Use unified position colors (slightly more saturated for scatter visibility)
        position_colors = {
            'Defender': '#60A5FA',   # Blue (matches DEF)
            'Midfielder': '#4ADE80', # Green (matches MID)
            'Forward': '#FB923C'     # Orange (matches FWD)
        }
        position_labels = {
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'FWD'
        }
        
        range_label = self._format_gw_range_label(selected_gws)
        computed_title_suffix = title_suffix or ('Season' if last_n_gw is None else f'Last {max(window_len, 1)} GWs')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each squad player
        for _, row in squad_usage.iterrows():
            pos = row['position']
            base_color = position_colors.get(pos, '#37003c')
            size = max(row['xGI'] * 50, 100)  # Minimum size for visibility
            
            ax.scatter(row['usage_per_90'], row['total_output'], 
                      s=size, color=base_color, alpha=0.85, 
                      edgecolors='white', linewidth=2)
            
            # Dynamic label positioning based on quadrant relative to median
            xytext = (8, 8)
            
            if len(squad_usage) > 1:
                x_med = squad_usage['usage_per_90'].median()
                y_med = squad_usage['total_output'].median()
                
                if row['usage_per_90'] < x_med and row['total_output'] < y_med:
                    xytext = (15, 15)
                elif row['usage_per_90'] < x_med and row['total_output'] > y_med:
                    xytext = (10, -15)
                elif row['usage_per_90'] > x_med and row['total_output'] < y_med:
                    xytext = (-10, 15)
                else:
                    xytext = (-10, -15)

            ax.annotate(row['web_name'], 
                       xy=(row['usage_per_90'], row['total_output']),
                       xytext=xytext, textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                               edgecolor=position_colors.get(pos, '#37003c'), alpha=0.85, linewidth=1.5))
        
        # Calculate medians for quadrant lines
        if len(squad_usage) > 1:
            x_median = squad_usage['usage_per_90'].median()
            y_median = squad_usage['total_output'].median()
            
            ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Get limits and add padding (increased to 15% to avoid label overlap)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - x_range*0.15, x_max + x_range*0.15)
        ax.set_ylim(y_min - y_range*0.15, y_max + y_range*0.15)
        
        # Quadrant labels (pegged to corners with transAxes)
        ax.text(0.98, 0.98, 
               'ELITE\n(Keep)', ha='right', va='top', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#2ecc71',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='#2ecc71', alpha=0.9, linewidth=2))
        
        ax.text(0.98, 0.02,
               'VOLUME\n(Buy Signal)', ha='right', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#0057ff',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#0057ff', alpha=0.9, linewidth=2))
        
        ax.text(0.02, 0.98,
               'CLINICAL\n(Sell Watch)', ha='left', va='top', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#ff9500',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#ff9500', alpha=0.9, linewidth=2))
        
        ax.text(0.02, 0.02,
               'AVOID', ha='left', va='bottom', transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='#e74c3c',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#e74c3c', alpha=0.9, linewidth=2))
        
        ax.set_xlabel('Usage: (Shots + Box Touches) per 90 mins', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Output: Goals + Assists', 
                     fontsize=12, fontweight='bold')
        
        ax.text(0.5, 1.02,
                f"{range_label} | Color = position",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=10, fontweight='semibold')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend for attacker type
        position_handles = [
            plt.scatter([], [], s=120, color=color, edgecolors='white', linewidth=1.2,
                        label=position_labels.get(pos, pos))
            for pos, color in position_colors.items()
        ]
        ax.legend(handles=position_handles, loc='upper right',
                 framealpha=0.9, fontsize=9, title='Attacker Type')
        
        plt.tight_layout()
        
        self._save_plot(filename)
        
        return filename

    def generate_usage_output_scatter_squad_recent(self, all_gw_data: Dict,
                                                   fpl_core_season_data: Dict,
                                                   squad_ids: List[int],
                                                   current_gw: int,
                                                   last_n_gw: int = 5) -> str:
        """Wrapper for recent-form squad-only usage vs output plot."""
        filename = f'usage_output_scatter_last{last_n_gw}_squad.png'
        return self.generate_usage_output_scatter_squad_only(
            all_gw_data=all_gw_data,
            fpl_core_season_data=fpl_core_season_data,
            squad_ids=squad_ids,
            current_gw=current_gw,
            last_n_gw=last_n_gw,
            filename=filename,
            title_suffix=None
        )

