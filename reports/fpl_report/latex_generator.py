"""LaTeX Report Generator Module

Generates comprehensive LaTeX documents for FPL team analysis.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
from .data_fetcher import get_bgw_dgw_gameweeks, get_top_global_teams
from utils.config import SEASON as DEFAULT_SEASON, TOP_GLOBAL_COUNT


class LaTeXReportGenerator:
    """Generates LaTeX report documents for FPL analysis."""

    def __init__(self, team_id: int, gameweek: int, plot_dir: Optional[Path] = None, session_cache=None):
        """Initialize the generator.

        Args:
            team_id: FPL team ID.
            gameweek: Current gameweek number.
            plot_dir: Directory containing generated plots.
            session_cache: Optional SessionCacheManager instance.
        """
        self.team_id = team_id
        self.gameweek = gameweek
        self.plot_dir = plot_dir
        self.session_cache = session_cache

    def generate_preamble(self) -> str:
        """Generate LaTeX document preamble with packages and styling."""
        return r"""\documentclass[11pt,a4paper]{article}
\pdfpageattr{/Rotate 0}

% Packages
\usepackage[margin=2cm, portrait]{geometry}

\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{shadows}
\usepackage{pgfplots}
\usepackage{pgf-pie}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{amssymb}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{fontawesome5}
\usepackage{hyperref}
\usepackage{parskip}
\usepackage{setspace}
\usepackage{multicol}
\usepackage[normalem]{ulem}
\usepackage{tcolorbox}
\usepackage{enumitem}

% Color definitions (FPL colors)
\definecolor{fplpurple}{RGB}{55,0,60}
\definecolor{fplgreen}{RGB}{0,255,135}
\definecolor{fplpink}{RGB}{255,47,130}
\definecolor{fpldark}{RGB}{28,28,30}
\definecolor{fplgray}{RGB}{128,128,128}
\definecolor{gold}{RGB}{255,215,0}
\definecolor{silver}{RGB}{192,192,192}
\definecolor{bronze}{RGB}{205,127,50}
\definecolor{rising}{RGB}{0,200,83}
\definecolor{falling}{RGB}{255,82,82}
\definecolor{stable}{RGB}{255,193,7}

% PGFPlots settings
\pgfplotsset{compat=1.18}

% Title formatting
\titleformat{\section}{\Large\bfseries\color{fplpurple}}{}{0em}{}[\color{fplgreen}\titlerule]
\titleformat{\subsection}{\large\bfseries\color{fpldark}}{}{0em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{fplpurple}}{}{0em}{}
"""

    def generate_header_footer(self, team_name: str, season: str) -> str:
        """Generate header/footer configuration."""
        return rf"""
% Header/Footer
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\textcolor{{fplpurple}}{{\textbf{{{team_name}}}}}}}
\fancyhead[R]{{\textcolor{{fplgray}}{{Season {season} | GW{self.gameweek}}}}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{2pt}}
\renewcommand{{\headrule}}{{\hbox to\headwidth{{\color{{fplgreen}}\leaders\hrule height \headrulewidth\hfill}}}}
\setlength{{\headheight}}{{24pt}}

\begin{{document}}
"""

    def generate_title_page(self, team_info: Dict, gw_history: List[Dict]) -> str:
        """Generate the title page."""
        team_name = team_info.get('team_name', 'Unknown')
        manager = team_info.get('manager_name', 'Unknown')
        total_points = team_info.get('overall_points', 0)
        overall_rank = team_info.get('overall_rank', 0)
        season = team_info.get('season', DEFAULT_SEASON)

        # Format rank with commas
        rank_str = f"{overall_rank:,}" if overall_rank else "N/A"

        return rf"""
% Title Page
\begin{{titlepage}}
    \centering
    \vspace*{{2cm}}

    {{\Huge\bfseries\color{{fplpurple}} Fantasy Premier League}}\\[0.5cm]
    {{\LARGE\color{{fplgreen}} Season Report {season}}}\\[2cm]

    \begin{{tikzpicture}}
        \fill[fplpurple] (0,0) circle (3cm);
        \node[white,font=\Huge\bfseries] at (0,0.5) {{{team_name[:2].upper()}}};
        \node[fplgreen,font=\large\bfseries] at (0,-0.5) {{{team_name[3:15] if len(team_name) > 3 else team_name}}};
    \end{{tikzpicture}}

    \vspace{{2cm}}

    {{\Large\color{{fpldark}} Manager: \textbf{{{manager}}}}}\\[0.5cm]
    {{\large\color{{fplgray}} Gameweek {self.gameweek} Report}}\\[2cm]

    \begin{{tikzpicture}}
        \node[draw=fplpurple,line width=2pt,rounded corners=10pt,inner sep=15pt,fill=fplpurple!5] {{
            \begin{{tabular}}{{ccc}}
                \textcolor{{fplpurple}}{{\Large\textbf{{{total_points}}}}} & \textcolor{{fplpurple}}{{\Large\textbf{{{rank_str}}}}} & \textcolor{{fplpurple}}{{\Large\textbf{{GW{self.gameweek}}}}}\\
                \textcolor{{fplgray}}{{Total Points}} & \textcolor{{fplgray}}{{Overall Rank}} & \textcolor{{fplgray}}{{Current GW}}
            \end{{tabular}}
        }};
    \end{{tikzpicture}}

    \vfill
    {{\small\color{{fplgray}} Report Generated: {datetime.now().strftime('%B %Y')}}}
\end{{titlepage}}
"""

    def generate_season_summary(self, team_info: Dict, gw_history: List[Dict],
                                 chips_used: List[Dict] = None, transfers: List[Dict] = None) -> str:
        """Generate the season summary section."""
        total_points = team_info.get('overall_points', 0)
        rank = team_info.get('overall_rank', 0)

        # Calculate stats from GW history
        if gw_history:
            points_list = [gw.get('points', 0) for gw in gw_history]
            ranks = [gw.get('overall_rank', 0) for gw in gw_history]
            bench_pts = [gw.get('points_on_bench', 0) for gw in gw_history]
            transfer_costs = [gw.get('event_transfers_cost', 0) for gw in gw_history]

            avg_pts = sum(points_list) / len(points_list) if points_list else 0
            best_pts = max(points_list) if points_list else 0
            worst_pts = min(points_list) if points_list else 0
            best_gw_idx = points_list.index(best_pts) + 1 if points_list else 0
            worst_gw_idx = points_list.index(worst_pts) + 1 if points_list else 0
            best_rank = min(ranks) if ranks else 0
            total_bench = sum(bench_pts)
            avg_bench = total_bench / len(bench_pts) if bench_pts else 0
            value = gw_history[-1].get('value', 1000) / 10 if gw_history else 100
            bank = gw_history[-1].get('bank', 0) / 10 if gw_history else 0
            total_hits = sum(transfer_costs)
        else:
            avg_pts = best_pts = worst_pts = best_rank = total_bench = avg_bench = 0
            best_gw_idx = worst_gw_idx = 0
            value = 100
            bank = 0
            total_hits = 0

        # Calculate chips and transfers
        chips_used = chips_used or []
        transfers = transfers or []
        chips_available = 5 - len(chips_used)
        
        # Count only regular transfers (exclude WC/FH transfers - those don't count against FTs)
        chip_gws = {c['event'] for c in chips_used if c.get('name') in ('wildcard', 'freehit')}
        total_transfers = len([t for t in transfers if t.get('event') not in chip_gws])

        # Calculate free transfers (2025-26 rules)
        # Rules:
        # - GW1: Unlimited transfers (pre-season)
        # - GW2+: Start with 1 FT, +1 per unused GW, cap at 5
        # - WC/FH: Don't consume FTs, but reset to 1 for NEXT GW
        #
        # We show FT balance going INTO the current GW (what user had available)
        
        if gw_history:
            # Create chip usage map
            chip_usage = {c['event']: c['name'] for c in chips_used} if chips_used else {}
            
            # Sort history by event
            sorted_history = sorted(gw_history, key=lambda x: x['event'])
            
            # Find the current GW (last in history)
            current_gw = sorted_history[-1]['event']
            
            # Calculate FT balance going INTO the current GW
            # We iterate from GW2 up to (but not including) current_gw
            current_simulated_ft = 1  # Start with 1 FT available for GW2
            
            if current_gw == 1:
                # GW1: Unlimited transfers
                free_transfers = "Unlimited"
            else:
                # Calculate FT accumulated up to the current GW
                for gw in range(2, current_gw):
                    # Find history for this GW
                    gw_data = next((g for g in sorted_history if g['event'] == gw), None)
                    
                    # Check if chip used in THIS GW
                    chip = chip_usage.get(gw)
                    
                    if chip in ['wildcard', 'freehit']:
                        # Chip used: FT resets to 1 for the following week
                        current_simulated_ft = 1
                    else:
                        # Regular GW: Deduct transfers used, then accumulate for next week
                        transfers_made = gw_data.get('event_transfers', 0) if gw_data else 0
                        
                        current_simulated_ft -= transfers_made
                        current_simulated_ft = max(0, current_simulated_ft)
                        
                        # Add 1 for next week, cap at 5
                        current_simulated_ft += 1
                        current_simulated_ft = min(5, current_simulated_ft)
                
                free_transfers = str(current_simulated_ft)
        else:
            # Pre-season
            free_transfers = "Unlimited"

        rank_str = f"{rank:,}" if rank else "N/A"
        best_rank_str = f"{best_rank:,}" if best_rank else "N/A"

        # Format chips used
        if chips_used:
            chips_str = ', '.join([f"{c.get('name', 'Unknown')} (GW{c.get('event', '?')})" for c in chips_used])
        else:
            chips_str = "None used"

        return rf"""
% Season Summary
\newpage
\section{{Season Summary}}

\begin{{center}}
\begin{{tikzpicture}}
    \node[draw=fplgreen,line width=2pt,rounded corners=5pt,inner sep=12pt,fill=fplgreen!10] {{
        \begin{{tabular}}{{cccc}}
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{total_points}}}}} &
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{rank_str}}}}} &
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{best_rank_str}}}}} &
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{len(gw_history)}}}}}\\[3pt]
            \textcolor{{fpldark}}{{Total Points}} &
            \textcolor{{fpldark}}{{Current Rank}} &
            \textcolor{{fpldark}}{{Best Rank}} &
            \textcolor{{fpldark}}{{GWs Played}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}

\vspace{{0.5cm}}

\subsection{{Key Statistics}}

\begin{{tabularx}}{{\textwidth}}{{X|X}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
Points per Gameweek (Average) & {avg_pts:.1f} \\
Best Gameweek Score & {best_pts} (GW{best_gw_idx}) \\
Worst Gameweek Score & {worst_pts} (GW{worst_gw_idx}) \\
Best Overall Rank & {best_rank_str} \\
Total Points on Bench & {total_bench} \\
Average Bench Points & {avg_bench:.1f} \\
Squad Value & \pounds{value:.1f}m \\
Bank Balance & \pounds{bank:.1f}m \\
\midrule
\textbf{{Transfers Made}} & {total_transfers} \\
\textbf{{Transfer Hits}} & -{total_hits} points \\
\textbf{{Free Transfers}} & {free_transfers} \\
\textbf{{Chips Available}} & {chips_available}/5 \\
\textbf{{Chips Used}} & {chips_str} \\
\bottomrule
\end{{tabularx}}
"""

    def generate_gw_performance_chart(self, gw_history: List[Dict]) -> str:
        """Generate the gameweek performance bar chart."""
        if not gw_history:
            return ""

        # Check if plot image exists
        if self.plot_dir:
            plot_path = self.plot_dir / "points_per_gw.png"
            if plot_path.exists():
                return rf"""
% Gameweek Performance Chart (Image)
\newpage
\section{{Gameweek Performance}}
\subsection{{Points per Gameweek}}
\begin{{center}}
    \includegraphics[width=\textwidth]{{{plot_path}}}
\end{{center}}
"""

        points_list = [gw.get('points', 0) for gw in gw_history]
        avg_pts = sum(points_list) / len(points_list) if points_list else 0

        # Generate coordinates
        coords = ' '.join([f"({i+1},{pts})" for i, pts in enumerate(points_list)])
        gw_ticks = ','.join([str(i+1) for i in range(len(points_list))])
        max_pts = max(points_list) + 15 if points_list else 100

        return rf"""
% Gameweek Performance Chart
\newpage
\section{{Gameweek Performance}}

\subsection{{Points per Gameweek}}

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\textwidth,
    height=7cm,
    ybar,
    bar width=12pt,
    xlabel={{Gameweek}},
    ylabel={{Points}},
    ymin=0,
    ymax={max_pts},
    xtick={{{gw_ticks}}},
    xticklabels={{{gw_ticks}}},
    enlarge x limits=0.04,
    grid=major,
    grid style={{dashed,gray!30}},
    legend style={{at={{(0.5,-0.15)}},anchor=north,legend columns=-1}},
]
\addplot[fill=fplpurple!70,draw=fplpurple,
    nodes near coords,
    nodes near coords style={{font=\scriptsize,color=black}},
    every node near coord/.append style={{anchor=south}}
] coordinates {{
    {coords}
}};
\addplot[red,line width=2pt,mark=none,sharp plot,forget plot] coordinates {{
    (1,{avg_pts:.1f}) ({len(points_list)},{avg_pts:.1f})
}};
\node[red,font=\small] at (axis cs:{len(points_list)+0.5},{avg_pts:.1f}) {{Avg: {avg_pts:.1f}}};
\legend{{Points}}
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}
"""

    def generate_player_points_breakdown(self, squad_analysis: List[Dict]) -> str:
        """Generate chart showing points per player in squad."""
        if not squad_analysis:
            return ""

        # Check if treemap exists
        if self.plot_dir:
            plot_path = self.plot_dir / "points_treemap.png"
            if plot_path.exists():
                return rf"""
% Player Points Breakdown (Treemap)
\subsection{{Points per Player Distribution}}
\begin{{center}}
    \includegraphics[width=\textwidth]{{{plot_path}}}
\end{{center}}
"""

        # Sort by total points descending
        sorted_players = sorted(
            squad_analysis,
            key=lambda p: p.get('raw_stats', {}).get('total_points', 0),
            reverse=True
        )

        # Build symbolic coords for horizontal bar
        player_entries = []
        y_labels = []
        max_pts = 0
        for i, player in enumerate(sorted_players):
            pts = player.get('raw_stats', {}).get('total_points', 0)
            name = player.get('name', 'Unknown')
            # Escape special LaTeX chars
            name = name.replace('_', r'\_').replace('&', r'\&')
            player_entries.append(f"({pts},{i})")
            y_labels.append(name)
            if pts > max_pts:
                max_pts = pts

        coords = ' '.join(player_entries)
        ytick_labels = ','.join(y_labels)
        yticks = ','.join([str(i) for i in range(len(sorted_players))])

        return rf"""
% Player Points Breakdown
\subsection{{Points per Player}}

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\textwidth,
    height=10cm,
    xbar,
    bar width=8pt,
    xlabel={{Total Points}},
    xmin=0,
    xmax={max_pts + 20},
    ytick={{{yticks}}},
    yticklabels={{{ytick_labels}}},
    y tick label style={{font=\small}},
    enlarge y limits=0.04,
    grid=major,
    grid style={{dashed,gray!30}},
    nodes near coords,
    nodes near coords style={{font=\tiny,color=black}},
]
\addplot[fill=fplpurple!70,draw=fplpurple] coordinates {{
    {coords}
}};
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}
"""

    def generate_position_breakdown(self, squad_analysis: List[Dict], season_history: List[Dict] = None) -> str:
        """Generate pie chart showing points distribution by position.
        
        Args:
            squad_analysis: Squad analysis data.
            season_history: Season history with squad data per GW (for accurate contributing points).
        """
        if not squad_analysis:
            return ""

        # Aggregate points by position
        position_points = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        position_labels = {'GKP': 'Goalkeeper', 'DEF': 'Defenders', 'MID': 'Midfielders', 'FWD': 'Forwards'}
        position_colors = {'GKP': 'fplpurple!60', 'DEF': 'blue!50', 'MID': 'fplgreen!60', 'FWD': 'fplpink!60'}

        # If season_history is available, calculate contributing points (only starting XI with captain multipliers)
        if season_history:
            # Calculate contributing points per player
            for gw_entry in season_history:
                squad = gw_entry.get('squad', [])
                for player in squad:
                    # Only count if in starting XI (position_in_squad <= 11)
                    position_in_squad = player.get('position_in_squad', 0)
                    if position_in_squad <= 11:
                        base_points = player.get('stats', {}).get('event_points', 0) or 0
                        
                        # Apply captain multiplier (2 for captain, 3 for triple captain)
                        multiplier = player.get('multiplier', 1)
                        points = base_points * multiplier
                        
                        # Get player's position directly from season_history (includes historical players)
                        pos = player.get('position', 'UNK')
                        if pos in position_points:
                            position_points[pos] += points
        else:
            # Fallback to raw stats if season_history not available
            for player in squad_analysis:
                pos = player.get('position', 'UNK')
                pts = player.get('raw_stats', {}).get('total_points', 0)
                if pos in position_points:
                    position_points[pos] += pts

        total_pts = sum(position_points.values()) or 1

        # Build pie chart data
        pie_data = []
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pts = position_points[pos]
            pct = pts / total_pts * 100
            label = position_labels[pos]
            color = position_colors[pos]
            pie_data.append(f"{pct}/{color}/{label} ({pts} pts)")

        pie_str = ', '.join(pie_data)

        # Also create a summary table
        table_rows = []
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pts = position_points[pos]
            pct = pts / total_pts * 100
            table_rows.append(rf"{position_labels[pos]} & {pts} & {pct:.1f}\%")

        table_content = r' \\'.join(table_rows)

        return rf"""
% Position Points Breakdown
\subsection{{Points by Position}}

\begin{{center}}
\begin{{minipage}}{{0.45\textwidth}}
\begin{{tikzpicture}}
\pie[
    text=legend,
    radius=2.5,
    color={{fplpurple!60, blue!50, fplgreen!60, fplpink!60}},
    explode={{0.05, 0.05, 0.05, 0.05}}
]{{
    {position_points['GKP']/total_pts*100:.1f}/GKP,
    {position_points['DEF']/total_pts*100:.1f}/DEF,
    {position_points['MID']/total_pts*100:.1f}/MID,
    {position_points['FWD']/total_pts*100:.1f}/FWD
}}
\end{{tikzpicture}}
\end{{minipage}}
\hfill
\begin{{minipage}}{{0.45\textwidth}}
\begin{{tabular}}{{l|r|r}}
\toprule
\textbf{{Position}} & \textbf{{Points}} & \textbf{{Share}} \\
\midrule
{table_content} \\
\midrule
\textbf{{Total}} & \textbf{{{total_pts}}} & \textbf{{100\%}} \\
\bottomrule
\end{{tabular}}
\end{{minipage}}
\end{{center}}
"""

    def generate_contribution_chart(self) -> str:
        """Generate player contribution heatmap."""
        if self.plot_dir:
            plot_path = self.plot_dir / "player_contribution.png"
            if plot_path.exists():
                return rf"""
% Player Contribution Heatmap
\newpage
\section{{Player Contribution Analysis}}
\begin{{center}}
    \includegraphics[width=\textwidth]{{{plot_path}}}
\end{{center}}
"""
        return ""

    def generate_hindsight_analysis(self) -> str:
        """Generate hindsight fixture analysis chart."""
        if self.plot_dir:
            plot_path = self.plot_dir / "hindsight_fixture_analysis.png"
            if plot_path.exists():
                return rf"""
% Hindsight Fixture Analysis
\subsection{{Hindsight Fixture Analysis}}

\textit{{This analysis shows fixture difficulty (with opponents) alongside actual points scored.
Use it to evaluate how well fixture difficulty predicted performance and identify which players
over/under-performed relative to their schedule.}}

\begin{{center}}
    \includegraphics[width=\textwidth]{{{plot_path}}}
\end{{center}}
"""
        return ""

    def generate_transfer_history(self, transfers: List[Dict]) -> str:
        """Generate detailed transfer history table."""
        if not transfers:
            return r"""
\newpage
\section{Transfer History}
\textit{No transfers made this season.}
"""

        sections = []
        
        # Add Transfer Matrix if available
        if self.plot_dir:
            plot_path = self.plot_dir / "transfer_matrix.png"
            if plot_path.exists():
                sections.append(r"\newpage")
                sections.append(r"\section{Transfer History}")
                sections.append(r"\subsection{Transfer Activity Matrix}")
                sections.append(rf"""
\begin{{center}}
    \includegraphics[width=0.8\textwidth]{{{plot_path}}}
\end{{center}}
""")
                sections.append(r"\subsection{Detailed Log}")
            else:
                sections.append(r"\newpage")
                sections.append(r"\section{Transfer History}")
        else:
             sections.append(r"\newpage")
             sections.append(r"\section{Transfer History}")

        # Group transfers by gameweek
        gw_transfers = {}
        for t in transfers:
            gw = t.get('event', 0)
            if gw not in gw_transfers:
                gw_transfers[gw] = []
            gw_transfers[gw].append(t)

        # Sort by gameweek descending (most recent first)
        sorted_gws = sorted(gw_transfers.keys(), reverse=True)

        if not sections or sections[-1] != r"\subsection{Detailed Log}":
             sections.append(rf"\textit{{{len(transfers)} transfers made this season.}}")
             sections.append(r"\vspace{0.5cm}")

        # Create table
        sections.append(r"""
\begin{center}
\small
\begin{tabular}{c|l|l|r|r}
\toprule
\textbf{GW} & \textbf{Player In} & \textbf{Player Out} & \textbf{In Cost} & \textbf{Out Cost} \\
\midrule
""")

        row_count = 0
        for gw in sorted_gws:
            for t in gw_transfers[gw]:
                in_name = t.get('element_in_name', 'Unknown')
                out_name = t.get('element_out_name', 'Unknown')
                in_cost = t.get('element_in_cost_m', 0)
                out_cost = t.get('element_out_cost_m', 0)

                # Escape LaTeX special chars
                in_name = in_name.replace('_', r'\_').replace('&', r'\&')
                out_name = out_name.replace('_', r'\_').replace('&', r'\&')
                
                # Check for None costs
                in_cost_str = f"\\pounds{in_cost:.1f}m" if in_cost is not None else "N/A"
                out_cost_str = f"\\pounds{out_cost:.1f}m" if out_cost is not None else "N/A"

                sections.append(rf"{gw} & {in_name} & {out_name} & {in_cost_str} & {out_cost_str} \\")
                row_count += 1

                # Limit to 25 rows to avoid overly long tables
                if row_count >= 25:
                    break
            if row_count >= 25:
                sections.append(r"\multicolumn{5}{c}{\textit{... and more transfers}} \\")
                break

        sections.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")

        return '\n'.join(sections)

    def generate_rank_progression(self, gw_history: List[Dict]) -> str:
        """Generate overall rank progression chart."""
        if not gw_history:
            return ""

        ranks = [(gw.get('overall_rank', 0) or 0) / 1_000_000 for gw in gw_history]
        coords = ' '.join([f"({i+1},{r:.2f})" for i, r in enumerate(ranks)])
        gw_ticks = ','.join([str(i+1) for i in range(len(ranks))])
        max_rank = max(ranks) + 1 if ranks else 10
        best_rank = min(ranks) if ranks else 0
        best_gw = ranks.index(best_rank) + 1 if ranks else 1

        return rf"""
\subsection{{Overall Rank Progression}}

\begin{{center}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=\textwidth,
    height=7cm,
    xlabel={{Gameweek}},
    ylabel={{Overall Rank (millions)}},
    ymin=0,
    ymax={max_rank:.1f},
    y dir=reverse,
    xtick={{{gw_ticks}}},
    grid=major,
    grid style={{dashed,gray!30}},
    mark=*,
    mark options={{fill=fplgreen,draw=fplpurple}},
    line width=2pt,
    color=fplpurple,
]
\addplot coordinates {{
    {coords}
}};
\node[pin={{[pin edge={{fplgreen,thick}}]90:\textcolor{{fplgreen}}{{\textbf{{Best: {best_rank:.2f}M}}}}}}] at (axis cs:{best_gw},{best_rank:.2f}) {{}};
\end{{axis}}
\end{{tikzpicture}}
\end{{center}}
"""

    def generate_formation_diagram(self, squad: List[Dict]) -> str:
        """Generate the tactical formation diagram."""
        # Sort squad by position in squad (1-11 are starters)
        starters = [p for p in squad if p.get('position_in_squad', 0) <= 11]
        bench = [p for p in squad if p.get('position_in_squad', 0) > 11]

        # Group by position
        gk = [p for p in starters if p.get('position') == 'GKP']
        defs = [p for p in starters if p.get('position') == 'DEF']
        mids = [p for p in starters if p.get('position') == 'MID']
        fwds = [p for p in starters if p.get('position') == 'FWD']

        formation = f"{len(defs)}-{len(mids)}-{len(fwds)}"

        def player_node(player: Dict, x: float, y: float) -> str:
            name = player.get('name', 'Unknown')
            if player.get('is_captain'):
                return rf"\node[captain] at ({x},{y}) {{\textbf{{{name} (C)}}}};"
            elif player.get('is_vice_captain'):
                return rf"\node[vice] at ({x},{y}) {{{name} (VC)}};"
            elif player.get('position') == 'GKP':
                return rf"\node[player,fill=fplpurple!15] at ({x},{y}) {{{name}}};"
            else:
                return rf"\node[player] at ({x},{y}) {{{name}}};"

        # Build nodes
        nodes = []

        # Forwards
        fwd_positions = self._distribute_positions(len(fwds), 7.8)
        for i, p in enumerate(fwds):
            nodes.append(player_node(p, fwd_positions[i], 7.8))

        # Midfielders
        mid_positions = self._distribute_positions(len(mids), 5.5)
        for i, p in enumerate(mids):
            y_offset = 0.3 if i in [0, len(mids)-1] else 0  # Wide players slightly higher
            nodes.append(player_node(p, mid_positions[i], 5.5 + y_offset))

        # Defenders
        def_positions = self._distribute_positions(len(defs), 2.3)
        for i, p in enumerate(defs):
            nodes.append(player_node(p, def_positions[i], 2.3))

        # Goalkeeper
        if gk:
            nodes.append(player_node(gk[0], 0, 0.3))

        # Bench
        bench_nodes = []
        bench_positions = self._distribute_positions(len(bench), -3.0)
        for i, p in enumerate(bench):
            name = p.get('name', 'Unknown')
            bench_nodes.append(rf"\node[bench] at ({bench_positions[i]},-3) {{{name}}};")

        nodes_str = '\n'.join(nodes)
        bench_str = '\n'.join(bench_nodes)

        return rf"""
% Current Squad
\newpage
\section{{Current Squad (GW{self.gameweek})}}

\begin{{center}}
\begin{{tikzpicture}}[
    player/.style={{draw=fplpurple,rounded corners=5pt,fill=white,minimum width=2.2cm,minimum height=0.8cm,font=\small,drop shadow={{shadow xshift=0.5pt,shadow yshift=-0.5pt,opacity=0.3}}}},
    captain/.style={{player,fill=fplgreen!40,line width=2pt,draw=fplpurple}},
    vice/.style={{player,fill=yellow!30,line width=1.5pt}},
    bench/.style={{draw=gray,rounded corners=5pt,fill=gray!15,minimum width=2cm,minimum height=0.7cm,font=\small}}
]

% Pitch background
\fill[green!30] (-6,-1) rectangle (6,10);
\fill[green!25] (-6,4.5) rectangle (6,10);

% Pitch markings
\draw[white,line width=3pt] (-6,4.5) -- (6,4.5);
\draw[white,line width=2pt] (0,4.5) circle (1.2);
\fill[white] (0,4.5) circle (0.1);
\draw[white,line width=2pt] (-3,10) -- (-3,8) -- (3,8) -- (3,10);
\draw[white,line width=1.5pt] (-1.5,10) -- (-1.5,9) -- (1.5,9) -- (1.5,10);
\draw[white,line width=2pt] (-3,-1) -- (-3,1) -- (3,1) -- (3,-1);

% Formation label
\node[font=\Large\bfseries,color=fplpurple,fill=white,rounded corners=3pt,inner sep=5pt] at (0,9.3) {{{formation}}};

% Players
{nodes_str}

% Bench
\fill[gray!10] (-5.5,-2) rectangle (5.5,-3.5);
\draw[gray,line width=1pt] (-5.5,-2) rectangle (5.5,-3.5);
\node[font=\bfseries,color=fplgray] at (0,-2.3) {{SUBSTITUTES}};
{bench_str}

\end{{tikzpicture}}
\end{{center}}
"""

    def _distribute_positions(self, count: int, y: float) -> List[float]:
        """Distribute player positions evenly across the pitch width."""
        if count == 0:
            return []
        if count == 1:
            return [0]
        if count == 2:
            return [-2.5, 2.5]
        if count == 3:
            return [-3.5, 0, 3.5]
        if count == 4:
            return [-4, -1.5, 1.5, 4]
        if count == 5:
            return [-4.2, -1.8, 0, 1.8, 4.2]
        return [i * 2 - count for i in range(count)]

    def generate_player_deep_dives(self, squad_analysis: List[Dict]) -> str:
        """Generate deep dive analysis for each player in a comprehensive table."""
        if not squad_analysis:
            return ""
        
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\section{Player Deep Dive Analysis}")
        sections.append(r"""
\small

This section provides comprehensive metrics for each player in your squad, organized by position.
The table includes performance stats, expected metrics, ICT analysis, and peer comparisons.

\vspace{0.3cm}
""")
        
        # Group players by position
        players_by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for player in squad_analysis:
            pos = player.get('position', 'UNK')
            if pos in players_by_pos:
                players_by_pos[pos].append(player)
        
        # Generate table for each position group
        for pos, pos_name in [('GKP', 'Goalkeepers'), ('DEF', 'Defenders'), 
                              ('MID', 'Midfielders'), ('FWD', 'Forwards')]:
            players = players_by_pos[pos]
            if not players:
                continue
            
            sections.append(self._generate_position_table(players, pos_name))
        
        # Add advanced finishing & creativity visualizations section
        sections.append(self._generate_advanced_finishing_creativity_section())
        
        return '\n'.join(sections)

    def _generate_position_table(self, players: List[Dict], position_name: str) -> str:
        """Generate a comprehensive metrics table for players of a position."""
        sections = []
        sections.append(rf"\subsection{{{position_name}}}")
        
        # Build table rows
        rows = []
        for player in players:
            name = player.get('name', 'Unknown')
            team = player.get('team', 'UNK')
            price = player.get('price', 0)
            ownership = player.get('ownership', 0)
            
            form = player.get('form_analysis', {})
            ict = player.get('ict_analysis', {})
            exp = player.get('expected_vs_actual', {})
            peer = player.get('peer_comparison', {})
            stats = player.get('raw_stats', {})
            
            # Form trend indicator
            trend = form.get('trend', 'stable')
            trend_color = 'rising' if trend == 'rising' else ('falling' if trend == 'falling' else 'stable')
            trend_symbol = r'$\uparrow$' if trend == 'rising' else (r'$\downarrow$' if trend == 'falling' else r'$\rightarrow$')
            
            # Peer percentile color
            overall_pct = peer.get('overall_rating', 50)
            pct_color = 'fplgreen' if overall_pct >= 75 else ('gold' if overall_pct >= 50 else ('orange' if overall_pct >= 25 else 'fplpink'))
            
            # Escape LaTeX chars
            name = name.replace('_', r'\_').replace('&', r'\&')
            team = str(team).replace('_', r'\_').replace('&', r'\&')
            
            # Get xG/xA difference indicators
            xg = exp.get('expected_goals', 0)
            actual_g = exp.get('actual_goals', 0)
            xa = exp.get('expected_assists', 0)
            actual_a = exp.get('actual_assists', 0)
            
            # Calculate over/under performance
            g_diff = actual_g - xg if xg else 0
            a_diff = actual_a - xa if xa else 0
            
            g_indicator = rf"\textcolor{{fplgreen}}{{+{g_diff:.1f}}}" if g_diff > 0.1 else (rf"\textcolor{{fplpink}}{{{g_diff:.1f}}}" if g_diff < -0.1 else "-")
            a_indicator = rf"\textcolor{{fplgreen}}{{+{a_diff:.1f}}}" if a_diff > 0.1 else (rf"\textcolor{{fplpink}}{{{a_diff:.1f}}}" if a_diff < -0.1 else "-")
            
            rows.append(rf"""
{name} & {team} & \pounds{price:.1f}m & {ownership:.1f}\% & {stats.get('total_points', 0)} & {exp.get('points_per_game', 0):.1f} & \textcolor{{{trend_color}}}{{{trend_symbol}}} {form.get('average', 0):.1f} & {stats.get('minutes', 0)} & {stats.get('bonus', 0)} & {ict.get('ict_index', 0):.1f} & {xg:.1f} ({g_indicator}) & {xa:.1f} ({a_indicator}) & \cellcolor{{{pct_color}!30}}{overall_pct:.0f}\% \\""")
        
        table_content = '\n'.join(rows)
        
        sections.append(rf"""
{{\footnotesize
\begin{{center}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l|l|r|r|r|r|c|r|r|r|c|c|c}}
\toprule
\textbf{{Player}} & \textbf{{Team}} & \textbf{{Price}} & \textbf{{Own\%}} & \textbf{{Pts}} & \textbf{{PPG}} & \textbf{{Form}} & \textbf{{Mins}} & \textbf{{Bonus}} & \textbf{{ICT}} & \textbf{{xG (vs Act)}} & \textbf{{xA (vs Act)}} & \textbf{{Peer\%}} \\
\midrule
{table_content}
\bottomrule
\end{{tabular}}
}}%
\end{{center}}
}}

\vspace{{0.5cm}}
""")
        
        return '\n'.join(sections)

    def _generate_advanced_finishing_creativity_section(self) -> str:
        """Generate advanced finishing and creativity visualization section.
        
        Includes Clinical/Wasteful goals, Clutch/Frustrated assists, and Usage vs Output plots.
        """
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\subsection{Advanced Finishing \& Creativity Analysis}")
        sections.append(r"""
\small

This section provides visual analysis of your squad's finishing efficiency and creative output
compared to league-wide performance. Charts show both single-gameweek and season-cumulative metrics,
with your squad players highlighted in bold.

\vspace{0.3cm}
""")
        
        # Clinical vs Wasteful Goals
        sections.append(r"\subsubsection{Clinical vs Wasteful: Goals}")
        sections.append(r"""
\small
Green bars indicate clinical finishers (Goals > xG), while red bars show wasteful players (Goals < xG).
Your squad players are highlighted in bold.

\vspace{0.3cm}
""")
        
        # Check if plots exist
        clinical_gw_path = self.plot_dir / 'clinical_wasteful_gw.png'
        clinical_season_path = self.plot_dir / 'clinical_wasteful_season.png'
        
        if clinical_gw_path.exists() and clinical_season_path.exists():
            sections.append(r"""
\begin{center}
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{plots/clinical_wasteful_gw.png}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{plots/clinical_wasteful_season.png}
\end{minipage}
\end{center}

\vspace{0.5cm}
""")
        
        # Clutch vs Frustrated Assists
        sections.append(r"\subsubsection{Clutch vs Frustrated: Assists}")
        sections.append(r"""
\small
Green bars indicate clutch playmakers (Assists > xA), while red bars show frustrated creators (Assists < xA).

\vspace{0.3cm}
""")
        
        clutch_gw_path = self.plot_dir / 'clutch_frustrated_gw.png'
        clutch_season_path = self.plot_dir / 'clutch_frustrated_season.png'
        
        if clutch_gw_path.exists() and clutch_season_path.exists():
            sections.append(r"""
\begin{center}
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{plots/clutch_frustrated_gw.png}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{plots/clutch_frustrated_season.png}
\end{minipage}
\end{center}

\vspace{0.5cm}
""")
        
        # Usage vs Output Scatter
        sections.append(r"\subsubsection{Usage vs Output Analysis}")
        
        # Info Box (How To Read)
        sections.append(r"""
\begin{center}
\fbox{
\begin{minipage}{0.9\textwidth}
\textbf{HOW TO READ THIS CHART}
\small
\begin{itemize}
    \item \textbf{Usage (X):} Shots + Box Touches per 90
    \item \textbf{Output (Y):} Goals + Assists
    \item \textbf{Bubble:} xGI (expected involvement)
    \item \textbf{Color:} DEF/MID/FWD (outline + bold = your squad)
    \item \textbf{GW Range:} Shown in title (e.g. GW1-17, GW12-17 last 5)
    \item \textbf{Quadrants:} ELITE (high/high); VOLUME (high use, low returns - buy); CLINICAL (low use, high returns - sell watch); AVOID (low/low)
\end{itemize}
\end{minipage}
}
\end{center}
""")
        
        usage_league_path = self.plot_dir / 'usage_output_scatter.png'
        usage_squad_path = self.plot_dir / 'usage_output_scatter_squad.png'
        usage_recent_path = self.plot_dir / 'usage_output_scatter_last5.png'
        usage_recent_squad_path = self.plot_dir / 'usage_output_scatter_last5_squad.png'
        usage_summary_path = self.plot_dir / 'usage_output_summary.json'
        
        if usage_league_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.85\linewidth]{plots/usage_output_scatter.png} \\
\textit{League-wide Context (Squad highlighted in bold)}
\end{center}
""")

        if usage_squad_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.85\linewidth]{plots/usage_output_scatter_squad.png} \\
\textit{Squad-only View}
\end{center}
""")

        if usage_recent_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.85\linewidth]{plots/usage_output_scatter_last5.png} \\
\textit{League-wide Recent Form (Last 5 GWs, Top 25 attackers)}
\end{center}
""")

        if usage_recent_squad_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.85\linewidth]{plots/usage_output_scatter_last5_squad.png} \\
\textit{Squad-only Recent Form (Last 5 GWs)}
\end{center}
""")

        # Summary tables + insights (if available)
        if usage_summary_path.exists():
            try:
                summary_data = json.loads(usage_summary_path.read_text())
            except Exception:
                summary_data = {}

            def render_table(key: str, title: str):
                payload = summary_data.get(key)
                if not payload:
                    return ""
                rows = payload.get('rows', [])
                if not rows:
                    return ""
                table_lines = [r"\begin{center}",
                               r"\begin{tabular}{l c c c c c}",
                               r"\toprule",
                               r"\textbf{Player} & \textbf{Pos} & \textbf{Use/90} & \textbf{G{+}A} & \textbf{xGI} & \textbf{Pts} \\",
                               r"\midrule"]
                for row in rows:
                    table_lines.append(
                        rf"{row.get('name','-')} & {row.get('pos','-')} & {row.get('usage_per_90',0)} & {row.get('ga',0)} & {row.get('xgi',0)} & {row.get('pts',0)} \\"
                    )
                table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\vspace{6pt}"])
                insights = payload.get('insights', [])
                insight_lines = []
                if insights:
                    insight_lines.append(r"\begin{itemize}")
                    for item in insights:
                        insight_lines.append(rf"    \item {item}")
                    insight_lines.append(r"\end{itemize}")
                range_text = payload.get('range_label', '')
                header_line = rf"\textbf{{{title}}} ({range_text})"
                return "\n".join([header_line, *table_lines, *insight_lines])

            season_block = render_table('season', 'Top Attackers - Season')
            recent_block = render_table('last5', 'Top Attackers - Last 5 GWs')

            if season_block or recent_block:
                sections.append(r"\vspace{0.5cm}")
                sections.append(r"\subsubsection{Usage vs Output Tables}")
                
                # Place tables side-by-side if both exist
                if season_block and recent_block:
                    sections.append(r"\noindent")
                    sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                    sections.append(season_block)
                    sections.append(r"\end{minipage}")
                    sections.append(r"\hfill")
                    sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                    sections.append(recent_block)
                    sections.append(r"\end{minipage}")
                else:
                    # Single table - use full width
                    if season_block:
                        sections.append(season_block)
                    if recent_block:
                        sections.append(recent_block)

            def render_categories(key: str, title: str):
                payload = summary_data.get(key)
                if not payload:
                    return []
                categories = payload.get('categories', {})
                if not categories:
                    return []
                tables = []
                def render_single(cat_key, cat_label):
                    entries = categories.get(cat_key, [])
                    if not entries:
                        return ""
                    # Use smaller font and tighter spacing for side-by-side layout
                    lines = [
                        rf"\textbf{{{cat_label}}} ({payload.get('range_label','')})",
                        r"\begin{center}",
                        r"{\scriptsize",
                        r"\setlength{\tabcolsep}{2pt}",
                        r"\renewcommand{\arraystretch}{0.95}",
                        r"\resizebox{\linewidth}{!}{%",
                        r"\begin{tabular}{l c c c c c}",
                        r"\toprule",
                        r"\textbf{Player} & \textbf{Pos} & \textbf{Use/90} & \textbf{G{+}A} & \textbf{xGI} & \textbf{Pts} \\",
                        r"\midrule",
                    ]
                    for e in entries:
                        lines.append(
                            rf"{e.get('name','-')} & {e.get('pos','-')} & {e.get('usage_per_90',0)} & {e.get('ga',0)} & {e.get('xgi',0)} & {e.get('pts',0)} \\"
                        )
                    lines.extend([r"\bottomrule", r"\end{tabular}", r"}%", r"}", r"\end{center}"])
                    return "\n".join(lines)
                for cat_key, cat_label in [('elite', 'ELITE (Keep)'), ('volume', 'VOLUME (Buy)'), ('clinical', 'CLINICAL (Sell Watch)')]:
                    block = render_single(cat_key, cat_label)
                    if block:
                        tables.append(block)
                return tables  # Return list instead of joined string

            season_cat = render_categories('season', 'Usage Quadrants - Season')
            recent_cat = render_categories('last5', 'Usage Quadrants - Last 5 GWs')

            if season_cat or recent_cat:
                sections.append(r"\vspace{0.4cm}")
                sections.append(r"\subsubsection{Usage vs Output Quadrants (Tables)}")
                
                # Helper function to layout three tables side-by-side
                def layout_three_tables(tables_list):
                    if not tables_list:
                        return
                    if len(tables_list) == 3:
                        # All three tables - place side-by-side
                        sections.append(r"\noindent")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[0])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[1])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[2])
                        sections.append(r"\end{minipage}")
                    elif len(tables_list) == 2:
                        # Two tables - place side-by-side with more width
                        sections.append(r"\noindent")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(tables_list[0])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(tables_list[1])
                        sections.append(r"\end{minipage}")
                    else:
                        # Single table - use full width
                        sections.append(tables_list[0])
                    sections.append(r"\vspace{0.5cm}")
                
                if season_cat:
                    layout_three_tables(season_cat)
                if recent_cat:
                    layout_three_tables(recent_cat)
        
        # Defensive Value Charts Section
        sections.append(r"\newpage")
        sections.append(r"\subsection{Defensive Value Charts}")
        
        # Info Box (How To Read)
        sections.append(r"""
\begin{tcolorbox}[colback=white,colframe=fplgreen!70!black,title=\textbf{How To Read This Chart},fonttitle=\small,boxrule=0.5pt,left=5pt,right=5pt,top=3pt,bottom=3pt]
\small
\begin{itemize}[leftmargin=*,nosep]
\item[$\rightarrow$] \textbf{X-Axis (Defensive Actions):} Tackles + Interceptions + Clearances + Blocks per 90 mins
\item[$\rightarrow$] \textbf{Y-Axis (Points):} Total FPL points over the period
\item[$\rightarrow$] \textbf{Bubble Size:} Price efficiency (pts/\pounds m) - bigger = better value
\item[$\rightarrow$] \textbf{Top-Right (ELITE):} High defensive activity AND high points - the complete defender package
\item[$\rightarrow$] \textbf{Bottom-Right (VOLUME):} High activity but low points - buy signal if clean sheets improve
\item[$\rightarrow$] \textbf{Top-Left (CS MERCHANTS):} Low activity but high points - riding clean sheets, fixture dependent
\end{itemize}
\end{tcolorbox}
\vspace{0.5cm}
""")
        
        # Recent Form (Last 5 Games)
        sections.append(r"\subsubsection{Last 5 Games (Recent Form)}")
        if (self.plot_dir / 'defensive_value_scatter_last5.png').exists():
            sections.append(rf"""
\begin{{center}}
\includegraphics[width=0.95\textwidth]{{{self.plot_dir}/defensive_value_scatter_last5.png}}
\end{{center}}
\vspace{{0.3cm}}
""")
        else:
            sections.append(r"\textit{Defensive value plot (recent form) not available.}")
        
        # Full Season
        sections.append(r"\subsubsection{Full Season (Baseline)}")
        if (self.plot_dir / 'defensive_value_scatter.png').exists():
            sections.append(rf"""
\begin{{center}}
\includegraphics[width=0.95\textwidth]{{{self.plot_dir}/defensive_value_scatter.png}}
\end{{center}}
""")
        else:
            sections.append(r"\textit{Defensive value plot (season) not available.}")
        
        # Defensive Value Tables (similar to attacker tables)
        def_summary_path = self.plot_dir / 'defensive_value_summary.json'
        if def_summary_path.exists():
            try:
                def_summary = json.loads(def_summary_path.read_text())
                
                # Render top defenders table
                def render_def_table(key, label):
                    data = def_summary.get(key, {})
                    rows = data.get('rows', [])
                    if not rows:
                        return ""
                    table_lines = [
                        rf"\textbf{{{label}}} ({data.get('range_label', '')})",
                        r"\begin{center}",
                        r"\begin{tabular}{l c c c c c}",
                        r"\toprule",
                        r"\textbf{Player} & \textbf{Def/90} & \textbf{CS} & \textbf{CS\%} & \textbf{Pts/\pounds} & \textbf{Pts} \\",
                        r"\midrule"
                    ]
                    for row in rows[:10]:
                        table_lines.append(
                            rf"{row.get('name', '-')} & {row.get('def_per_90', 0)} & {row.get('cs', 0)} & {row.get('cs_pct', 0)}\% & {row.get('pts_per_m', 0)} & {row.get('pts', 0)} \\"
                        )
                    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
                    return '\n'.join(table_lines)
                
                # Render category tables
                def render_def_categories(key, label):
                    data = def_summary.get(key, {})
                    cats = data.get('categories', {})
                    if not cats:
                        return []
                    
                    cat_labels = {
                        'elite': ('ELITE (Keep)', 'High Activity + High Points'),
                        'volume': ('VOLUME (Buy Signal)', 'High Activity, Low Points'),
                        'cs_merchants': ('CS MERCHANTS (Fixture Watch)', 'Low Activity, High Points'),
                    }
                    
                    range_label = data.get('range_label', '')
                    tables = []
                    
                    for cat_key, (cat_label, cat_desc) in cat_labels.items():
                        entries = cats.get(cat_key, [])
                        if not entries:
                            continue
                        lines = [
                            rf"\textbf{{{cat_label}}} ({range_label})",
                            r"\begin{center}",
                            r"{\scriptsize",
                            r"\setlength{\tabcolsep}{2pt}",
                            r"\renewcommand{\arraystretch}{0.95}",
                            r"\resizebox{\linewidth}{!}{%",
                            r"\begin{tabular}{l c c c c}",
                            r"\toprule",
                            r"\textbf{Player} & \textbf{Def/90} & \textbf{CS} & \textbf{CS\%} & \textbf{Pts} \\",
                            r"\midrule",
                        ]
                        for e in entries[:8]:
                            lines.append(
                                rf"{e.get('name', '-')} & {e.get('def_per_90', 0)} & {e.get('cs', 0)} & {e.get('cs_pct', 0)}\% & {e.get('pts', 0)} \\"
                            )
                        lines.extend([r"\bottomrule", r"\end{tabular}", r"}%", r"}", r"\end{center}"])
                        tables.append('\n'.join(lines))
                    
                    return tables  # Return list instead of joined string
                
                # Top Defenders Tables
                season_table = render_def_table('season', 'Top Defenders - Season')
                recent_table = render_def_table('last5', 'Top Defenders - Last 5 GWs')
                
                if season_table or recent_table:
                    sections.append(r"\vspace{0.5cm}")
                    sections.append(r"\subsubsection{Top Defenders Tables}")
                    
                    # Place tables side-by-side if both exist
                    if season_table and recent_table:
                        sections.append(r"\noindent")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(season_table)
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(recent_table)
                        sections.append(r"\end{minipage}")
                    else:
                        # Single table - use full width
                        if recent_table:
                            sections.append(recent_table)
                        if season_table:
                            sections.append(season_table)
                
                # Category Tables
                season_cat = render_def_categories('season', 'Defensive Quadrants - Season')
                recent_cat = render_def_categories('last5', 'Defensive Quadrants - Last 5 GWs')
                
                if season_cat or recent_cat:
                    sections.append(r"\vspace{0.4cm}")
                    sections.append(r"\subsubsection{Defensive Value Quadrants (Tables)}")
                    
                    # Helper function to layout three tables side-by-side
                    def layout_def_tables(tables_list):
                        if not tables_list:
                            return
                        if len(tables_list) == 3:
                            # All three tables - place side-by-side
                            sections.append(r"\noindent")
                            sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                            sections.append(tables_list[0])
                            sections.append(r"\end{minipage}")
                            sections.append(r"\hfill")
                            sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                            sections.append(tables_list[1])
                            sections.append(r"\end{minipage}")
                            sections.append(r"\hfill")
                            sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                            sections.append(tables_list[2])
                            sections.append(r"\end{minipage}")
                        elif len(tables_list) == 2:
                            # Two tables - place side-by-side with more width
                            sections.append(r"\noindent")
                            sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                            sections.append(tables_list[0])
                            sections.append(r"\end{minipage}")
                            sections.append(r"\hfill")
                            sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                            sections.append(tables_list[1])
                            sections.append(r"\end{minipage}")
                        else:
                            # Single table - use full width
                            sections.append(tables_list[0])
                        sections.append(r"\vspace{0.5cm}")
                    
                    if season_cat:
                        layout_def_tables(season_cat)
                    if recent_cat:
                        layout_def_tables(recent_cat)
                        
            except Exception as e:
                sections.append(rf"\textit{{Could not load defensive summary data: {str(e)[:50]}}}")
        
        # Goalkeeper Shot-Stopping Charts Section
        sections.append(r"\newpage")
        sections.append(r"\subsection{Goalkeeper Shot-Stopping Charts}")
        
        # Info Box (How To Read)
        sections.append(r"""
\begin{tcolorbox}[colback=white,colframe=fplgreen!70!black,title=\textbf{How To Read This Chart},fonttitle=\small,boxrule=0.5pt,left=5pt,right=5pt,top=3pt,bottom=3pt]
\small
\begin{itemize}[leftmargin=*,nosep]
\item[$\rightarrow$] \textbf{X-Axis (Goals Prevented):} xGOT faced minus Goals Conceded - positive = outperforming expected
\item[$\rightarrow$] \textbf{Y-Axis (Points):} Total FPL points over the period
\item[$\rightarrow$] \textbf{Bubble Size:} Clean sheets - bigger = more clean sheets
\item[$\rightarrow$] \textbf{Top-Right (ELITE):} Making saves AND getting points - the premium GK picks
\item[$\rightarrow$] \textbf{Bottom-Right (UNLUCKY):} Good shot stopping but low points - could bounce back with better fixtures
\item[$\rightarrow$] \textbf{Top-Left (PROTECTED):} Low shots faced, high points - riding good defence, fixture dependent
\end{itemize}
\end{tcolorbox}
\vspace{0.5cm}
""")
        
        gk_recent_path = self.plot_dir / 'goalkeeper_value_scatter_last5.png'
        gk_season_path = self.plot_dir / 'goalkeeper_value_scatter.png'
        
        # Recent Form (Last 5 Games)
        sections.append(r"\subsubsection{Last 5 Games (Recent Form)}")
        if gk_recent_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.95\textwidth]{plots/goalkeeper_value_scatter_last5.png}
\end{center}
\vspace{0.3cm}
""")
        else:
            sections.append(r"\textit{Goalkeeper plot (recent form) not available.}")
        
        # Full Season
        sections.append(r"\subsubsection{Full Season (Baseline)}")
        if gk_season_path.exists():
            sections.append(r"""
\begin{center}
\includegraphics[width=0.95\textwidth]{plots/goalkeeper_value_scatter.png}
\end{center}
""")
        else:
            sections.append(r"\textit{Goalkeeper plot (season) not available.}")

        # Goalkeeper tables (if summary available)
        gk_summary_path = self.plot_dir / 'goalkeeper_value_summary.json'
        if gk_summary_path.exists():
            try:
                gk_summary = json.loads(gk_summary_path.read_text())
            except Exception:
                gk_summary = {}

            def render_gk_table(key: str, title: str) -> str:
                payload = gk_summary.get(key)
                if not payload:
                    return ""
                rows = payload.get('rows', [])
                if not rows:
                    return ""
                range_text = payload.get('range_label', '')
                lines = [
                    rf"\textbf{{{title}}} ({range_text})",
                    r"\begin{center}",
                    r"{\small",
                    r"\setlength{\tabcolsep}{3pt}",
                    r"\renewcommand{\arraystretch}{1.0}",
                    r"\resizebox{\linewidth}{!}{%",
                    r"\begin{tabular}{l c c c c}",
                    r"\toprule",
                    r"\textbf{Player} & \textbf{GP} & \textbf{Save\%} & \textbf{CS} & \textbf{Pts} \\",
                    r"\midrule",
                ]
                for row in rows[:10]:
                    lines.append(
                        rf"{row.get('name','-')} & {row.get('gp',0)} & {row.get('save_pct',0)}\% & {row.get('cs',0)} & {row.get('pts',0)} \\"
                    )
                lines.extend([r"\bottomrule", r"\end{tabular}", r"}%", r"}", r"\end{center}"])
                return "\n".join(lines)

            season_table = render_gk_table('season', 'Top Goalkeepers - Season')
            recent_table = render_gk_table('last5', 'Top Goalkeepers - Last 5 Games')

            if season_table or recent_table:
                sections.append(r"\vspace{0.4cm}")
                sections.append(r"\subsubsection{Top Goalkeepers Tables}")
                if season_table and recent_table:
                    sections.append(r"\noindent")
                    sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                    sections.append(season_table)
                    sections.append(r"\end{minipage}")
                    sections.append(r"\hfill")
                    sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                    sections.append(recent_table)
                    sections.append(r"\end{minipage}")
                else:
                    if season_table:
                        sections.append(season_table)
                    if recent_table:
                        sections.append(recent_table)

            def render_gk_categories(key: str):
                payload = gk_summary.get(key)
                if not payload:
                    return []
                categories = payload.get('categories', {})
                if not categories:
                    return []
                range_label = payload.get('range_label', '')
                tables = []

                def render_single(cat_key: str, cat_label: str) -> str:
                    entries = categories.get(cat_key, [])
                    if not entries:
                        return ""
                    lines = [
                        rf"\textbf{{{cat_label}}} ({range_label})",
                        r"\begin{center}",
                        r"{\scriptsize",
                        r"\setlength{\tabcolsep}{2pt}",
                        r"\renewcommand{\arraystretch}{0.95}",
                        r"\resizebox{\linewidth}{!}{%",
                        r"\begin{tabular}{l c c c c}",
                        r"\toprule",
                        r"\textbf{Player} & \textbf{GP} & \textbf{Save\%} & \textbf{CS} & \textbf{Pts} \\",
                        r"\midrule",
                    ]
                    for e in entries[:8]:
                        lines.append(
                            rf"{e.get('name','-')} & {e.get('gp',0)} & {e.get('save_pct',0)}\% & {e.get('cs',0)} & {e.get('pts',0)} \\"
                        )
                    lines.extend([r"\bottomrule", r"\end{tabular}", r"}%", r"}", r"\end{center}"])
                    return "\n".join(lines)

                for cat_key, cat_label in [
                    ('elite', 'ELITE (Premium Picks)'),
                    ('protected', 'PROTECTED (Good Defence)'),
                    ('unlucky', 'UNLUCKY (Due Points)'),
                ]:
                    block = render_single(cat_key, cat_label)
                    if block:
                        tables.append(block)
                return tables

            season_cat = render_gk_categories('season')
            recent_cat = render_gk_categories('last5')

            if season_cat or recent_cat:
                sections.append(r"\vspace{0.4cm}")
                sections.append(r"\subsubsection{Goalkeeper Shot-Stopping Quadrants (Tables)}")

                def layout_three(tables_list):
                    if not tables_list:
                        return
                    if len(tables_list) == 3:
                        sections.append(r"\noindent")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[0])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[1])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.31\textwidth}")
                        sections.append(tables_list[2])
                        sections.append(r"\end{minipage}")
                    elif len(tables_list) == 2:
                        sections.append(r"\noindent")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(tables_list[0])
                        sections.append(r"\end{minipage}")
                        sections.append(r"\hfill")
                        sections.append(r"\begin{minipage}[t]{0.48\textwidth}")
                        sections.append(tables_list[1])
                        sections.append(r"\end{minipage}")
                    else:
                        sections.append(tables_list[0])
                    sections.append(r"\vspace{0.5cm}")

                if season_cat:
                    layout_three(season_cat)
                if recent_cat:
                    layout_three(recent_cat)
        
        sections.append(r"\vspace{0.5cm}")
        
        return '\n'.join(sections)

    def generate_transfer_recommendations(self, recommendations: List[Dict],
                                           captain_picks: List[Dict]) -> str:
        """Generate transfer recommendations section (legacy, kept for compatibility)."""
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\section{Transfer Recommendations}")

        if not recommendations:
            sections.append(r"\textit{No underperformers identified. Squad is performing well!}")
        else:
            sections.append(r"\subsection{Underperformers to Consider Replacing}")

            for rec in recommendations[:3]:  # Top 3
                out_player = rec.get('out', {})
                in_options = rec.get('in_options', [])

                sections.append(rf"""
\subsubsection*{{\textcolor{{fplpink}}{{OUT:}} {out_player.get('name', 'Unknown')} ({out_player.get('position', 'UNK')}) - \pounds{out_player.get('price', 0):.1f}m}}
\textbf{{Reasons:}} {', '.join(out_player.get('reasons', ['Low performance']))}

\textbf{{Suggested Replacements:}}""")

                if not in_options:
                    sections.append(r"\textit{No affordable replacements found within budget. Consider using a Free Hit or Wildcard.}")
                    sections.append(r"\vspace{5pt}")
                    continue

                sections.append(r"""
\begin{tabular}{l|cccccc|c}
\toprule
\textbf{Player} & \textbf{Price} & \textbf{Form} & \textbf{xP (RF)} & \textbf{Peer\%} & \textbf{FDR} & \textbf{Trend} & \textbf{Fixtures} \\
\midrule
""")
                for opt in in_options[:3]:
                    fixtures = opt.get('fixtures', [])
                    fix_str = ' '.join([f"{f.get('opponent', '?')}({'H' if f.get('is_home') else 'A'})" for f in fixtures[:3]])
                    peer_rank = opt.get('peer_rank', 50)
                    avg_fdr = opt.get('avg_fdr', 3.0)
                    pred_pts = opt.get('predicted_points', 0.0)
                    fdr_color = 'fplgreen' if avg_fdr <= 2.5 else ('gold' if avg_fdr <= 3.5 else 'fplpink')
                    
                    # Transfer momentum trend indicator
                    momentum = opt.get('transfer_momentum', 'neutral')
                    if momentum == 'bandwagon':
                        trend_str = r'\textcolor{fplgreen}{$\uparrow$}'
                    elif momentum == 'falling_knife':
                        trend_str = r'\textcolor{fplpink}{$\downarrow$}'
                    else:
                        trend_str = r'\textcolor{fplgray}{--}'
                    
                    sections.append(rf"{opt.get('name', 'Unknown')} & \pounds{opt.get('price', 0)}m & {opt.get('form', 0)} & \textbf{{{pred_pts}}} & {peer_rank}\% & \textcolor{{{fdr_color}}}{{{avg_fdr}}} & {trend_str} & {fix_str} \\")

                sections.append(r"""
\bottomrule
\end{tabular}
\vspace{5pt}
""")

        # Captain picks
        sections.append(r"\subsection{Captain Picks for Next GW}")
        if captain_picks:
            sections.append(r"\begin{enumerate}")
            for pick in captain_picks[:3]:
                reasons = ', '.join(pick.get('reasons', ['Good form']))
                sections.append(rf"\item \textbf{{{pick.get('name', 'Unknown')}}} ({pick.get('position', 'UNK')}) - {reasons}")
            sections.append(r"\end{enumerate}")

        return '\n'.join(sections)

    def generate_multi_week_strategy(self, multi_week_strategy: Dict,
                                      captain_picks: List[Dict]) -> str:
        """Generate comprehensive multi-week transfer strategy section.
        
        Args:
            multi_week_strategy: Strategy dict from TransferStrategyPlanner.
            captain_picks: Captain pick suggestions.
            
        Returns:
            LaTeX content for the transfer strategy section.
        """
        if not multi_week_strategy:
            return self.generate_transfer_recommendations([], captain_picks)
        
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\section{Transfer Strategy \& Planning Horizon}")
        
        current_gw = multi_week_strategy.get('current_gameweek', self.gameweek)
        horizon = multi_week_strategy.get('planning_horizon', 5)
        
        # MIP Solver Status
        mip_rec = multi_week_strategy.get('mip_recommendation')
        is_mip_optimal = mip_rec and mip_rec.get('status') == 'optimal'

        # Model Performance Metrics
        metrics = multi_week_strategy.get('model_metrics', {})
        confidence = multi_week_strategy.get('model_confidence', 'unknown')
        

        
        # Expected Value Analysis
        ev_data = multi_week_strategy.get('expected_value', {})
        current_ev = ev_data.get('current_squad', 0)
        optimized_ev = ev_data.get('optimized_squad', 0)
        potential_gain = ev_data.get('potential_gain', 0)
        
        gain_color = 'fplgreen' if potential_gain > 0 else ('fplpink' if potential_gain < 0 else 'fplgray')
        gain_sign = '+' if potential_gain > 0 else ''
        
        sections.append(rf"""
\subsection{{Expected Value Analysis (Next {horizon} GWs)}}

\begin{{center}}
\begin{{tikzpicture}}
    \node[draw=fplgreen,line width=2pt,rounded corners=8pt,inner sep=15pt,fill=fplgreen!10] {{
        \begin{{tabular}}{{ccc}}
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{current_ev:.1f}}}}} &
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{optimized_ev:.1f}}}}} &
            \textcolor{{{gain_color}}}{{\LARGE\textbf{{{gain_sign}{potential_gain:.1f}}}}}\\[3pt]
            \textcolor{{fpldark}}{{Current Squad xP}} &
            \textcolor{{fpldark}}{{Optimized xP}} &
            \textcolor{{fpldark}}{{Potential Gain}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}
""")
        
        # Fixture Heatmap
        fixture_analysis = multi_week_strategy.get('fixture_analysis', {})
        squad_predictions = multi_week_strategy.get('squad_predictions', {})
        
        if fixture_analysis:
            sections.append(self._generate_fixture_heatmap(
                fixture_analysis, squad_predictions, current_gw, horizon
            ))
        
        # --- STRATEGY SECTIONS ---
        if is_mip_optimal:
            # === MIP STRATEGY ONLY ===
            sections.append(self._generate_mip_recommendation(mip_rec, current_gw, horizon))
        else:
            # === HEURISTIC FALLBACK ===
            # Immediate Recommendations
            immediate_recs = multi_week_strategy.get('immediate_recommendations', [])
            if immediate_recs:
                sections.append(self._generate_immediate_recommendations(immediate_recs, current_gw))
            else:
                sections.append(rf"""
\subsection{{Immediate Recommendations (GW{current_gw + 1})}}
\textit{{No urgent transfers recommended. Your squad is well-positioned!}}
""")
            
            # Planned Transfer Sequence
            planned_transfers = multi_week_strategy.get('planned_transfers', [])
            if planned_transfers:
                sections.append(self._generate_transfer_sequence(planned_transfers))
            
            # Alternative Strategies
            alternatives = multi_week_strategy.get('alternative_strategies', {})
            if alternatives:
                sections.append(self._generate_alternative_strategies(alternatives))
                
            if mip_rec and mip_rec.get('status') == 'unavailable':
                 sections.append(rf"""
\subsection{{Optimal Transfer Plan (MIP Solver)}}
\textit{{\small MIP solver not available. Install sasoptpy and highspy for mathematically optimal transfers.}}
""")
        
        # Captain Picks (Always show)
        sections.append(r"\subsection{Captain Picks for Next GW}")
        if captain_picks:
            sections.append(r"\begin{enumerate}")
            for pick in captain_picks[:3]:
                reasons = ', '.join(pick.get('reasons', ['Good form']))
                sections.append(rf"\item \textbf{{{pick.get('name', 'Unknown')}}} ({pick.get('position', 'UNK')}) - {reasons}")
            sections.append(r"\end{enumerate}")
        else:
            sections.append(r"\textit{Insufficient data for captain recommendations.}")
        
        return '\n'.join(sections)

    def _generate_fixture_heatmap(self, fixture_analysis: Dict, 
                                   squad_predictions: Dict,
                                   current_gw: int, horizon: int) -> str:
        """Generate fixture difficulty heatmap for squad."""
        sections = []
        sections.append(rf"\subsection{{Fixture Difficulty Heatmap (GW{current_gw + 1}-{current_gw + horizon})}}")
        
        # Build table header
        gw_headers = ' & '.join([rf"\textbf{{GW{current_gw + i + 1}}}" for i in range(min(horizon, 5))])
        
        sections.append(rf"""
\begin{{center}}
\small
\begin{{tabular}}{{l|l|{'c' * min(horizon, 5)}|r}}
\toprule
\textbf{{Player}} & \textbf{{Swing}} & {gw_headers} & \textbf{{5-GW xP}} \\
\midrule
""")
        
        # Sort by position (position now stored in fixture_analysis)
        sorted_players = sorted(
            fixture_analysis.items(),
            key=lambda x: {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}.get(
                x[1].get('position', 'UNK'), 4
            )
        )
        
        for pid, fix_data in sorted_players[:15]:  # Limit to 15 players
            # Get player info from fixture_analysis (now includes player_name and position)
            pred_data = squad_predictions.get(pid, {})
            
            fixtures = fix_data.get('fixtures', [])
            swing = fix_data.get('swing', 'neutral')
            
            # Get player name from fixture_analysis
            player_name = fix_data.get('player_name', f'Player {pid}')
            
            # Swing indicator
            swing_color = 'rising' if swing == 'improving' else ('falling' if swing == 'worsening' else 'stable')
            swing_symbol = r'$\uparrow$' if swing == 'improving' else (r'$\downarrow$' if swing == 'worsening' else r'$\rightarrow$')
            
            # Build fixture cells
            fix_cells = []
            for i in range(min(horizon, 5)):
                if i < len(fixtures):
                    fix = fixtures[i]
                    opp = fix.get('opponent', '?')
                    is_home = fix.get('is_home', False)
                    diff = fix.get('difficulty', 3)
                    
                    # Check for Elo probabilities (Win/Draw/Loss)
                    win_prob = fix.get('win_prob')
                    
                    if win_prob is not None:
                        # Color based on Win Probability (Elo)
                        # High Win % = Easy (Green)
                        if win_prob >= 0.60:
                            cell_color = 'fplgreen!50'
                        elif win_prob >= 0.45:
                            cell_color = 'fplgreen!25'
                        elif win_prob >= 0.35:
                            cell_color = 'gold!40'
                        elif win_prob >= 0.20:
                            cell_color = 'orange!40'
                        else:
                            cell_color = 'fplpink!40'
                        
                        venue = 'H' if is_home else 'A'
                        # Display Opponent overlayed with Win % probability
                        fix_cells.append(rf"\cellcolor{{{cell_color}}}\shortstack{{{opp}({venue})\\{{\tiny {int(win_prob*100)}\%}}}}")
                    else:
                        # Fallback to standard 1-5 Difficulty
                        if diff <= 2:
                            cell_color = 'fplgreen!40'
                        elif diff == 3:
                            cell_color = 'gold!40'
                        elif diff == 4:
                            cell_color = 'orange!40'
                        else:
                            cell_color = 'fplpink!40'
                        
                        venue = 'H' if is_home else 'A'
                        fix_cells.append(rf"\cellcolor{{{cell_color}}}{opp}({venue})")
                else:
                    fix_cells.append(r"\cellcolor{gray!20}BGW")
            
            fix_str = ' & '.join(fix_cells)
            
            # Get 5-GW expected points
            cumulative = pred_data.get('cumulative', 0)
            
            # Escape LaTeX special chars in player name
            player_name = player_name.replace('_', r'\_').replace('&', r'\&')
            
            sections.append(rf"{player_name} & \textcolor{{{swing_color}}}{{{swing_symbol}}} & {fix_str} & {cumulative:.1f} \\")
        
        sections.append(r"""
\bottomrule
\end{tabular}
\end{center}

\textit{\small Colors: \colorbox{fplgreen!40}{Easy (1-2)} \colorbox{gold!40}{Medium (3)} \colorbox{orange!40}{Hard (4)} \colorbox{fplpink!40}{Very Hard (5)}}
""")
        
        return '\n'.join(sections)

    def _generate_immediate_recommendations(self, recommendations: List[Dict],
                                             current_gw: int) -> str:
        """Generate immediate transfer recommendations with 5-GW context."""
        sections = []
        sections.append(rf"\subsection{{Immediate Recommendations (GW{current_gw + 1})}}")
        
        for rec in recommendations[:3]:
            out_player = rec.get('out', {})
            in_options = rec.get('in_options', [])
            priority = rec.get('priority', 'medium')
            
            # Priority badge
            priority_color = 'fplpink' if priority == 'high' else ('gold' if priority == 'medium' else 'fplgray')
            
            out_5gw = out_player.get('5gw_expected', 0)
            out_swing = out_player.get('fixture_swing', 'neutral')
            swing_indicator = r'$\uparrow$' if out_swing == 'improving' else (r'$\downarrow$' if out_swing == 'worsening' else '')
            
            sections.append(rf"""
\subsubsection*{{\colorbox{{{priority_color}!30}}{{\small {priority.upper()}}} \textcolor{{fplpink}}{{OUT:}} {out_player.get('name', 'Unknown')} ({out_player.get('position', 'UNK')}) - \pounds{out_player.get('price', 0):.1f}m}}
\textbf{{Reasons:}} {', '.join(out_player.get('reasons', ['Low performance']))}\\
\textbf{{5-GW Expected:}} {out_5gw:.1f} pts {swing_indicator}

\textbf{{Suggested Replacements:}}""")
            
            if not in_options:
                sections.append(r"\textit{No affordable replacements found within budget.}")
                sections.append(r"\vspace{5pt}")
                continue
            
            sections.append(r"""
\begin{tabular}{l|cccc|cc|c}
\toprule
\textbf{Player} & \textbf{Price} & \textbf{Form} & \textbf{xP} & \textbf{5-GW xP} & \textbf{Gain} & \textbf{FDR} & \textbf{Fixtures} \\
\midrule
""")
            
            for opt in in_options[:3]:
                fixtures = opt.get('fixtures', [])
                fix_str = ' '.join([f"{f.get('opponent', '?')}({'H' if f.get('is_home') else 'A'})" for f in fixtures[:3]])
                
                pred_pts = opt.get('predicted_points', 0.0)
                five_gw = opt.get('5gw_expected', 0)
                gain = opt.get('expected_gain', 0)
                avg_fdr = opt.get('avg_fdr', 3.0)
                
                fdr_color = 'fplgreen' if avg_fdr <= 2.5 else ('gold' if avg_fdr <= 3.5 else 'fplpink')
                gain_color = 'fplgreen' if gain > 0 else 'fplpink'
                gain_sign = '+' if gain > 0 else ''
                
                sections.append(rf"{opt.get('name', 'Unknown')} & \pounds{opt.get('price', 0)}m & {opt.get('form', 0)} & {pred_pts:.1f} & \textbf{{{five_gw:.1f}}} & \textcolor{{{gain_color}}}{{{gain_sign}{gain:.1f}}} & \textcolor{{{fdr_color}}}{{{avg_fdr}}} & {fix_str} \\")
            
            sections.append(r"""
\bottomrule
\end{tabular}
\vspace{5pt}
""")
        
        return '\n'.join(sections)

    def _generate_transfer_sequence(self, planned_transfers: List[Dict]) -> str:
        """Generate planned transfer sequence timeline."""
        sections = []
        sections.append(r"\subsection{Recommended Transfer Sequence}")
        
        sections.append(r"""
\begin{center}
\begin{tikzpicture}[
    transfer/.style={draw=fplpurple,rounded corners=5pt,minimum width=10cm,minimum height=1.2cm,fill=white,font=\small},
    consider/.style={draw=fplgray,rounded corners=5pt,minimum width=10cm,minimum height=1.2cm,fill=gray!10,font=\small,dashed},
    gwlabel/.style={font=\bfseries\small,color=fplpurple}
]
""")
        
        y_pos = 0
        for i, transfer in enumerate(planned_transfers[:4]):  # Limit to 4
            gw = transfer.get('gameweek', '?')
            action = transfer.get('action', 'transfer')
            out_name = transfer.get('out', 'Unknown')
            in_name = transfer.get('in', 'Unknown')
            gain = transfer.get('expected_gain', 0)
            reasoning = transfer.get('reasoning', '')
            take_hit = transfer.get('take_hit', False)
            
            # Escape LaTeX chars
            out_name = out_name.replace('_', r'\_').replace('&', r'\&')
            in_name = in_name.replace('_', r'\_').replace('&', r'\&')
            reasoning = reasoning.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
            
            style = 'transfer' if action == 'transfer' else 'consider'
            hit_text = ' (Take -4 hit)' if take_hit else ''
            gain_sign = '+' if gain > 0 else ''
            
            sections.append(rf"""
\node[gwlabel] at (-6,{y_pos}) {{GW{gw}}};
\node[{style}] at (0,{y_pos}) {{{out_name} $\rightarrow$ {in_name}{hit_text} \hfill \textcolor{{fplgreen}}{{({gain_sign}{gain:.1f} xP)}} }};
\node[font=\tiny,color=fplgray,text width=9cm,align=left] at (0,{y_pos - 0.7}) {{{reasoning}}};
""")
            y_pos -= 2
        
        sections.append(r"""
\end{tikzpicture}
\end{center}
""")
        
        return '\n'.join(sections)

    def _generate_alternative_strategies(self, alternatives: Dict) -> str:
        """Generate alternative transfer strategies section."""
        sections = []
        sections.append(r"\subsection{Alternative Strategies}")
        
        aggressive = alternatives.get('aggressive', {})
        conservative = alternatives.get('conservative', {})
        wildcard = alternatives.get('wildcard_consideration', False)
        
        # Determine which is recommended
        agg_rec = aggressive.get('recommended', False)
        con_rec = conservative.get('recommended', False)
        
        agg_color = 'fplgreen!20' if agg_rec else 'white'
        con_color = 'fplgreen!20' if con_rec else 'white'
        
        agg_gain = aggressive.get('net_gain', 0)
        con_gain = conservative.get('net_gain', 0)
        
        sections.append(rf"""
\begin{{center}}
\begin{{tabular}}{{l|c|c|c|c}}
\toprule
\textbf{{Strategy}} & \textbf{{Transfers}} & \textbf{{Hits}} & \textbf{{Net Gain}} & \textbf{{Recommendation}} \\
\midrule
\rowcolor{{{agg_color}}}
Aggressive & {aggressive.get('transfers', 0)} & -{aggressive.get('hits', 0) * 4} pts & \textcolor{{fplgreen}}{{+{agg_gain:.1f}}} & {'Recommended' if agg_rec else 'Optional'} \\
\rowcolor{{{con_color}}}
Conservative & {conservative.get('transfers', 1)} & 0 pts & \textcolor{{fplgreen}}{{+{con_gain:.1f}}} & {'Recommended' if con_rec else 'Optional'} \\
\bottomrule
\end{{tabular}}
\end{{center}}
""")
        
        if wildcard:
            wc_reason = alternatives.get('wildcard_reason', 'Multiple beneficial transfers identified')
            sections.append(rf"""
\vspace{{0.3cm}}
\begin{{center}}
\colorbox{{gold!30}}{{\textbf{{Wildcard Consideration:}} {wc_reason}}}
\end{{center}}
""")
        
        return '\n'.join(sections)

    def _generate_mip_recommendation(self, mip_rec: Dict, current_gw: int, horizon: int) -> str:
        """Generate MIP solver optimal transfer recommendation section.
        
        Args:
            mip_rec: MIP solver result dictionary.
            current_gw: Current gameweek number.
            horizon: Planning horizon in weeks.
            
        Returns:
            LaTeX content for the MIP recommendation section.
        """
        sections = []
        sections.append(r"\subsection{Optimal Transfer Plan (MIP Solver)}")
        
        # Solver info box
        solver_time = mip_rec.get('solver_time', 0)
        expected_pts = mip_rec.get('expected_points', 0)
        hit_cost = mip_rec.get('hit_cost', 0)
        num_transfers = mip_rec.get('num_transfers', 0)
        budget_remaining = mip_rec.get('budget_remaining', 0)
        
        sections.append(rf"""
\begin{{center}}
\begin{{tikzpicture}}
    \node[draw=fplgreen,line width=2pt,rounded corners=10pt,inner sep=15pt,fill=fplgreen!10] {{
        \begin{{tabular}}{{cccc}}
            \textcolor{{fplpurple}}{{\LARGE\textbf{{{expected_pts:.1f}}}}} &
            \textcolor{{fplpurple}}{{\Large\textbf{{{num_transfers}}}}} &
            \textcolor{{fplpink}}{{\Large\textbf{{-{hit_cost}}}}} &
            \textcolor{{fplpurple}}{{\Large\textbf{{\pounds{budget_remaining:.1f}m}}}}\\[5pt]
            \textcolor{{fplgray}}{{\small Expected Points}} &
            \textcolor{{fplgray}}{{\small Transfers}} &
            \textcolor{{fplgray}}{{\small Hit Cost}} &
            \textcolor{{fplgray}}{{\small Budget Remaining}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}

\small\textit{{Solved in {solver_time:.2f}s using Mixed-Integer Programming}}
""")
        
        # Transfers section
        transfers_out = mip_rec.get('transfers_out', [])
        transfers_in = mip_rec.get('transfers_in', [])
        
        if transfers_out and transfers_in:
            sections.append(r"\subsubsection*{Recommended Transfers}")
            sections.append(r"""
\begin{center}
\begin{tabular}{l|r|c|l|r}
\toprule
\textbf{OUT} & \textbf{Sell} & & \textbf{IN} & \textbf{Buy} \\
\midrule
""")
            
            max_transfers = max(len(transfers_out), len(transfers_in))
            for i in range(max_transfers):
                out_name = transfers_out[i].get('name', '-') if i < len(transfers_out) else '-'
                out_price = f"\\pounds{transfers_out[i].get('sell_price', 0):.1f}m" if i < len(transfers_out) else '-'
                in_name = transfers_in[i].get('name', '-') if i < len(transfers_in) else '-'
                in_price = f"\\pounds{transfers_in[i].get('buy_price', 0):.1f}m" if i < len(transfers_in) else '-'
                
                # Escape LaTeX special chars
                out_name = out_name.replace('_', r'\_').replace('&', r'\&')
                in_name = in_name.replace('_', r'\_').replace('&', r'\&')
                
                sections.append(rf"\textcolor{{fplpink}}{{{out_name}}} & {out_price} & $\rightarrow$ & \textcolor{{fplgreen}}{{{in_name}}} & {in_price} \\")
            
            sections.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")
            # "Why This Move?" Explanation Box
            sections.append(r"\subsubsection*{Why These Moves?}")
            sections.append(r"\begin{tcolorbox}[colback=fplgreen!5,colframe=fplgreen!50,title=Transfer Analysis]")
            
            for i in range(min(len(transfers_out), len(transfers_in))):
                p_out = transfers_out[i]
                p_in = transfers_in[i]
                
                out_name = p_out.get('name', 'Unknown').replace('_', r'\_')
                in_name = p_in.get('name', 'Unknown').replace('_', r'\_')
                
                # Calculate xP gain from per_gw_xp if available
                in_xp = p_in.get('xp', p_in.get('total_xp', 0))
                out_xp = p_out.get('xp', p_out.get('total_xp', 0))
                
                if isinstance(in_xp, list):
                    in_total = sum(in_xp[:5])
                else:
                    in_total = float(in_xp) if in_xp else 0
                    
                if isinstance(out_xp, list):
                    out_total = sum(out_xp[:5])
                else:
                    out_total = float(out_xp) if out_xp else 0
                
                xp_gain = in_total - out_total
                
                # Price difference
                in_price = p_in.get('buy_price', p_in.get('price', 0))
                out_price = p_out.get('sell_price', p_out.get('price', 0))
                price_diff = out_price - in_price
                
                # Classification
                net_gain = xp_gain - (hit_cost / max(num_transfers, 1))
                if net_gain >= 5:
                    icon = r'\textcolor{fplgreen}{$\checkmark\checkmark$}'
                    verdict = 'Excellent'
                elif net_gain >= 2:
                    icon = r'\textcolor{fplgreen}{$\checkmark$}'
                    verdict = 'Good'
                elif net_gain >= 0:
                    icon = r'\textcolor{gold}{$\approx$}'
                    verdict = 'Marginal'
                else:
                    icon = r'\textcolor{fplpink}{$\times$}'
                    verdict = 'Reconsider'
                
                # Build explanation
                price_text = f"Saves \\pounds{price_diff:.1f}m" if price_diff > 0 else f"Costs \\pounds{abs(price_diff):.1f}m extra"
                
                sections.append(rf"\textbf{{{out_name} $\rightarrow$ {in_name}}} {icon} \textit{{{verdict}}}")
                sections.append(rf"\begin{{itemize}}[noitemsep,topsep=0pt]")
                sections.append(rf"    \item Expected gain: +{xp_gain:.1f} xP over 5 GWs")
                sections.append(rf"    \item {price_text}")
                sections.append(rf"\end{{itemize}}")
            
            sections.append(r"\end{tcolorbox}")
        else:
            sections.append(r"""
\begin{center}
\colorbox{fplgreen!20}{\textbf{No transfers recommended - your squad is already optimal!}}
\end{center}
""")
        
        # Per-GW Expected Points Timeline
        per_gw_xp = mip_rec.get('per_gw_xp', [])
        if per_gw_xp:
            sections.append(r"\subsubsection*{Expected Points Timeline}")
            
            # Visual timeline using TikZ
            sections.append(r"\begin{center}")
            sections.append(r"\begin{tikzpicture}[")
            sections.append(r"    week/.style={draw=fplpurple,rounded corners=5pt,minimum width=2cm,minimum height=1.5cm,fill=white},")
            sections.append(r"    arrow/.style={->,thick,color=fplpurple}")
            sections.append(r"]")
            
            for i, xp in enumerate(per_gw_xp[:5]):  # Max 5 weeks
                x_pos = i * 2.8
                gw = current_gw + i + 1
                
                # Color based on xP (green for high, orange for low)
                if xp >= 50:
                    fill_color = "fplgreen!20"
                elif xp >= 40:
                    fill_color = "gold!20"
                else:
                    fill_color = "orange!20"
                
                sections.append(rf"\node[week,fill={fill_color}] at ({x_pos},0) {{}};")
                sections.append(rf"\node[font=\scriptsize\bfseries,color=fplpurple] at ({x_pos},0.4) {{GW{gw}}};")
                sections.append(rf"\node[font=\large\bfseries,color=fplpurple] at ({x_pos},-0.1) {{{xp:.1f}}};")
                
                # Arrow to next
                if i < len(per_gw_xp) - 1 and i < 4:
                    sections.append(rf"\draw[arrow] ({x_pos + 1.1},0) -- ({x_pos + 1.7},0);")
            
            sections.append(r"\end{tikzpicture}")
            sections.append(r"\end{center}")
            
            # Also show as table
            gw_headers = ' & '.join([rf"\textbf{{GW{current_gw + i + 1}}}" for i in range(len(per_gw_xp))])
            gw_values = ' & '.join([rf"{xp:.1f}" for xp in per_gw_xp])
            
            sections.append(rf"""
\begin{{center}}
\small
\begin{{tabular}}{{{'c' * len(per_gw_xp)}|c}}
\toprule
{gw_headers} & \textbf{{Total}} \\
\midrule
{gw_values} & \textbf{{{sum(per_gw_xp):.1f}}} \\
\bottomrule
\end{{tabular}}
\end{{center}}
""")
        
        # Optimal Starting XI for next GW
        starting_xi = mip_rec.get('starting_xi', [])
        captain = mip_rec.get('captain', {})
        formation = mip_rec.get('formation', '')
        
        if starting_xi:
            captain_name = captain.get('name', '') if captain else ''
            sections.append(rf"\subsubsection*{{Optimal Starting XI (Formation: {formation})}}")
            
            sections.append(r"""
\begin{center}
\begin{tabular}{l|l|l|r}
\toprule
\textbf{Position} & \textbf{Player} & \textbf{Team} & \textbf{5-GW xP} \\
\midrule
""")
            
            for player in starting_xi:
                pos = player.get('position', 'UNK')
                name = player.get('name', 'Unknown').replace('_', r'\_').replace('&', r'\&')
                team = player.get('team', 'UNK')
                total_xp = player.get('total_xp', 0)
                
                # Mark captain
                if name.replace(r'\_', '_').replace(r'\&', '&') == captain_name:
                    name = rf"\textbf{{{name} (C)}}"
                
                sections.append(rf"{pos} & {name} & {team} & {total_xp:.1f} \\")
            
            sections.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")
        
        return '\n'.join(sections)

    def generate_chip_strategy(self, chips_used: List[Dict], gw_history: List[Dict] = None) -> str:
        """Generate comprehensive chip usage strategy section."""
        used_chips = {c.get('name', ''): c.get('event', 0) for c in chips_used}
        all_chips = ['wildcard', 'freehit', 'bboost', '3xc']
        chip_labels = {'wildcard': 'Wildcard', 'freehit': 'Free Hit', 'bboost': 'Bench Boost', '3xc': 'Triple Captain'}

        # Determine current gameweek
        current_gw = len(gw_history) if gw_history else self.gameweek

        # Calculate which half of season we're in (for 2025-26 dual chip system)
        first_half = current_gw <= 19
        half_label = "First Half (GW1-19)" if first_half else "Second Half (GW20-38)"

        chip_nodes = []
        for i, chip in enumerate(all_chips):
            label = chip_labels.get(chip, chip)
            if chip in used_chips:
                status = f"Used GW{used_chips[chip]}"
                fill = "fplgray!30"
            else:
                status = "Available"
                fill = "fplgreen!20"
            chip_nodes.append(rf"\node[draw=fplpurple,rounded corners=5pt,minimum width=2.3cm,minimum height=1.5cm,fill={fill}] at ({i*3.5},0) {{}};"
                              rf"\node[font=\small\bfseries,color=fplpurple] at ({i*3.5},0.3) {{{label}}};"
                              rf"\node[font=\scriptsize,color=fplgray] at ({i*3.5},-0.3) {{{status}}};")

        # Count available chips
        available_chips = [c for c in all_chips if c not in used_chips]
        chips_remaining = len(available_chips)
        
        # Build DYNAMIC chip timing recommendations based on current GW
        timing_rows = []
        
        # Wildcard 1 (first half) - only if not used and still in first half
        if 'wildcard' not in used_chips and first_half:
            if current_gw < 7:
                timing_rows.append(("Wildcard 1", "GW7-9", "Fixture swing period"))
            elif current_gw < 10:
                timing_rows.append(("Wildcard 1", f"GW{current_gw+1}-{min(current_gw+3, 19)}", "Use soon - first half ends GW19"))
            else:
                timing_rows.append(("Wildcard 1", "Before GW20", "Use soon - does not carry over!"))
        
        # Wildcard 2 (second half) - only if in second half and not used
        if 'wildcard' not in used_chips and not first_half:
            if current_gw < 30:
                timing_rows.append(("Wildcard 2", "GW30-32", "Prepare for run-in"))
            else:
                timing_rows.append(("Wildcard 2", "Before GW38", "Use before season ends"))
        
        # Free Hit - only if not used
        if 'freehit' not in used_chips:
            if first_half:
                timing_rows.append(("Free Hit 1", "Next BGW", "Team gaps during blanks"))
            else:
                timing_rows.append(("Free Hit 2", "BGW/DGW", "Flexibility for late fixtures"))
        
        # Bench Boost - typically spring DGWs
        if 'bboost' not in used_chips:
            if current_gw < 34:
                timing_rows.append(("Bench Boost", "DGW34-37", "Maximum doubles opportunity"))
            else:
                timing_rows.append(("Bench Boost", "Next DGW", "Use on any remaining double"))
        
        # Triple Captain
        if '3xc' not in used_chips:
            timing_rows.append(("Triple Captain", "Next DGW", "Premium player with 2 easy fixtures"))
        
        # Build timing table content
        if timing_rows:
            timing_table_rows = "\n".join([rf"{chip} & {target} & {reason} \\" for chip, target, reason in timing_rows])
            timing_section = rf"""
\subsection{{Recommended Timing}}

\textit{{Based on GW{current_gw} - showing only available chips.}}

\begin{{center}}
\begin{{tabular}}{{l|c|l}}
\toprule
\textbf{{Chip}} & \textbf{{Target GW}} & \textbf{{Reasoning}} \\
\midrule
{timing_table_rows}
\bottomrule
\end{{tabular}}
\end{{center}}
"""
        else:
            timing_section = r"""
\subsection{Recommended Timing}

\textit{All chips have been used this season. Well done on strategic chip management!}
"""
        
        # Fetch BGW/DGW data
        try:
            bgw_dgw_data = get_bgw_dgw_gameweeks(use_cache=True, session_cache=self.session_cache)
            bgws = bgw_dgw_data.get('bgw', [])
            dgws = bgw_dgw_data.get('dgw', [])
            
            # Filter to future gameweeks only
            future_bgws = [b for b in bgws if b.get('gw', 0) > current_gw]
            future_dgws = [d for d in dgws if d.get('gw', 0) > current_gw]
            
            if future_bgws or future_dgws:
                bgw_dgw_rows = []
                
                # Add BGWs
                for bgw in future_bgws[:3]:  # Limit to next 3
                    gw = bgw.get('gw', 0)
                    teams = bgw.get('teams_missing', 0)
                    bgw_dgw_rows.append(rf"GW{gw} & \textcolor{{fplpink}}{{\textbf{{BGW}}}} & {teams} teams without fixtures \\")
                
                # Add DGWs
                for dgw in future_dgws[:3]:  # Limit to next 3
                    gw = dgw.get('gw', 0)
                    teams = dgw.get('teams_doubled', 0)
                    bgw_dgw_rows.append(rf"GW{gw} & \textcolor{{fplgreen}}{{\textbf{{DGW}}}} & {teams} teams with double fixtures \\")
                
                if bgw_dgw_rows:
                    bgw_dgw_table = "\n".join(bgw_dgw_rows)
                    bgw_dgw_section = rf"""
\textit{{Detected from fixtures data - plan your chips around these!}}

\begin{{center}}
\begin{{tabular}}{{c|c|l}}
\toprule
\textbf{{Gameweek}} & \textbf{{Type}} & \textbf{{Details}} \\
\midrule
{bgw_dgw_table}
\bottomrule
\end{{tabular}}
\end{{center}}

\textit{{\textcolor{{fplpink}}{{BGW = Blank Gameweek}} | \textcolor{{fplgreen}}{{DGW = Double Gameweek}}}}
"""
                else:
                    bgw_dgw_section = r"\textit{No confirmed BGW/DGW detected yet for remaining gameweeks.}"
            else:
                bgw_dgw_section = r"\textit{No confirmed BGW/DGW detected yet for remaining gameweeks.}"
        except Exception:
            bgw_dgw_section = r"\textit{BGW/DGW data unavailable.}"

        return rf"""
\newpage
\section{{Chip Usage Strategy}}

\subsection{{Current Chip Status}}

\begin{{center}}
\begin{{tikzpicture}}
{chr(10).join(chip_nodes)}
\end{{tikzpicture}}
\end{{center}}

\vspace{{0.3cm}}
\textbf{{Season Position:}} {half_label} | \textbf{{Chips Remaining:}} {chips_remaining}/4

\subsection{{Strategic Chip Planning}}

The 2025-26 season features two sets of chips - one for each half (GW1-19 and GW20-38). Unused chips from the first half do NOT carry over, so plan accordingly.

\subsubsection*{{Wildcard Strategy}}
\begin{{itemize}}
    \item \textbf{{First Half (GW7-9):}} Use around fixture swings when multiple premium players need replacing
    \item \textbf{{Second Half (GW30-32):}} Prepare for late-season Double Gameweeks and navigate rotation
    \item \textbf{{Key Trigger:}} When 4+ transfers are needed or major template shift occurs
\end{{itemize}}

\subsubsection*{{Free Hit Strategy}}
\begin{{itemize}}
    \item \textbf{{Target:}} Blank Gameweeks when your team has 6+ players without fixtures
    \item \textbf{{Alternative:}} Use during a favorable DGW if Bench Boost is already used
    \item \textbf{{Avoid:}} Using too early - save for fixture congestion in spring
\end{{itemize}}

\subsubsection*{{Bench Boost Strategy}}
\begin{{itemize}}
    \item \textbf{{Ideal Setup:}} All 15 players have double fixtures in a DGW
    \item \textbf{{Preparation:}} Build bench value 2-3 weeks before target DGW
    \item \textbf{{Target GWs:}} Historically GW34-37 has best DGW opportunities
    \item \textbf{{Minimum:}} At least 13 players with doubles for optimal value
\end{{itemize}}

\subsubsection*{{Triple Captain Strategy}}
\begin{{itemize}}
    \item \textbf{{Primary Target:}} Premium attacker (Haaland, Salah) in a DGW with 2 easy fixtures
    \item \textbf{{Form Check:}} Player should be in top form with consistent returns
    \item \textbf{{Fixture Analysis:}} Both DGW opponents should be FDR 2-3 (favorable)
    \item \textbf{{Avoid:}} Using on single GW unless truly exceptional fixture
\end{{itemize}}

{timing_section}

\subsection{{Blank \& Double Gameweek Calendar}}
{bgw_dgw_section}
"""

    def generate_insights(self, squad_analysis: List[Dict], gw_history: List[Dict]) -> str:
        """Generate strategic insights section based on actual squad data."""
        observations = []
        
        if squad_analysis:
            # Find top scorer in squad
            sorted_by_points = sorted(squad_analysis, 
                key=lambda x: x.get('raw_stats', {}).get('total_points', 0), reverse=True)
            if sorted_by_points:
                top_scorer = sorted_by_points[0]
                pts = top_scorer.get('raw_stats', {}).get('total_points', 0)
                observations.append(rf"\item \textbf{{Top performer:}} {top_scorer.get('name', 'Unknown')} ({pts} pts)")
            
            # Find best form player
            sorted_by_form = sorted(squad_analysis, 
                key=lambda x: x.get('form_analysis', {}).get('average', 0), reverse=True)
            if sorted_by_form:
                best_form = sorted_by_form[0]
                form_avg = best_form.get('form_analysis', {}).get('average', 0)
                if form_avg > 0 and best_form.get('name') != sorted_by_points[0].get('name'):
                    observations.append(rf"\item \textbf{{Best current form:}} {best_form.get('name', 'Unknown')} ({form_avg:.1f} PPG)")
            
            # Find underperformers
            underperformers = [p for p in squad_analysis 
                if p.get('form_analysis', {}).get('trend') == 'falling']
            if underperformers:
                worst = underperformers[0]
                observations.append(rf"\item \textbf{{Declining form:}} {worst.get('name', 'Unknown')} - consider monitoring")
            
            # xG analysis
            for p in squad_analysis:
                exp = p.get('expected_vs_actual', {})
                if exp.get('expected_goals', 0) > 1:
                    diff = exp.get('goals_diff', 0)
                    if diff > 1:
                        observations.append(rf"\item \textbf{{xG overperformer:}} {p.get('name')} (+{diff:.1f}) - regression risk")
                        break
                    elif diff < -1:
                        observations.append(rf"\item \textbf{{xG underperformer:}} {p.get('name')} ({diff:.1f}) - due positive regression")
                        break
        
        if not observations:
            observations = [r"\item Review squad performance in the deep dive section"]
        
        # Calculate projected performance
        total_pts = sum(gw.get('points', 0) for gw in gw_history) if gw_history else 0
        gws_played = len(gw_history) if gw_history else 1
        avg_per_gw = total_pts / gws_played if gws_played > 0 else 0
        gws_remaining = 38 - gws_played
        projected_total = total_pts + (avg_per_gw * gws_remaining)
        
        observations_str = "\n".join(observations)
        
        return rf"""
\newpage
\section{{Season Insights}}

\subsection{{Key Observations}}
Based on your squad's actual performance data:

\begin{{itemize}}
{observations_str}
\end{{itemize}}

\subsection{{Projected Performance}}

\begin{{center}}
\begin{{tikzpicture}}
\node[draw=fplpurple,rounded corners=10pt,minimum width=10cm,minimum height=2cm,fill=fplgreen!10] {{
    \begin{{tabular}}{{ccc}}
        \textcolor{{fplpurple}}{{\Large\textbf{{{total_pts}}}}} &
        \textcolor{{fplpurple}}{{\Large\textbf{{{avg_per_gw:.1f}}}}} &
        \textcolor{{fplpurple}}{{\Large\textbf{{{projected_total:.0f}}}}} \\[5pt]
        \textcolor{{fplgray}}{{\small Current Points}} &
        \textcolor{{fplgray}}{{\small Avg per GW}} &
        \textcolor{{fplgray}}{{\small Projected Final}}
    \end{{tabular}}
}};
\end{{tikzpicture}}
\end{{center}}

\textit{{Projection based on {gws_played} gameweeks played at {avg_per_gw:.1f} PPG average.}}
"""

    def _calculate_bonus_points(self, season_history: List[Dict], chips_used: List[Dict]) -> tuple:
        """Calculate bonus points from captaincy choices and chip usage.
        
        Bonus points are the extra points gained from:
        - Captaincy: (multiplier - 1) * captain_base_points per GW
        - Triple Captain: Additional captain_base_points (3x instead of 2x)
        - Bench Boost: Points from bench players during BB weeks
        
        Args:
            season_history: List of gameweek entries with squad information.
            chips_used: List of chips used with 'event' and 'name' fields.
            
        Returns:
            Tuple of (bonus_points, percentage_share)
        """
        if not season_history:
            return 0, 0.0
        
        # Build chip map for quick lookup
        chip_map = {c.get('event', 0): c.get('name', '') for c in chips_used}
        
        total_bonus = 0
        total_points = 0
        
        for gw_entry in season_history:
            gw_num = gw_entry.get('gameweek', 0)
            squad = gw_entry.get('squad', [])
            chip_this_gw = chip_map.get(gw_num, '')
            is_bench_boost = chip_this_gw in ('bboost', 'Bench Boost')
            
            for player in squad:
                position_in_squad = player.get('position_in_squad', 0)
                base_points = player.get('stats', {}).get('event_points', 0) or 0
                multiplier = player.get('multiplier', 1)
                
                # Calculate actual points contributed
                if position_in_squad <= 11:
                    # Starting XI player
                    actual_points = base_points * multiplier
                    total_points += actual_points
                    
                    # Captain bonus: extra points from multiplier > 1
                    if multiplier > 1:
                        captain_bonus = base_points * (multiplier - 1)
                        total_bonus += captain_bonus
                elif is_bench_boost:
                    # Bench players count during Bench Boost
                    total_points += base_points
                    total_bonus += base_points  # All bench points are "bonus"
        
        # Calculate percentage
        pct_share = (total_bonus / total_points * 100) if total_points > 0 else 0.0
        
        return total_bonus, pct_share

    def generate_competitive_analysis(self, competitive_data: List[Dict]) -> str:
        """Generate competitive analysis section comparing multiple teams.

        Args:
            competitive_data: List of entry dicts with team_info, gw_history, squad, etc.

        Returns:
            LaTeX string for the competitive analysis section.
        """
        if not competitive_data:
            return ""

        # Build summary table rows
        summary_rows = []
        for entry in competitive_data:
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', 'Unknown'))
            manager = self._escape_latex(team_info.get('manager_name', 'Unknown'))
            points = team_info.get('overall_points', 0)
            team_value = entry.get('team_value', 100.0)
            bank = entry.get('bank', 0.0)
            total_hits = entry.get('total_hits', 0)
            chips_used = entry.get('chips_used', [])
            chips_str = ', '.join([c.get('name', '')[:2].upper() for c in chips_used]) or '-'
            
            # Calculate free transfers using same methodology as season summary
            gw_history = entry.get('gw_history', [])
            free_transfers = self._calculate_free_transfers(gw_history, chips_used)
            
            # Calculate bonus points from captaincy and chips
            season_history = entry.get('season_history', [])
            bonus_pts, bonus_pct = self._calculate_bonus_points(season_history, chips_used)
            bonus_str = f"{bonus_pts} ({bonus_pct:.1f}\\%)"

            summary_rows.append(
                rf"{team_name} & {manager} & {points} & {bonus_str} & {team_value:.1f} & {bank:.1f} & {total_hits} & {free_transfers} & {chips_str} \\"
            )

        summary_table = '\n'.join(summary_rows)

        squad_comparison_content = self._generate_squad_comparison_section(competitive_data)

        # Build player contribution treemap section
        treemap_sections = []
        for entry in competitive_data:
            entry_id = entry.get('entry_id', 0)
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', f'Team {entry_id}'))
            treemap_file = f"plots/treemap_team_{entry_id}.png"
            
            treemap_sections.append(rf"""
\begin{{minipage}}{{0.48\textwidth}}
\begin{{center}}
\textbf{{{team_name}}}\\[0.3cm]
\includegraphics[width=\textwidth]{{{treemap_file}}}
\end{{center}}
\end{{minipage}}""")
        
        # Arrange treemaps in 2-column layout
        treemap_rows = []
        for i in range(0, len(treemap_sections), 2):
            if i + 1 < len(treemap_sections):
                treemap_rows.append(treemap_sections[i] + r'\hfill' + treemap_sections[i + 1])
            else:
                treemap_rows.append(treemap_sections[i])
        
        treemap_content = '\n\n\\vspace{0.5cm}\n\n'.join(treemap_rows)

        # Build transfer activity section
        transfer_content = self._generate_transfer_activity_section(competitive_data)

        # Build historical transfer section with XI boxes
        transfer_history_content = self._generate_transfer_history_section(competitive_data)

        return rf"""
\newpage
\section{{Competitive Analysis}}

\subsection{{League Overview}}

Compare performance metrics across your mini-league competitors.

{{\footnotesize
\begin{{center}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l|l|r|r|r|r|r|r|l}}
\toprule
\textbf{{Team}} & \textbf{{Manager}} & \textbf{{Pts}} & \textbf{{Bonus (\% Share)}} & \textbf{{TV}} & \textbf{{ITB}} & \textbf{{Hits}} & \textbf{{FT}} & \textbf{{Chips}} \\
\midrule
{summary_table}
\bottomrule
\end{{tabular}}
}}%
\end{{center}}
}}

\subsection{{Gameweek Performance Comparison}}

Side-by-side comparison of net points (excluding hits) per gameweek. Chip usage is marked with symbols: * = Wildcard, s = Free Hit, D = Bench Boost, \^{{}} = Triple Captain.

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{plots/competitive_points_per_gw.png}}
\end{{center}}

\subsection{{Points Progression}}

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{plots/competitive_points_progression.png}}
\end{{center}}

\subsection{{Rank Progression}}

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{plots/competitive_rank_progression.png}}
\end{{center}}

\subsection{{Player Contribution Breakdown}}

How each player contributed to their manager's total points this season.

{treemap_content}

\vspace{{0.5cm}}

{transfer_content}

{transfer_history_content}

{squad_comparison_content}
"""

    def _generate_squad_comparison_section(self, competitive_data: List[Dict]) -> str:
        """Generate the squad comparison subsection for a set of teams."""
        if not competitive_data:
            return ""

        # Group players by position for each team
        squad_sections = []

        for pos_name, pos_code in [('GKP', 'GKP'), ('DEF', 'DEF'), ('MID', 'MID'), ('FWD', 'FWD')]:
            pos_row_parts = [rf"\textbf{{{pos_name}}}"]

            for entry in competitive_data:
                squad = entry.get('squad', [])
                # Filter players by position
                pos_players = [p for p in squad if p.get('position') == pos_code]
                # Sort by position_in_squad (XI first, then bench)
                pos_players_sorted = sorted(pos_players, key=lambda x: x.get('position_in_squad', 99))

                player_lines = []
                for p in pos_players_sorted:
                    name = self._escape_latex(p.get('name', 'Unknown'))
                    suffix = ''
                    if p.get('is_captain'):
                        suffix = ' (C)'
                    elif p.get('is_vice_captain'):
                        suffix = ' (VC)'

                    # Mark bench players
                    pos_in_squad = p.get('position_in_squad', 0)
                    if pos_in_squad > 11:
                        name = rf"\textcolor{{fplgray}}{{{name}{suffix}}}"
                    else:
                        name = rf"{name}{suffix}"

                    player_lines.append(name)

                cell_content = r' \newline '.join(player_lines) if player_lines else '-'
                pos_row_parts.append(cell_content)

            squad_sections.append(' & '.join(pos_row_parts) + r' \\')

        squad_table = '\n\\midrule\n'.join(squad_sections)

        # Build column spec for teams
        num_teams = len(competitive_data)
        team_header_parts = []
        for e in competitive_data:
            team_info = e.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', 'Team'))
            team_header_parts.append(rf"\textbf{{{team_name}}}")
        team_headers = ' & '.join(team_header_parts)

        # Calculate column width for squad table based on number of teams
        squad_col_width = max(1.8, 13.0 / (num_teams + 1))
        # Build column spec for squad table (e.g., l|p{2.0cm}|p{2.0cm}|...)
        squad_col_spec = 'l|' + '|'.join([f'p{{{squad_col_width:.1f}cm}}' for _ in range(num_teams)])

        return rf"""
\subsection{{Squad Comparison (GW{self.gameweek})}}

Current squad selections across all teams. Bench players shown in gray.

{{\scriptsize
\begin{{center}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{squad_col_spec}}}
\toprule
\textbf{{Pos}} & {team_headers} \\
\midrule
{squad_table}
\bottomrule
\end{{tabular}}
}}%
\end{{center}}
}}
"""

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        if not text:
            return ''
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]
        result = str(text)
        for char, replacement in replacements:
            result = result.replace(char, replacement)
        return result

    def _calculate_free_transfers(self, gw_history: List[Dict], chips_used: List[Dict]) -> str:
        """Calculate free transfers available for next gameweek.
        
        Uses the same methodology as generate_season_summary:
        - Start with 1 FT for GW2 (GW1 unlimited)
        - +1 per GW, cap at 5
        - Used transfers subtract from balance
        - WC/FH reset balance to 1 for next GW
        
        Args:
            gw_history: List of gameweek history entries
            chips_used: List of chips used this season
            
        Returns:
            String representation of free transfers (e.g., "2", "Unlimited")
        """
        if not gw_history:
            return "Unlimited"
        
        # Create chip usage map
        chip_usage = {c['event']: c['name'] for c in chips_used} if chips_used else {}
        
        # Sort history by event
        sorted_history = sorted(gw_history, key=lambda x: x['event'])
        
        # Find the last processed GW
        last_gw = sorted_history[-1]['event']
        
        # Iterate from GW2 to last_gw to simulate FT accumulation
        current_simulated_ft = 1  # Start with 1 FT available for GW2
        
        for gw in range(2, last_gw + 1):
            # Find history for this GW
            gw_data = next((g for g in sorted_history if g['event'] == gw), None)
            
            # Check if chip used in THIS GW
            chip = chip_usage.get(gw)
            
            if chip in ['wildcard', 'freehit']:
                # Chip used: Transfers don't cost FTs.
                # Balance for NEXT week (gw+1) resets to 1.
                current_simulated_ft = 1
            else:
                # Regular GW: Deduct transfers used
                transfers_made = gw_data.get('event_transfers', 0) if gw_data else 0
                
                current_simulated_ft -= transfers_made
                current_simulated_ft = max(0, current_simulated_ft)
                
                # Add 1 for next week
                current_simulated_ft += 1
                current_simulated_ft = min(5, current_simulated_ft)
        
        return str(current_simulated_ft)

    def _generate_transfer_activity_section(self, competitive_data: List[Dict]) -> str:
        """Generate transfer activity comparison section.
        
        Shows transfers made by each competitor in the current GW vs prior GW,
        with points impact and chip usage handling.
        
        Args:
            competitive_data: List of entry dicts with gw_transfers data.
            
        Returns:
            LaTeX string for the transfer activity subsection.
        """
        if not competitive_data:
            return ""
        
        # Determine prior GW for header
        prior_gw = max(1, self.gameweek - 1)
        
        # Build transfer activity rows for each team
        transfer_sections = []
        
        for entry in competitive_data:
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', 'Unknown'))
            manager = self._escape_latex(team_info.get('manager_name', 'Unknown'))
            gw_transfers = entry.get('gw_transfers', {})
            
            # Handle missing transfer data
            if not gw_transfers:
                transfer_sections.append(rf"""
\textbf{{{team_name}}} \textit{{({manager})}}\\
\quad No transfer data available\\[0.3cm]""")
                continue
            
            chip_used = gw_transfers.get('chip_used')
            prior_chip = gw_transfers.get('prior_chip_used')
            is_wildcard = gw_transfers.get('is_wildcard', False)
            is_free_hit = gw_transfers.get('is_free_hit', False)
            transfers_in = gw_transfers.get('transfers_in', [])
            transfers_out = gw_transfers.get('transfers_out', [])
            net_points = gw_transfers.get('net_points', 0)
            transfer_cost = gw_transfers.get('transfer_cost', 0)
            num_changes = gw_transfers.get('num_changes', 0)
            
            # Handle initial squad (GW1)
            if chip_used == 'initial_squad':
                transfer_sections.append(rf"""
\textbf{{{team_name}}} \textit{{({manager})}}\\
\quad \textcolor{{fplgray}}{{Initial Squad - No prior gameweek}}\\[0.3cm]""")
                continue
            
            # Handle Wildcard
            if is_wildcard:
                chip_text = r'\textcolor{gold}{\faStar} Wildcard Used'
                transfer_sections.append(rf"""
\textbf{{{team_name}}} \textit{{({manager})}}\\
\quad {chip_text} - Full squad rebuilt ({num_changes} players changed)\\[0.3cm]""")
                continue
            
            # Handle Free Hit
            if is_free_hit:
                chip_text = r'\textcolor{fplpink}{\faSync} Free Hit Used'
                transfer_sections.append(rf"""
\textbf{{{team_name}}} \textit{{({manager})}}\\
\quad {chip_text} - Temporary squad for this GW ({num_changes} players different)\\[0.3cm]""")
                continue
            
            # Normal transfers
            section_parts = []
            section_parts.append(rf"\textbf{{{team_name}}} \textit{{({manager})}}")
            
            # Add chip indicator if Bench Boost or Triple Captain
            if chip_used == 'bboost':
                section_parts.append(r'\quad \textcolor{fplgreen}{\faChair} Bench Boost Active')
            elif chip_used == '3xc':
                section_parts.append(r'\quad \textcolor{fplpurple}{\faCrown} Triple Captain Active')
            
            # Add note if prior GW had Free Hit (squad reverted)
            if prior_chip == 'freehit':
                section_parts.append(rf'\quad \textit{{\textcolor{{fplgray}}{{(Comparing to GW{prior_gw - 1} - squad reverted after Free Hit)}}}}')
            
            # No transfers case
            if not transfers_in and not transfers_out:
                section_parts.append(r'\quad No transfers made')
            else:
                # Transfers IN
                if transfers_in:
                    in_strs = []
                    for t in transfers_in:
                        name = self._escape_latex(t.get('name', 'Unknown'))
                        pos = t.get('position', 'UNK')
                        pts = t.get('gw_points', 0)
                        pts_color = 'rising' if pts > 0 else 'fplgray'
                        in_strs.append(rf"{name} ({pos}) \textcolor{{{pts_color}}}{{{pts} pts}}")
                    in_list = ', '.join(in_strs)
                    section_parts.append(rf'\quad \textcolor{{rising}}{{\faArrowUp}} IN: {in_list}')
                
                # Transfers OUT
                if transfers_out:
                    out_strs = []
                    for t in transfers_out:
                        name = self._escape_latex(t.get('name', 'Unknown'))
                        pos = t.get('position', 'UNK')
                        pts = t.get('gw_points', 0)
                        pts_color = 'falling' if pts > 0 else 'fplgray'
                        out_strs.append(rf"{name} ({pos}) \textcolor{{{pts_color}}}{{{pts} pts}}")
                    out_list = ', '.join(out_strs)
                    section_parts.append(rf'\quad \textcolor{{falling}}{{\faArrowDown}} OUT: {out_list}')
                
                # Net points and cost
                net_color = 'rising' if net_points > 0 else ('falling' if net_points < 0 else 'fplgray')
                net_sign = '+' if net_points > 0 else ''
                cost_str = f" | Cost: -{transfer_cost} pts" if transfer_cost > 0 else ""
                section_parts.append(rf'\quad \textbf{{Net: \textcolor{{{net_color}}}{{{net_sign}{net_points} pts}}}}{cost_str}')
            
            transfer_sections.append(r'\\'.join(section_parts) + r'\\[0.3cm]')
        
        transfer_content = '\n'.join(transfer_sections)
        
        return rf"""
\subsection{{Transfer Activity (GW{self.gameweek} vs GW{prior_gw})}}

Transfers made between the current and prior gameweek. Points shown are what the player scored in GW{self.gameweek}.

{{\small
{transfer_content}
}}
"""

    def _generate_transfer_history_section(self, competitive_data: List[Dict]) -> str:
        """Generate historical transfer activity section with 5 GW columns.
        
        Shows starting XI progression as a horizontal table with one column per GW.
        Green = transferred in that GW, Red = transferred out that GW.
        
        Args:
            competitive_data: List of entry dicts with transfer_history data.
            
        Returns:
            LaTeX string for the historical transfer section.
        """
        if not competitive_data:
            return ""
        
        team_sections = []
        
        for entry in competitive_data:
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', 'Unknown'))
            transfer_history = entry.get('transfer_history', {})
            
            if not transfer_history:
                continue
            
            gw_range = transfer_history.get('gw_range', [])
            gw_squads_data = transfer_history.get('gw_squads_data', {})
            chips_timeline = transfer_history.get('chips_timeline', {})
            
            if not gw_range or not gw_squads_data:
                continue
            
            # Build the 5-column table
            table_content = self._build_gw_progression_table(gw_range, gw_squads_data, chips_timeline)
            
            team_sections.append(rf"""
\noindent\textbf{{{team_name}}}

{table_content}
""")
        
        team_content = '\n\n\\vspace{0.3cm}\\hrule\\vspace{0.3cm}\n\n'.join(team_sections)
        
        return rf"""
\subsection{{Squad Evolution (Past 5 Gameweeks)}}

Starting XI across recent gameweeks. \colorbox{{rising!30}}{{\small IN}} = transferred in, \colorbox{{falling!30}}{{\small OUT}} = transferred out that GW.

{team_content}
"""

    def _build_gw_progression_table(self, gw_range: List[int], gw_squads_data: Dict, chips_timeline: Dict) -> str:
        """Build a table showing starting XI across 5 gameweeks.
        
        Args:
            gw_range: List of gameweek numbers.
            gw_squads_data: Dict mapping GW to squad data.
            chips_timeline: Dict mapping GW to chip name.
            
        Returns:
            LaTeX string for the progression table.
        """
        num_gws = len(gw_range)
        if num_gws == 0:
            return ""
        
        # Build header row with GW numbers and chip indicators
        header_cells = []
        for gw in gw_range:
            chip = chips_timeline.get(gw)
            chip_str = ''
            if chip == 'wildcard':
                chip_str = r' \faStar'
            elif chip == 'freehit':
                chip_str = r' \faSync'
            elif chip == 'bboost':
                chip_str = r' \faChair'
            elif chip == '3xc':
                chip_str = r' \faCrown'
            header_cells.append(rf"\textbf{{GW{gw}}}{chip_str}")
        
        header_row = ' & '.join(header_cells)
        
        # Build rows for each position group
        position_rows = []
        for pos_name in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_cells = []
            for gw in gw_range:
                gw_data = gw_squads_data.get(gw, {})
                xi = gw_data.get('xi', [])
                transfers_out = gw_data.get('transfers_out', [])
                
                # Get players in this position
                pos_players = [p for p in xi if p.get('position') == pos_name]
                
                # Get players transferred out in this position
                out_players = [p for p in transfers_out if p.get('position') == pos_name]
                
                player_strs = []
                for p in pos_players:
                    name = self._escape_latex(p.get('name', '?'))
                    is_new = p.get('is_new_this_gw', False)
                    is_captain = p.get('is_captain', False)
                    is_vc = p.get('is_vice_captain', False)
                    
                    suffix = ''
                    if is_captain:
                        suffix = ' (C)'
                    elif is_vc:
                        suffix = ' (VC)'
                    
                    if is_new:
                        player_strs.append(rf"\colorbox{{rising!30}}{{{name}{suffix}}}")
                    else:
                        player_strs.append(rf"{name}{suffix}")
                
                # Add transferred out players (strikethrough in red)
                for p in out_players:
                    name = self._escape_latex(p.get('name', '?'))
                    player_strs.append(rf"\colorbox{{falling!30}}{{\sout{{{name}}}}}")
                
                cell_content = r' \newline '.join(player_strs) if player_strs else '-'
                pos_cells.append(cell_content)
            
            pos_row = ' & '.join(pos_cells)
            position_rows.append(rf"\textbf{{{pos_name}}} & {pos_row} \\")
        
        # Build bench row
        bench_cells = []
        for gw in gw_range:
            gw_data = gw_squads_data.get(gw, {})
            bench = gw_data.get('bench', [])
            bench_names = []
            for p in bench[:2]:  # Show only first 2 bench players to save space
                name = self._escape_latex(p.get('name', '?'))
                is_new = p.get('is_new_this_gw', False)
                if is_new:
                    bench_names.append(rf"\colorbox{{rising!30}}{{{name}}}")
                else:
                    bench_names.append(name)
            if len(bench) > 2:
                bench_names.append('...')
            bench_cells.append(r'\tiny{' + ', '.join(bench_names) + '}')
        
        bench_row = ' & '.join(bench_cells)
        
        # Calculate column width
        col_width = max(2.0, 14.0 / (num_gws + 1))
        col_spec = f"l|{'|'.join([f'p{{{col_width:.1f}cm}}' for _ in gw_range])}"
        
        return rf"""
{{\scriptsize
\begin{{center}}
\begin{{tabular}}{{{col_spec}}}
\toprule
\textbf{{Pos}} & {header_row} \\
\midrule
{chr(10).join(position_rows)}
\midrule
\textbf{{Bench}} & {bench_row} \\
\bottomrule
\end{{tabular}}
\end{{center}}
}}
"""

    def generate_wildcard_team_section(self, wildcard_team: Dict) -> str:
        """Generate Wildcard ideal team section.
        
        Shows optimized 15-player squad built with current budget,
        including starting XI, bench, formation, and captain picks.
        
        Args:
            wildcard_team: Dict from WildcardOptimizer.build_squad() containing:
                - budget: {total, spent, remaining}
                - squad: List of player dicts
                - starting_xi: List of starting players
                - bench: List of bench players
                - formation: Formation string (e.g., '4-4-2')
                - captain: Captain dict
                - vice_captain: Vice captain dict
        
        Returns:
            LaTeX string for the wildcard section.
        """
        if not wildcard_team:
            return ""
        
        budget = wildcard_team.get('budget', {})
        squad = wildcard_team.get('squad', [])
        starting_xi = wildcard_team.get('starting_xi', [])
        bench = wildcard_team.get('bench', [])
        formation = wildcard_team.get('formation', '?-?-?')
        captain = wildcard_team.get('captain', {})
        vice_captain = wildcard_team.get('vice_captain', {})
        ev_analysis = wildcard_team.get('ev_analysis', {})
        
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\section{Wildcard Draft (Current Budget)}")
        
        # Expected Value Analysis box (if available)
        if ev_analysis:
            current_xp = ev_analysis.get('current_squad_xp', 0)
            optimized_xp = ev_analysis.get('optimized_xp', 0)
            gain = ev_analysis.get('potential_gain', 0)
            horizon = ev_analysis.get('horizon', '5 GWs')
            
            gain_color = 'fplgreen' if gain > 0 else 'fplpink'
            gain_sign = '+' if gain > 0 else ''
            
            sections.append(rf"""
\subsection{{Expected Value Analysis (Next {horizon})}}

\begin{{center}}
\begin{{tikzpicture}}
    \node[draw=fplgreen,line width=2pt,rounded corners=8pt,inner sep=12pt,fill=fplgreen!10] {{
        \begin{{tabular}}{{ccc}}
            \textcolor{{fplpurple}}{{\Large\textbf{{{current_xp:.1f}}}}} &
            \textcolor{{fplpurple}}{{\Large\textbf{{{optimized_xp:.1f}}}}} &
            \textcolor{{{gain_color}}}{{\Large\textbf{{{gain_sign}{gain:.1f}}}}}\\[3pt]
            \textcolor{{fplgray}}{{Current Squad xP}} &
            \textcolor{{fplgray}}{{Optimized xP}} &
            \textcolor{{fplgray}}{{Potential Gain}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}
""")
        
        # Budget summary
        total = budget.get('total', 100.0)
        spent = budget.get('spent', 0.0)
        remaining = budget.get('remaining', 0.0)
        
        sections.append(rf"""
\subsection{{Budget Overview}}

\begin{{center}}
\begin{{tikzpicture}}
    \node[draw=fplpurple,line width=2pt,rounded corners=8pt,inner sep=12pt,fill=fplpurple!5] {{
        \begin{{tabular}}{{ccc}}
            \textcolor{{fplpurple}}{{\Large\textbf{{\pounds{total:.1f}m}}}} &
            \textcolor{{fplpurple}}{{\Large\textbf{{\pounds{spent:.1f}m}}}} &
            \textcolor{{fplgreen}}{{\Large\textbf{{\pounds{remaining:.1f}m}}}}\\[3pt]
            \textcolor{{fplgray}}{{Total Budget}} &
            \textcolor{{fplgray}}{{Spent}} &
            \textcolor{{fplgray}}{{In The Bank}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}
""")
        
        # Formation and Captain info
        cap_name = self._escape_latex(captain.get('name', 'Unknown'))
        vc_name = self._escape_latex(vice_captain.get('name', 'Unknown'))
        
        sections.append(rf"""
\subsection{{Suggested Setup}}

\begin{{center}}
\begin{{tabular}}{{ll}}
\textbf{{Formation:}} & {formation} \\
\textbf{{Captain:}} & {cap_name} \\
\textbf{{Vice Captain:}} & {vc_name} \\
\end{{tabular}}
\end{{center}}
""")
        
        # Starting XI table with FDR
        sections.append(r"\subsection{Starting XI}")
        sections.append(r"\textit{FDR colors: \colorbox{fplgreen!40}{Easy (1-2)} \colorbox{gold!40}{Medium (3)} \colorbox{orange!40}{Hard (4)} \colorbox{fplpink!40}{Very Hard (5)}}")
        sections.append(r"""
\begin{center}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|c|c|r|r|r|r|ccccc}
\toprule
\textbf{Player} & \textbf{Pos} & \textbf{Team} & \textbf{Price} & \textbf{PPG} & \textbf{5-GW xP} & \textbf{Score} & \multicolumn{5}{c}{\textbf{Next 5 Fixtures (FDR)}} \\
\midrule
""")
        
        # Group XI by position
        xi_by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for p in starting_xi:
            pos = p.get('position', 'UNK')
            if pos in xi_by_pos:
                xi_by_pos[pos].append(p)
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            for p in xi_by_pos[pos]:
                name = self._escape_latex(p.get('name', 'Unknown'))
                team = self._escape_latex(str(p.get('team', 'UNK')))
                price = p.get('price', 0.0)
                ppg = p.get('ppg', 0.0)
                xp_5gw = p.get('xp_5gw', 0.0)
                score = p.get('score', 0.0)
                fixtures = p.get('fixtures', [])
                
                # Mark captain/VC
                suffix = ''
                if p.get('name') == captain.get('name'):
                    suffix = r' \textbf{(C)}'
                elif p.get('name') == vice_captain.get('name'):
                    suffix = r' \textbf{(VC)}'
                
                # Format FDR fixtures with color coding
                fdr_cells = []
                for i in range(5):
                    if i < len(fixtures):
                        fix = fixtures[i]
                        opp = fix.get('opponent', '?')
                        fdr = fix.get('fdr', 3)
                        is_home = fix.get('is_home', False)
                        
                        if fdr <= 2:
                            fdr_color = 'fplgreen!40'
                        elif fdr == 3:
                            fdr_color = 'gold!40'
                        elif fdr == 4:
                            fdr_color = 'orange!40'
                        else:
                            fdr_color = 'fplpink!40'
                        
                        venue = '' if is_home else '(a)'
                        fdr_cells.append(rf"\cellcolor{{{fdr_color}}}{opp}{venue}")
                    else:
                        fdr_cells.append('-')
                
                fdr_str = ' & '.join(fdr_cells)
                sections.append(rf"{name}{suffix} & {pos} & {team} & \pounds{price:.1f}m & {ppg:.2f} & {xp_5gw:.1f} & {score:.1f} & {fdr_str} \\")
        
        # XI totals
        xi_cost = sum(p.get('price', 0) for p in starting_xi)
        xi_xp_5gw = sum(p.get('xp_5gw', 0) for p in starting_xi)
        xi_score = sum(p.get('score', 0) for p in starting_xi)
        
        sections.append(rf"""
\midrule
\textbf{{Total XI}} & & & \textbf{{\pounds{xi_cost:.1f}m}} & & \textbf{{{xi_xp_5gw:.1f}}} & \textbf{{{xi_score:.1f}}} & & & & & \\
\bottomrule
\end{{tabular}}
\end{{center}}
""")
        
        # Bench table with FDR
        sections.append(r"\subsection{Bench}")
        sections.append(r"""
\begin{center}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l|c|c|r|r|r|ccccc}
\toprule
\textbf{Player} & \textbf{Pos} & \textbf{Team} & \textbf{Price} & \textbf{5-GW xP} & \textbf{Score} & \multicolumn{5}{c}{\textbf{Next 5 Fixtures}} \\
\midrule
""")
        
        for p in bench:
            name = self._escape_latex(p.get('name', 'Unknown'))
            pos = p.get('position', 'UNK')
            team = self._escape_latex(str(p.get('team', 'UNK')))
            price = p.get('price', 0.0)
            xp_5gw = p.get('xp_5gw', 0.0)
            score = p.get('score', 0.0)
            fixtures = p.get('fixtures', [])
            
            # Format FDR fixtures
            fdr_cells = []
            for i in range(5):
                if i < len(fixtures):
                    fix = fixtures[i]
                    opp = fix.get('opponent', '?')
                    fdr = fix.get('fdr', 3)
                    is_home = fix.get('is_home', False)
                    
                    if fdr <= 2:
                        fdr_color = 'fplgreen!40'
                    elif fdr == 3:
                        fdr_color = 'gold!40'
                    elif fdr == 4:
                        fdr_color = 'orange!40'
                    else:
                        fdr_color = 'fplpink!40'
                    
                    venue = '' if is_home else '(a)'
                    fdr_cells.append(rf"\cellcolor{{{fdr_color}}}{opp}{venue}")
                else:
                    fdr_cells.append('-')
            
            fdr_str = ' & '.join(fdr_cells)
            sections.append(rf"{name} & {pos} & {team} & \pounds{price:.1f}m & {xp_5gw:.1f} & {score:.1f} & {fdr_str} \\")
        
        bench_cost = sum(p.get('price', 0) for p in bench)
        bench_xp_5gw = sum(p.get('xp_5gw', 0) for p in bench)
        
        sections.append(rf"""
\midrule
\textbf{{Bench Total}} & & & \textbf{{\pounds{bench_cost:.1f}m}} & \textbf{{{bench_xp_5gw:.1f}}} & & & & & & \\
\bottomrule
\end{{tabular}}
\end{{center}}
""")
        
        # Full squad by position
        sections.append(r"\subsection{Full Squad by Position}")
        
        squad_by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for p in squad:
            pos = p.get('position', 'UNK')
            if pos in squad_by_pos:
                squad_by_pos[pos].append(p)
        
        for pos, pos_name in [('GKP', 'Goalkeepers'), ('DEF', 'Defenders'), 
                              ('MID', 'Midfielders'), ('FWD', 'Forwards')]:
            players = squad_by_pos[pos]
            if not players:
                continue
            
            pos_cost = sum(p.get('price', 0) for p in players)
            player_list = ', '.join([
                self._escape_latex(p.get('name', 'Unknown')) + 
                f" (\\pounds{p.get('price', 0):.1f}m)"
                for p in players
            ])
            
            sections.append(rf"""
\textbf{{{pos_name}}} ({len(players)}): {player_list}\\
\textit{{Position spend: \pounds{pos_cost:.1f}m}}

""")
        
        # Strategy note
        sections.append(r"""
\subsection{Strategy Notes}

This Wildcard draft prioritizes \textbf{season-balanced} selection:
\begin{itemize}
    \item Players chosen based on long-term metrics (PPG, total points, minutes)
    \item Cheap bench strategy to maximize starting XI quality
    \item Fixtures are considered but don't dominate selection
    \item All players are currently available (no injuries/suspensions)
\end{itemize}

\textit{Note: This is a suggested squad based on current data. Always verify player availability and news before activating your Wildcard.}
""")
        
        return '\n'.join(sections)

    def generate_free_hit_team_section(self, free_hit_team: Dict) -> str:
        """Generate Free Hit draft section optimized for a single gameweek.
        
        Shows optimized 15-player squad built with current budget,
        using ep_next (expected points) and league ownership data
        to find differentials that beat your league.
        
        Args:
            free_hit_team: Dict from FreeHitOptimizer.build_squad() containing:
                - budget: {total, spent, remaining}
                - squad: List of player dicts
                - starting_xi: List of starting players
                - bench: List of bench players
                - formation: Formation string (e.g., '4-4-2')
                - captain: Captain dict
                - vice_captain: Vice captain dict
                - target_gw: Target gameweek number
                - strategy: Strategy mode (safe/balanced/aggressive)
                - league_analysis: {sample_size, differentials, template_picks}
        
        Returns:
            LaTeX string for the Free Hit section.
        """
        if not free_hit_team:
            return ""
        
        # Get all permutations or create single-item list for backward compatibility
        all_permutations = free_hit_team.get('all_permutations', [free_hit_team])
        target_gw = free_hit_team.get('target_gw', '?')
        primary_strategy = free_hit_team.get('strategy', 'balanced')
        
        sections = []
        sections.append(r"\newpage")
        sections.append(r"\section{Free Hit Draft}")
        
        sections.append(rf"""
\textit{{Optimized for \textbf{{GW{target_gw}}} - Three strategic approaches to consider based on your league position and risk appetite.}}
""")
        
        # =========================================================
        # COMPARISON TABLE - All 3 Strategies Overview
        # =========================================================
        sections.append(r"\subsection{Strategy Comparison}")
        
        strategy_colors = {
            'safe': 'fplgreen',
            'balanced': 'gold',
            'aggressive': 'fplpink'
        }
        strategy_icons = {
            'safe': r'\faShield',
            'balanced': r'\faBalanceScale',
            'aggressive': r'\faBolt'
        }
        
        sections.append(r"""
\begin{center}
\small
\begin{tabular}{l|c|c|c}
\toprule
\textbf{Metric} & \textbf{Safe (Template)} & \textbf{Balanced} & \textbf{Aggressive (Diff)} \\
\midrule
""")
        
        # Build comparison rows
        for perm in all_permutations:
            strat = perm.get('strategy', 'balanced')
            budget = perm.get('budget', {})
            ev = perm.get('ev_analysis', {})
            captain = perm.get('captain', {})
            formation = perm.get('formation', '?-?-?')
        
        # Extract metrics for each strategy
        metrics = {}
        for perm in all_permutations:
            strat = perm.get('strategy', 'balanced')
            metrics[strat] = {
                'spent': perm.get('budget', {}).get('spent', 0),
                'itb': perm.get('budget', {}).get('remaining', 0),
                'xp': perm.get('ev_analysis', {}).get('optimized_xp', 0),
                'gain': perm.get('ev_analysis', {}).get('potential_gain', 0),
                'captain': perm.get('captain', {}).get('name', 'Unknown'),
                'formation': perm.get('formation', '?-?-?'),
            }
        
        # Add rows
        safe = metrics.get('safe', {})
        balanced = metrics.get('balanced', {})
        aggressive = metrics.get('aggressive', {})
        
        current_xp = all_permutations[0].get('ev_analysis', {}).get('current_squad_xp', 0) if all_permutations else 0
        
        sections.append(rf"Expected Points (xP) & {safe.get('xp', 0):.1f} & {balanced.get('xp', 0):.1f} & {aggressive.get('xp', 0):.1f} \\")
        
        # Gain row with color coding
        def gain_color(g):
            if g > 0:
                return rf"\textcolor{{fplgreen}}{{+{g:.1f}}}"
            elif g < 0:
                return rf"\textcolor{{fplpink}}{{{g:.1f}}}"
            else:
                return f"{g:.1f}"
        
        sections.append(rf"vs Current Squad & {gain_color(safe.get('gain', 0))} & {gain_color(balanced.get('gain', 0))} & {gain_color(aggressive.get('gain', 0))} \\")
        sections.append(rf"Budget Spent & \pounds{safe.get('spent', 0):.1f}m & \pounds{balanced.get('spent', 0):.1f}m & \pounds{aggressive.get('spent', 0):.1f}m \\")
        sections.append(rf"In The Bank & \pounds{safe.get('itb', 0):.1f}m & \pounds{balanced.get('itb', 0):.1f}m & \pounds{aggressive.get('itb', 0):.1f}m \\")
        sections.append(rf"Formation & {safe.get('formation', '?')} & {balanced.get('formation', '?')} & {aggressive.get('formation', '?')} \\")
        sections.append(rf"Captain & {self._escape_latex(safe.get('captain', '?'))} & {self._escape_latex(balanced.get('captain', '?'))} & {self._escape_latex(aggressive.get('captain', '?'))} \\")
        
        sections.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")
        
        sections.append(rf"""
\textit{{Your current squad projected: \textbf{{{current_xp:.1f} xP}} for GW{target_gw}}}
""")
        
        # =========================================================
        # DETAILED SQUADS - One subsection per strategy
        # =========================================================
        strategy_descriptions = {
            'safe': ('Template Squad', 'High ownership players to protect your rank. Minimize risk by matching the league template.'),
            'balanced': ('Balanced Squad', 'Mix of template picks and differentials. Good risk-reward balance.'),
            'aggressive': ('Differential Squad', 'Low ownership picks to gain ground. Higher risk but bigger potential reward.')
        }
        
        for perm in all_permutations:
            strat = perm.get('strategy', 'balanced')
            strat_title, strat_desc = strategy_descriptions.get(strat, ('Squad', ''))
            
            starting_xi = perm.get('starting_xi', [])
            bench = perm.get('bench', [])
            captain = perm.get('captain', {})
            vice_captain = perm.get('vice_captain', {})
            budget = perm.get('budget', {})
            league_analysis = perm.get('league_analysis', {})
            sample_size = league_analysis.get('sample_size', 0)
            
            # Strategy header with colored box
            strat_color = strategy_colors.get(strat, 'fplgray')
            budget_spent = budget.get('spent', 0)
            budget_itb = budget.get('remaining', 0)
            formation_str = perm.get('formation', '?')
            captain_name = self._escape_latex(captain.get('name', '?'))
            
            sections.append(rf"""
\subsection{{{strat_title} ({strat.capitalize()})}}
\textit{{{strat_desc}}}

\begin{{center}}
\begin{{tikzpicture}}
    \node[draw={strat_color},line width=1.5pt,rounded corners=6pt,inner sep=8pt,fill={strat_color}!10] {{
        \begin{{tabular}}{{cccc}}
            \textcolor{{{strat_color}}}{{\textbf{{\pounds{budget_spent:.1f}m}}}} &
            \textcolor{{{strat_color}}}{{\textbf{{\pounds{budget_itb:.1f}m}}}} &
            \textcolor{{{strat_color}}}{{\textbf{{{formation_str}}}}} &
            \textcolor{{{strat_color}}}{{\textbf{{{captain_name} (C)}}}}\\[2pt]
            \textcolor{{fplgray}}{{\small Spent}} &
            \textcolor{{fplgray}}{{\small ITB}} &
            \textcolor{{fplgray}}{{\small Formation}} &
            \textcolor{{fplgray}}{{\small Captain}}
        \end{{tabular}}
    }};
\end{{tikzpicture}}
\end{{center}}
""")
            
            # Starting XI table
            if sample_size > 0:
                sections.append(rf"\textit{{\footnotesize League ownership based on {sample_size} teams. FDR: \colorbox{{fplgreen!40}}{{Easy}} \colorbox{{gold!40}}{{Med}} \colorbox{{orange!40}}{{Hard}} \colorbox{{fplpink!40}}{{V.Hard}}}}")
            
            sections.append(rf"""
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l|c|c|r|r|r|c}}
\toprule
\textbf{{Player}} & \textbf{{Pos}} & \textbf{{Team}} & \textbf{{Price}} & \textbf{{xP}} & \textbf{{Own\%}} & \textbf{{GW{target_gw}}} \\
\midrule
""")
            
            # Group XI by position
            xi_by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            for p in starting_xi:
                pos = p.get('position', 'UNK')
                if pos in xi_by_pos:
                    xi_by_pos[pos].append(p)
            
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                for p in xi_by_pos[pos]:
                    name = self._escape_latex(p.get('name', 'Unknown'))
                    team = self._escape_latex(str(p.get('team', 'UNK')))
                    price = p.get('price', 0.0)
                    ep_next = p.get('ep_next', 0.0)
                    if ep_next <= 0:
                        ep_next = p.get('xp_5gw', 0.0) / 5.0
                    league_own = p.get('league_ownership', 0.0)
                    fixtures = p.get('fixtures', [])
                    
                    suffix = ''
                    if p.get('name') == captain.get('name'):
                        suffix = r' \textbf{(C)}'
                    elif p.get('name') == vice_captain.get('name'):
                        suffix = r' \textbf{(VC)}'
                    
                    if fixtures:
                        fix = fixtures[0]
                        opp = fix.get('opponent', '?')
                        fdr = fix.get('fdr', 3)
                        is_home = fix.get('is_home', False)
                        
                        if fdr <= 2:
                            fdr_color = 'fplgreen!40'
                        elif fdr == 3:
                            fdr_color = 'gold!40'
                        elif fdr == 4:
                            fdr_color = 'orange!40'
                        else:
                            fdr_color = 'fplpink!40'
                        
                        venue = '' if is_home else '(a)'
                        fdr_cell = rf"\cellcolor{{{fdr_color}}}{opp}{venue}"
                    else:
                        fdr_cell = '-'
                    
                    sections.append(rf"{name}{suffix} & {pos} & {team} & \pounds{price:.1f}m & {ep_next:.1f} & {league_own:.0f}\% & {fdr_cell} \\")
            
            xi_cost = sum(p.get('price', 0) for p in starting_xi)
            xi_xp = sum(p.get('ep_next', 0) or (p.get('xp_5gw', 0) / 5.0) for p in starting_xi)
            
            sections.append(rf"""
\midrule
\textbf{{Total XI}} & & & \textbf{{\pounds{xi_cost:.1f}m}} & \textbf{{{xi_xp:.1f}}} & & \\
\bottomrule
\end{{tabular}}
\end{{center}}
""")
            
            # Bench (compact)
            if bench:
                bench_names = ', '.join([self._escape_latex(p.get('name', '?')) for p in bench])
                bench_cost = sum(p.get('price', 0) for p in bench)
                sections.append(rf"\textit{{\footnotesize \textbf{{Bench}} (\pounds{bench_cost:.1f}m): {bench_names}}}")
                sections.append("")
        
        # =========================================================
        # STRATEGY NOTES
        # =========================================================
        sections.append(r"""
\subsection{Strategy Guide}

\begin{itemize}
    \item \textbf{Safe/Template}: Best if you're protecting a lead in your mini-league. Minimizes variance.
    \item \textbf{Balanced}: Good all-round choice. Template core with upside from differentials.
    \item \textbf{Aggressive/Differential}: Best if you're chasing. Higher risk but can gain significant ground.
\end{itemize}

\textit{Note: Verify player availability and news before activating your Free Hit chip.}
""")
        
        return '\n'.join(sections)

    def generate_top_global_teams(self) -> str:
        """Generate section showing top globally-ranked FPL managers.
        
        Returns:
            LaTeX string for the top global teams analysis section.
        """
        try:
            top_teams = get_top_global_teams(n=TOP_GLOBAL_COUNT)
            
            if not top_teams:
                return ""
            
            # Build table rows
            table_rows = []
            for team in top_teams:
                rank = team.get('rank', 0)
                team_name = self._escape_latex(team.get('team_name', 'Unknown'))
                manager = self._escape_latex(team.get('manager_name', 'Unknown'))
                points = team.get('total_points', 0)
                
                table_rows.append(rf"{rank} & {team_name} & {manager} & {points} \\")
            
            table_content = "\n".join(table_rows)
            
            return rf"""
\newpage
\section{{Top Global Managers}}

\textit{{Learn from the best - these are the current top {TOP_GLOBAL_COUNT} ranked FPL managers globally.}}

\begin{{center}}
\begin{{tabular}}{{c|l|l|r}}
\toprule
\textbf{{Rank}} & \textbf{{Team Name}} & \textbf{{Manager}} & \textbf{{Points}} \\
\midrule
{table_content}
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection{{Key Takeaways}}

\begin{{itemize}}
\item Elite managers consistently make optimal captain picks and avoid transfer hits
\item Chip timing is crucial - top managers save chips for DGW/BGW opportunities
\item Template adherence with well-timed differentials separates good from great
\item Study their squad selections for insights into value picks
\end{{itemize}}
"""
        except Exception:
            return ""

    def generate_global_competitive_analysis(self, top_global_data: List[Dict]) -> str:
        """Generate full competitive analysis comparing user vs top global managers.
        
        Args:
            top_global_data: List of dicts with team_info, gw_history, squad, season_history.
            
        Returns:
            LaTeX string for the global competitive analysis section.
        """
        if not top_global_data:
            return ""
        
        # Build summary table
        summary_rows = []
        for entry in top_global_data:
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', 'Unknown'))
            manager = self._escape_latex(team_info.get('manager_name', 'Unknown'))
            points = team_info.get('overall_points', 0)
            rank = team_info.get('overall_rank', 0)
            
            # Calculate bonus points from captaincy and chips
            season_history = entry.get('season_history', [])
            chips_used = entry.get('chips_used', [])
            bonus_pts, bonus_pct = self._calculate_bonus_points(season_history, chips_used)
            bonus_str = f"{bonus_pts} ({bonus_pct:.1f}\\%)"
            
            summary_rows.append(f"    {team_name} & {manager} & {points:,} & {rank:,} & {bonus_str} \\\\")
        
        summary_table = "\n".join(summary_rows)
        
        # Build treemap sections (2 per row)
        treemap_sections = []
        for entry in top_global_data:
            entry_id = entry.get('entry_id', 0)
            team_info = entry.get('team_info', {})
            team_name = self._escape_latex(team_info.get('team_name', f'Team {entry_id}'))
            
            # Use global_ prefix for filenames
            treemap_file = f'global_treemap_team_{entry_id}.png'
            treemap_sections.append(rf"""
\begin{{minipage}}{{0.48\textwidth}}
    \centering
    \includegraphics[width=\textwidth]{{{self.plot_dir}/{treemap_file}}}
    \captionof*{{figure}}{{{team_name}}}
\end{{minipage}}""")
        
        # Pair treemaps for display
        treemap_rows = []
        for i in range(0, len(treemap_sections), 2):
            if i + 1 < len(treemap_sections):
                treemap_rows.append(treemap_sections[i] + r'\hfill' + treemap_sections[i + 1])
            else:
                treemap_rows.append(treemap_sections[i])
        
        treemap_content = '\n\n\\vspace{0.5cm}\n\n'.join(treemap_rows)

        # Reuse the same transfer + squad section builders as the mini-league competitive section
        transfer_content = self._generate_transfer_activity_section(top_global_data)
        squad_evolution_content = self._generate_transfer_history_section(top_global_data)
        squad_comparison_content = self._generate_squad_comparison_section(top_global_data)
        
        return rf"""
\newpage
\section{{Benchmarking: Top {TOP_GLOBAL_COUNT} Global Managers}}

\textit{{See how your team compares to the top {TOP_GLOBAL_COUNT} managers in the world. Learn from their strategies.}}

\subsection{{Summary Comparison}}

\begin{{tabularx}}{{\textwidth}}{{Xllrr}}
\toprule
\textbf{{Team}} & \textbf{{Manager}} & \textbf{{Points}} & \textbf{{Rank}} & \textbf{{Bonus (\% Share)}} \\
\midrule
{summary_table}
\bottomrule
\end{{tabularx}}

\subsection{{Points Progression}}

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{{self.plot_dir}/global_top_points_progression.png}}
\end{{center}}

\subsection{{Rank Progression}}

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{{self.plot_dir}/global_top_rank_progression.png}}
\end{{center}}

\subsection{{Points per GW Comparison}}

\begin{{center}}
\includegraphics[width=0.95\textwidth]{{{self.plot_dir}/global_top_points_per_gw.png}}
\end{{center}}

\subsection{{Player Contribution Treemaps}}

{treemap_content}

\vspace{{0.5cm}}

{transfer_content}

{squad_evolution_content}

{squad_comparison_content}

\subsection{{Key Insights}}
\begin{{itemize}}
    \item \textbf{{Gap Analysis}}: Study the points gap per gameweek to identify where elite managers gain.
    \item \textbf{{Chip Strategy}}: Top managers often save chips for BGW/DGW opportunities.
    \item \textbf{{Captain Picks}}: Elite players typically captain high-ceiling options.
    \item \textbf{{Transfer Discipline}}: The best managers minimize transfer hits—quality over quantity.
\end{{itemize}}
"""

    def generate_footer(self) -> str:
        """Generate document footer."""
        return rf"""
\vspace{{1cm}}
\begin{{center}}
\rule{{0.5\textwidth}}{{1pt}}\\[0.5cm]
{{\small\color{{fplgray}} Generated by FPL Report Generator}}\\
{{\small\color{{fplgray}} Data Source: Official FPL API | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}}}
\end{{center}}

\end{{document}}
"""

    def compile_report(self, team_info: Dict, gw_history: List[Dict],
                       squad: List[Dict], squad_analysis: List[Dict],
                       recommendations: List[Dict], captain_picks: List[Dict],
                       chips_used: List[Dict], transfers: List[Dict] = None,
                       multi_week_strategy: Dict = None,
                       competitive_data: List[Dict] = None,
                       wildcard_team: Dict = None,
                       free_hit_team: Dict = None,
                       season_history: List[Dict] = None,
                       top_global_data: List[Dict] = None) -> str:
        """Compile the complete LaTeX report.

        Args:
            team_info: Team information dictionary.
            gw_history: List of gameweek history entries.
            squad: Current squad with player details.
            squad_analysis: Deep analysis of each player.
            recommendations: Transfer recommendations (legacy, used if no multi_week_strategy).
            captain_picks: Captain pick suggestions.
            chips_used: List of chips used.
            transfers: List of transfers made this season.
            multi_week_strategy: Multi-week transfer strategy from TransferStrategyPlanner.
            competitive_data: Optional list of competitor team data for comparison.
            wildcard_team: Optional Wildcard draft squad from WildcardOptimizer.
            free_hit_team: Optional Free Hit draft squad from FreeHitOptimizer.
            season_history: Full season history with squad data per GW.

        Returns:
            Complete LaTeX document as string.
        """
        team_name = team_info.get('team_name', 'Unknown')
        season = team_info.get('season', DEFAULT_SEASON)

        # Choose transfer strategy section based on available data
        if multi_week_strategy:
            transfer_section = self.generate_multi_week_strategy(multi_week_strategy, captain_picks)
        else:
            transfer_section = self.generate_transfer_recommendations(recommendations, captain_picks)

        parts = [
            self.generate_preamble(),
            self.generate_header_footer(team_name, season),
            self.generate_title_page(team_info, gw_history),
            self.generate_season_summary(team_info, gw_history, chips_used, transfers),
            self.generate_gw_performance_chart(gw_history),
            self.generate_rank_progression(gw_history),
            self.generate_player_points_breakdown(squad_analysis),
            self.generate_contribution_chart(),
            self.generate_hindsight_analysis(),
            self.generate_position_breakdown(squad_analysis, season_history),
            self.generate_formation_diagram(squad),
            self.generate_player_deep_dives(squad_analysis),
            self.generate_transfer_history(transfers or []),
            transfer_section,
        ]

        # Add Wildcard draft section if data is provided
        if wildcard_team:
            parts.append(self.generate_wildcard_team_section(wildcard_team))

        # Add Free Hit draft section if data is provided
        if free_hit_team:
            parts.append(self.generate_free_hit_team_section(free_hit_team))

        parts.extend([
            self.generate_chip_strategy(chips_used, gw_history),
            self.generate_insights(squad_analysis, gw_history),
        ])

        # Add competitive analysis section at the end if data is provided
        if competitive_data:
            parts.append(self.generate_competitive_analysis(competitive_data))
        
        # Add full global competitive analysis (vs top 5 global managers) if data is provided
        if top_global_data:
            parts.append(self.generate_global_competitive_analysis(top_global_data))
        else:
            # Fallback to simple table
            parts.append(self.generate_top_global_teams())

        parts.append(self.generate_footer())

        return '\n'.join(parts)
