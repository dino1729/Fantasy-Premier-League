"""Report generation for FPL simulation results.

This module generates JSON audit trails and summary PDF reports
from simulation results.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from simulation.state import SimulationResult, GameweekState

def _is_plotting_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        import squarify  # noqa: F401
        return True
    except Exception:
        return False


class BacktestReportGenerator:
    """Generates comprehensive simulation reports.

    Outputs:
    1. JSON file with full audit trail
    2. Summary PDF report via LaTeX
    """

    def __init__(self, output_dir: Path = None):
        """Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir or 'simulation_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        simulation_result: SimulationResult,
        baseline_result: Optional[SimulationResult] = None,
        data_adapter: Optional[object] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Generate JSON data file and PDF summary.

        Args:
            simulation_result: Main simulation results
            baseline_result: Optional baseline for comparison

        Returns:
            Tuple of (json_path, pdf_path or None)
        """
        # Generate JSON
        json_path = self._generate_json(simulation_result, baseline_result)

        # Generate plots (optional)
        plot_paths: Dict[str, Path] = {}
        if data_adapter is not None and _is_plotting_available():
            try:
                plot_paths = self._generate_contribution_treemaps(simulation_result, data_adapter)
            except Exception as e:
                print(f"Treemap plot generation failed: {e}")

        # Generate PDF
        pdf_path = self._generate_pdf(simulation_result, baseline_result, plot_paths=plot_paths)

        return json_path, pdf_path

    @staticmethod
    def _latex_escape(text: str) -> str:
        """Escape common LaTeX special characters."""
        if text is None:
            return ""
        text = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    @staticmethod
    def _pick_snapshot_states(states: List[GameweekState]) -> Dict[str, Optional[GameweekState]]:
        """Pick start/mid/end snapshot states from a season run."""
        if not states:
            return {'start': None, 'mid': None, 'end': None}

        by_gw = {s.gameweek: s for s in states}
        start = by_gw.get(1) or states[0]

        max_gw = max(by_gw) if by_gw else 0
        mid_target = 19 if max_gw >= 19 else max(1, max_gw // 2)
        mid = by_gw.get(mid_target) or states[len(states) // 2]

        end = by_gw.get(max_gw) or states[-1]
        return {'start': start, 'mid': mid, 'end': end}

    @staticmethod
    def _squad_to_list(state: Optional[GameweekState]) -> List[Dict]:
        """Convert a state squad to a JSON-friendly list."""
        if not state:
            return []
        return [
            {
                'id': p.id,
                'name': p.name,
                'position': p.position,
                'team_id': p.team_id,
                'team_name': p.team_name,
                'purchase_price': p.purchase_price,
                'current_price': p.current_price,
            }
            for p in state.squad
        ]

    def _render_squad_table(self, state: Optional[GameweekState]) -> str:
        """Render a squad table for LaTeX."""
        if not state:
            return "No data available."

        chip = self._latex_escape(state.decisions.chip_used or "-")
        header = (
            f"\\noindent Gameweek: {state.gameweek} \\quad "
            f"Bank: {state.bank:.1f} \\quad "
            f"Free Transfers: {state.free_transfers} \\quad "
            f"Chip Used: {chip}\n"
        )

        player_by_id = {p.id: p for p in state.squad}

        lineup_ids = list(state.decisions.lineup or [])
        bench_ids = list(state.decisions.bench_order or [])

        # Fallback if lineup/bench not present for some reason
        if len(lineup_ids) != 11:
            lineup_ids = [p.id for p in state.squad[:11]]
        if len(bench_ids) != 4:
            bench_ids = [p.id for p in state.squad if p.id not in set(lineup_ids)][:4]

        def row_for(pid: int) -> Optional[str]:
            p = player_by_id.get(pid)
            if not p:
                return None
            return (
                f"{self._latex_escape(p.position)} & "
                f"{self._latex_escape(p.name)} & "
                f"{self._latex_escape(p.team_name)} & "
                f"{p.current_price:.1f} \\\\"
            )

        xi_rows = [row_for(pid) for pid in lineup_ids]
        xi_rows = [r for r in xi_rows if r]
        bench_rows = [row_for(pid) for pid in bench_ids]
        bench_rows = [r for r in bench_rows if r]

        rows_tex = (
            "\\multicolumn{4}{l}{\\textbf{Starting XI}} \\\\\n"
            "\\midrule\n"
            + "\n".join(xi_rows)
            + "\n\\midrule\n"
            "\\multicolumn{4}{l}{\\textbf{Bench}} \\\\\n"
            "\\midrule\n"
            + "\n".join(bench_rows)
        )

        return (
            header
            + """
\\begingroup
\\footnotesize
\\setlength{\\tabcolsep}{4pt}
\\renewcommand{\\arraystretch}{1.05}
\\begin{longtable}{p{0.10\\textwidth}p{0.46\\textwidth}p{0.30\\textwidth}r}
\\toprule
Pos & Player & Team & Price \\\\
\\midrule
\\endhead
"""
            + rows_tex
            + """
\\bottomrule
\\end{longtable}
\\endgroup
"""
        )

    def _generate_json(
        self,
        result: SimulationResult,
        baseline: Optional[SimulationResult] = None,
    ) -> Path:
        """Generate JSON audit trail.

        Args:
            result: Simulation result
            baseline: Optional baseline result

        Returns:
            Path to JSON file
        """
        data = {
            'metadata': {
                'season': result.season,
                'simulation_date': datetime.now().isoformat(),
                'gameweeks_simulated': len(result.states),
            },
            'summary': {
                'total_points': result.total_points,
                'total_hits': result.total_hits,
                'transfers_made': result.transfers_made,
                'chips_used': result.chips_used,
                'average_gw_points': result.get_average_gw_points(),
                'best_gw': result.get_best_gw(),
                'worst_gw': result.get_worst_gw(),
            },
            'comparison': None,
            'squad_snapshots': {},
            'gameweeks': [],
        }

        snapshots = self._pick_snapshot_states(result.states)
        for key, state in snapshots.items():
            if not state:
                continue
            data['squad_snapshots'][key] = {
                'gameweek': state.gameweek,
                'bank': state.bank,
                'free_transfers': state.free_transfers,
                'chip_used': state.decisions.chip_used,
                'lineup': list(state.decisions.lineup or []),
                'bench_order': list(state.decisions.bench_order or []),
                'squad': self._squad_to_list(state),
            }

        if baseline:
            data['comparison'] = {
                'baseline_points': baseline.total_points,
                'improvement': result.total_points - baseline.total_points,
                'improvement_percentage': (
                    (result.total_points - baseline.total_points) / baseline.total_points * 100
                    if baseline.total_points > 0 else 0
                ),
            }

        # Add gameweek details
        for state in result.states:
            gw_data = self._state_to_dict(state)
            data['gameweeks'].append(gw_data)

        json_path = self.output_dir / 'simulation_results.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        return json_path

    def _state_to_dict(self, state: GameweekState) -> Dict:
        """Convert GameweekState to dict for JSON."""
        return {
            'gameweek': state.gameweek,
            'points': state.results.gw_points,
            'points_before_hits': state.results.gw_points_before_hits,
            'hit_cost': state.results.hit_cost,
            'total_cumulative': state.total_points,
            'bank': state.bank,
            'free_transfers': state.free_transfers,
            'decisions': {
                'transfers_count': len(state.decisions.transfers),
                'transfers': [
                    {
                        'out': t.player_out_name,
                        'in': t.player_in_name,
                        'price_diff': t.price_in - t.price_out,
                        'is_hit': t.is_hit,
                    }
                    for t in state.decisions.transfers
                ],
                'captain': state.decisions.captain_id,
                'vice_captain': state.decisions.vice_captain_id,
                'chip_used': state.decisions.chip_used,
                'formation': state.decisions.formation,
            },
            'results': {
                'auto_subs_count': len(state.results.auto_subs),
                'auto_subs': [
                    {
                        'out': s.player_out_name,
                        'in': s.player_in_name,
                    }
                    for s in state.results.auto_subs
                ],
                'effective_captain_id': state.results.effective_captain_id,
                'captain_points': state.results.captain_points,
                'bench_points': state.results.bench_points,
            },
        }

    def _generate_pdf(
        self,
        result: SimulationResult,
        baseline: Optional[SimulationResult] = None,
        plot_paths: Optional[Dict[str, Path]] = None,
    ) -> Optional[Path]:
        """Generate PDF summary report via LaTeX.

        Args:
            result: Simulation result
            baseline: Optional baseline result

        Returns:
            Path to PDF file or None if generation failed
        """
        # Generate LaTeX content
        tex_content = self._generate_latex(result, baseline, plot_paths=plot_paths or {})

        tex_path = self.output_dir / 'simulation_report.tex'
        with open(tex_path, 'w') as f:
            f.write(tex_content)

        # Compile to PDF
        try:
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'simulation_report.tex'],
                cwd=self.output_dir,
                capture_output=True,
                check=True,
            )
            # Run twice for references
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'simulation_report.tex'],
                cwd=self.output_dir,
                capture_output=True,
                check=True,
            )
            return self.output_dir / 'simulation_report.pdf'
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"PDF generation failed (pdflatex required): {e}")
            return None

    def _generate_latex(
        self,
        result: SimulationResult,
        baseline: Optional[SimulationResult] = None,
        plot_paths: Optional[Dict[str, Path]] = None,
    ) -> str:
        """Generate LaTeX document content.

        Args:
            result: Simulation result
            baseline: Optional baseline result

        Returns:
            LaTeX document string
        """
        improvement = ""
        if baseline:
            diff = result.total_points - baseline.total_points
            sign = "+" if diff >= 0 else ""
            improvement = f"""
\\subsection*{{Comparison vs No-Transfer Baseline}}
\\begin{{tabular}}{{ll}}
Baseline Points & {baseline.total_points} \\\\
Simulation Points & {result.total_points} \\\\
Improvement & {sign}{diff} ({sign}{diff/baseline.total_points*100:.1f}\\%) \\\\
\\end{{tabular}}
"""

        plot_paths = plot_paths or {}

        # Build GW summary table
        gw_rows = []
        for state in result.states:
            chip = self._latex_escape(state.decisions.chip_used or "-")
            transfers = len(state.decisions.transfers)
            hit = f"-{state.results.hit_cost}" if state.results.hit_cost > 0 else "-"
            gw_rows.append(
                f"{state.gameweek} & {state.results.gw_points} & "
                f"{state.total_points} & {transfers} & {hit} & {chip} \\\\"
            )

        gw_table = "\n".join(gw_rows)

        # Chips used summary
        chips_summary = ""
        if result.chips_used:
            chips_list = ", ".join(
                f"{self._latex_escape(chip)}: GW{gw}" for chip, gw in result.chips_used.items()
            )
            chips_summary = f"Chips Used: {chips_list}"
        chips_block = f"\n\\medskip\n\\noindent {chips_summary}\n" if chips_summary else ""

        snapshots = self._pick_snapshot_states(result.states)
        squad_snapshots_section = f"""
\\section*{{Squad Snapshots}}
\\subsection*{{Starting Squad}}
{self._render_squad_table(snapshots.get('start'))}
\\subsection*{{Mid-Season Squad}}
{self._render_squad_table(snapshots.get('mid'))}
\\subsection*{{End-of-Season Squad}}
{self._render_squad_table(snapshots.get('end'))}
"""

        plots_section = ""
        if plot_paths:
            begin_img = plot_paths.get('begin')
            mid_img = plot_paths.get('mid')
            end_img = plot_paths.get('end')

            def include(p: Optional[Path]) -> str:
                if not p:
                    return ""
                try:
                    rel = p.relative_to(self.output_dir).as_posix()
                except Exception:
                    rel = p.as_posix()
                return f"\\includegraphics[width=\\textwidth]{{\\detokenize{{{rel}}}}}"

            plots_section = f"""
\\section*{{Player Point Contribution (Treemaps)}}
\\subsection*{{Beginning (GW1--GW13)}}
{include(begin_img)}
\\subsection*{{Mid-Season (GW14--GW26)}}
{include(mid_img)}
\\subsection*{{End-of-Season (GW27--GW38)}}
{include(end_img)}
"""

        tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{graphicx}}
\\sloppy
\\emergencystretch=2em

\\title{{FPL Season Simulation Report}}
\\author{{Generated by FPL Backtester}}
\\date{{{datetime.now().strftime('%Y-%m-%d %H:%M')}}}

\\begin{{document}}
\\maketitle

\\section*{{Season Summary}}
\\begin{{tabular}}{{l>{{\\raggedright\\arraybackslash}}p{{0.72\\textwidth}}}}
\\toprule
Season & {result.season} \\\\
Gameweeks Simulated & {len(result.states)} \\\\
\\midrule
Total Points & \\textbf{{{result.total_points}}} \\\\
Total Hits & {result.total_hits} \\\\
Transfers Made & {result.transfers_made} \\\\
Average GW Points & {result.get_average_gw_points():.1f} \\\\
Best GW & GW{result.get_best_gw()[0]} ({result.get_best_gw()[1]} pts) \\\\
Worst GW & GW{result.get_worst_gw()[0]} ({result.get_worst_gw()[1]} pts) \\\\
\\bottomrule
\\end{{tabular}}
{chips_block}

{improvement}

{squad_snapshots_section}

{plots_section}

\\section*{{Gameweek Details}}
\\begin{{longtable}}{{cccccc}}
\\toprule
GW & Points & Cumulative & Transfers & Hits & Chip \\\\
\\midrule
\\endhead
{gw_table}
\\bottomrule
\\end{{longtable}}

\\section*{{Transfer Activity}}
\\begingroup
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{longtable}}{{p{{0.10\\textwidth}}p{{0.86\\textwidth}}}}
\\toprule
GW & Transfers \\\\
\\midrule
\\endhead
"""

        # Add transfer details
        for state in result.states:
            if state.decisions.transfers:
                transfers_str = "; \\allowbreak ".join(
                    f"{self._latex_escape(t.player_out_name)} $\\rightarrow$ {self._latex_escape(t.player_in_name)}"
                    for t in state.decisions.transfers
                )
                tex += f"GW{state.gameweek} & {transfers_str} \\\\\n"

        tex += """\\bottomrule
\\end{longtable}
\\endgroup

\\end{document}
"""
        return tex

    def _compute_period_contributions(
        self,
        result: SimulationResult,
        data_adapter,
        start_gw: int,
        end_gw: int,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, str]]]:
        """Compute per-player points contribution to the team over a GW range."""
        states_by_gw = {s.gameweek: s for s in result.states}
        contrib: Dict[int, int] = {}
        meta: Dict[int, Dict[str, str]] = {}

        for gw in range(start_gw, end_gw + 1):
            state = states_by_gw.get(gw)
            if not state:
                continue

            # Track metadata for any squad players seen this GW
            for p in state.squad:
                if p.id not in meta:
                    meta[p.id] = {'name': p.name, 'position': p.position}

            lineup_ids = list(state.decisions.lineup or [])
            bench_ids = list(state.decisions.bench_order or [])

            final_xi = list(lineup_ids)
            sub_in_ids = []
            for s in state.results.auto_subs:
                if s.player_out_id in final_xi:
                    idx = final_xi.index(s.player_out_id)
                    final_xi[idx] = s.player_in_id
                    sub_in_ids.append(s.player_in_id)

            effective_captain_id = int(state.results.effective_captain_id or 0)
            captain_multiplier = 3 if state.decisions.chip_used == 'triple_captain' else 2

            # XI points (with captain multiplier if applicable)
            for pid in final_xi:
                pts = int(data_adapter.get_player_actual_points(pid, gw) or 0)
                if effective_captain_id and pid == effective_captain_id:
                    pts *= captain_multiplier
                contrib[pid] = contrib.get(pid, 0) + pts

            # Bench boost: remaining bench points count
            if state.decisions.chip_used == 'bench_boost':
                sub_in = set(sub_in_ids)
                for pid in bench_ids:
                    if pid in sub_in:
                        continue
                    pts = int(data_adapter.get_player_actual_points(pid, gw) or 0)
                    contrib[pid] = contrib.get(pid, 0) + pts

        return contrib, meta

    def _generate_contribution_treemaps(
        self,
        result: SimulationResult,
        data_adapter,
        top_n: int = 30,
    ) -> Dict[str, Path]:
        """Generate treemap PNGs for beginning/mid/end season contribution."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import squarify

        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        periods = {
            'begin': (1, 13),
            'mid': (14, 26),
            'end': (27, 38),
        }

        pos_colors = {
            'GKP': '#f4c2c2',
            'DEF': '#b0d0ff',
            'MID': '#b7f0c0',
            'FWD': '#f6c48c',
        }

        out: Dict[str, Path] = {}

        for key, (start_gw, end_gw) in periods.items():
            contrib, meta = self._compute_period_contributions(result, data_adapter, start_gw, end_gw)
            items = [(pid, pts) for pid, pts in contrib.items() if pts > 0]
            items.sort(key=lambda x: x[1], reverse=True)

            if not items:
                continue

            top = items[:top_n]
            rest = items[top_n:]
            other_points = sum(p for _, p in rest)

            labels = []
            sizes = []
            colors = []

            for pid, pts in top:
                info = meta.get(pid, {'name': f'Player {pid}', 'position': ''})
                name = info.get('name', f'Player {pid}')
                pos = info.get('position', '')
                labels.append(f"{name}\\n({pts})")
                sizes.append(pts)
                colors.append(pos_colors.get(pos, '#dddddd'))

            if other_points > 0:
                labels.append(f"Other\\n({other_points})")
                sizes.append(other_points)
                colors.append('#dddddd')

            fig, ax = plt.subplots(figsize=(16, 9))
            squarify.plot(
                sizes=sizes,
                label=labels,
                color=colors,
                ax=ax,
                pad=True,
                text_kwargs={'fontsize': 11},
            )
            ax.axis('off')
            title = {
                'begin': 'Points per Player Distribution (GW1–GW13)',
                'mid': 'Points per Player Distribution (GW14–GW26)',
                'end': 'Points per Player Distribution (GW27–GW38)',
            }.get(key, f'Points per Player Distribution (GW{start_gw}–GW{end_gw})')
            ax.set_title(title, fontsize=16, pad=12)

            legend_patches = [
                Patch(facecolor=pos_colors['GKP'], label='GKP'),
                Patch(facecolor=pos_colors['DEF'], label='DEF'),
                Patch(facecolor=pos_colors['MID'], label='MID'),
                Patch(facecolor=pos_colors['FWD'], label='FWD'),
            ]
            ax.legend(handles=legend_patches, title='Position', loc='upper right')

            fname = f'points_treemap_{key}.png'
            path = plots_dir / fname
            fig.tight_layout(pad=1.2)
            fig.savefig(path, dpi=200)
            plt.close(fig)

            out[key] = path

        return out

    def generate_summary_text(
        self,
        result: SimulationResult,
        baseline: Optional[SimulationResult] = None,
    ) -> str:
        """Generate plain text summary.

        Args:
            result: Simulation result
            baseline: Optional baseline

        Returns:
            Text summary string
        """
        lines = [
            "=" * 60,
            "FPL SEASON SIMULATION SUMMARY",
            "=" * 60,
            f"Season: {result.season}",
            f"Gameweeks: {len(result.states)}",
            "",
            f"TOTAL POINTS: {result.total_points}",
            f"Total Hits: {result.total_hits}",
            f"Transfers Made: {result.transfers_made}",
            f"Avg GW Points: {result.get_average_gw_points():.1f}",
            f"Best GW: GW{result.get_best_gw()[0]} ({result.get_best_gw()[1]} pts)",
            f"Worst GW: GW{result.get_worst_gw()[0]} ({result.get_worst_gw()[1]} pts)",
        ]

        if result.chips_used:
            lines.append("")
            lines.append("Chips Used:")
            for chip, gw in result.chips_used.items():
                lines.append(f"  - {chip}: GW{gw}")

        if baseline:
            diff = result.total_points - baseline.total_points
            pct = diff / baseline.total_points * 100 if baseline.total_points > 0 else 0
            lines.extend([
                "",
                "-" * 60,
                "COMPARISON VS NO-TRANSFER BASELINE",
                "-" * 60,
                f"Baseline Points: {baseline.total_points}",
                f"Simulation Points: {result.total_points}",
                f"Improvement: {'+' if diff >= 0 else ''}{diff} ({'+' if pct >= 0 else ''}{pct:.1f}%)",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
