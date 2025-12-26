"""HTML report generation for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import logging
from datetime import datetime
from pathlib import Path

from jinja2 import Template

from f1strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1strategy import viz

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>F1 Strategy Report - {{ race_info.event_name }}</title>
    <style>
        body { font-family: Arial; max-width: 1400px; margin: 0 auto; padding: 20px;
               background: #0f0f0f; color: #e0e0e0; }
        h1 { color: #ff1e1e; border-bottom: 3px solid #ff1e1e; }
        h2 { color: #1e90ff; margin-top: 30px; }
        .header { background: #1a1a1a; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
        .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
        .info-item { background: #1a1a1a; padding: 15px; border-radius: 8px; }
        .recommendation { background: #1a3a1a; padding: 20px; border-radius: 10px;
                         border-left: 5px solid #00ff00; margin: 20px 0; }
        .plot { margin: 30px 0; text-align: center; }
        .assumptions { background: #2d1a1a; padding: 15px; border-radius: 5px;
                      border-left: 4px solid #ff6b6b; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏎️ F1 Race Strategy Simulator</h1>
        <h2>{{ race_info.event_name }}</h2>
        <p><strong>Location:</strong> {{ race_info.location }}, {{ race_info.country }}</p>
        <p><strong>Date:</strong> {{ race_info.date }}</p>
        <p><strong>Total Laps:</strong> {{ race_info.total_laps }}</p>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
    </div>

    <div class="recommendation">
        <h3>💡 Recommended Strategy</h3>
        <p><strong>{{ best_strategy }}</strong></p>
        <p>Expected race time: {{ best_time }}</p>
    </div>

    <h2>Tire Degradation Models</h2>
    <div class="plot">{{ plot_degradation }}</div>

    <h2>Race Time Distributions</h2>
    <div class="plot">{{ plot_distributions }}</div>

    <h2>Cumulative Probability</h2>
    <div class="plot">{{ plot_cdf }}</div>

    <h2>Strategy Comparison</h2>
    <div class="plot">{{ plot_comparison }}</div>

    <div class="assumptions">
        <h3>⚠️ Modeling Assumptions</h3>
        <ul>
            <li>Tire degradation modeled from historical stint data</li>
            <li>Monte Carlo simulation with {{ n_sims }} iterations</li>
            <li>Safety car probability: {{ sc_prob }}%</li>
            <li>Pit loss: {{ pit_loss }} ± {{ pit_std }}s</li>
            <li>No traffic, incidents, or weather changes modeled</li>
            <li>Driver performance assumed consistent</li>
        </ul>
    </div>

    <div style="margin-top: 50px; text-align: center; color: #666;">
        <p><strong>F1 Race Strategy Simulator</strong> | Author: João Pedro Cunha</p>
        <p>Built with Claude Code</p>
    </div>
</body>
</html>
"""


def generate_report(
    degradation_models: dict,
    results_dict: dict,
    comparison_df,
    race_info: dict,
    best_strategy_name: str,
    config: StrategyConfig = DEFAULT_CONFIG,
    output_path: Path = None,
) -> str:
    """Generate HTML strategy report."""
    logger.info("Generating HTML report...")

    # Create plots
    plot_deg = viz.plot_degradation_curves(degradation_models, config).to_html(
        include_plotlyjs="cdn", div_id="deg_plot"
    )
    plot_dist = viz.plot_race_time_distributions(results_dict, config).to_html(
        include_plotlyjs=False, div_id="dist_plot"
    )
    plot_cdf = viz.plot_cumulative_distribution(results_dict, config).to_html(
        include_plotlyjs=False, div_id="cdf_plot"
    )
    plot_comp = viz.plot_strategy_comparison(comparison_df, config).to_html(
        include_plotlyjs=False, div_id="comp_plot"
    )

    # Find best time
    best_times = [r.total_time for r in results_dict[best_strategy_name]]
    best_time_str = f"{min(best_times)/60:.2f} - {max(best_times)/60:.2f} min"

    # Render template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        race_info=race_info,
        best_strategy=best_strategy_name,
        best_time=best_time_str,
        plot_degradation=plot_deg,
        plot_distributions=plot_dist,
        plot_cdf=plot_cdf,
        plot_comparison=plot_comp,
        n_sims=config.n_simulations,
        sc_prob=int(config.safety_car_prob * 100),
        pit_loss=config.pit_loss_mean,
        pit_std=config.pit_loss_std,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Report saved to: {output_path}")

    return html
