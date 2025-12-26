"""Command-line interface for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from f1strategy import config as cfg, data_loader, degrade_model, optimizer, report, simulator

logger = logging.getLogger(__name__)


def run_strategy_analysis(args: argparse.Namespace) -> int:
    """Run strategy analysis from CLI arguments."""
    try:
        # Create config
        config = cfg.StrategyConfig(
            n_simulations=args.n_sims,
            random_seed=args.seed,
        )

        # Load race data
        logger.info(f"Loading race data: {args.year} {args.event}")
        session = data_loader.load_race_session(args.year, args.event, config)
        race_info = data_loader.get_race_info(session)

        # Extract stints for driver
        logger.info(f"Extracting stints for {args.driver}")
        stint_data = data_loader.extract_stints(session, args.driver)

        # Fit degradation models
        logger.info("Fitting degradation models...")
        deg_models = degrade_model.fit_all_compounds(stint_data, config)

        if not deg_models:
            logger.error("No degradation models could be fitted")
            return 1

        # Optimize strategy
        logger.info("Optimizing strategies...")
        ranked_strategies, results_dict = optimizer.optimize_strategy(
            deg_models,
            race_info['total_laps'],
            config,
        )

        # Create comparison
        comparison_df = simulator.compare_strategies(results_dict)

        # Generate output directory
        run_id = args.run_id or str(uuid.uuid4())[:8]
        output_dir = config.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        summary = {
            'race_info': race_info,
            'driver': args.driver,
            'best_strategy': ranked_strategies[0].description,
            'comparison': comparison_df.to_dict('records'),
        }

        json_path = output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {json_path}")

        # Generate HTML report
        report_path = output_dir / 'report.html'
        report.generate_report(
            deg_models,
            results_dict,
            comparison_df,
            race_info,
            ranked_strategies[0].description,
            config,
            report_path,
        )

        logger.info(f"✅ Analysis complete! Output: {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=args.verbose)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  f1strategy run --year 2024 --event "Bahrain" --driver "VER" --n-sims 2000
  f1strategy run --year 2023 --event "Monaco" --driver "LEC" --n-sims 1000 --seed 123

Author: João Pedro Cunha
        """,
    )

    subparsers = parser.add_subparsers(dest='command')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run strategy analysis')
    run_parser.add_argument('--year', type=int, required=True)
    run_parser.add_argument('--event', type=str, required=True)
    run_parser.add_argument('--driver', type=str, required=True)
    run_parser.add_argument('--n-sims', type=int, default=1000)
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--run-id', type=str)
    run_parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.command == 'run':
        return run_strategy_analysis(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
