"""Command-line interface for F1 Race Strategy Simulator.

Enhanced professional CLI with:
- Validation command for model credibility assessment
- Undercut command for pit window analysis
- Full optimization with risk profiles

Author: João Pedro Cunha
"""

import argparse
import json
import logging
import sys
import uuid

from f1strategy import (
    config as cfg,
    data_loader,
    degrade_model,
    optimizer,
    report,
    simulator,
    validation,
    undercut,
)

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
            race_info["total_laps"],
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
            "race_info": race_info,
            "driver": args.driver,
            "best_strategy": ranked_strategies[0].description,
            "comparison": comparison_df.to_dict("records"),
        }

        json_path = output_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {json_path}")

        # Generate HTML report
        report_path = output_dir / "report.html"
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


def run_validation(args: argparse.Namespace) -> int:
    """Run model validation against historical race data."""
    try:
        logger.info(f"Validating models for {args.driver} at {args.year} {args.event}")

        # Create config
        config = cfg.StrategyConfig(random_seed=args.seed)

        # Load race data
        session = data_loader.load_race_session(args.year, args.event, config)
        stint_data = data_loader.extract_stints(session, args.driver)

        # Validate
        validation_result = validation.validate_race(
            args.year,
            args.event,
            args.driver,
            stint_data,
            config,
        )

        # Print results
        print(f"\n{'='*80}")
        print(f"MODEL VALIDATION: {args.driver} - {args.year} {args.event}")
        print(f"{'='*80}\n")

        print(f"Overall MAE:  {validation_result.overall_mae:.3f}s")
        print(f"Overall RMSE: {validation_result.overall_rmse:.3f}s\n")

        # Per-compound metrics
        val_report = validation.generate_validation_report(validation_result)
        print(val_report.to_string(index=False))

        # Save results
        run_id = args.run_id or str(uuid.uuid4())[:8]
        output_dir = config.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "validation_results.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "driver": args.driver,
                    "year": args.year,
                    "event": args.event,
                    "overall_mae": validation_result.overall_mae,
                    "overall_rmse": validation_result.overall_rmse,
                    "metrics": {
                        compound: {
                            "mae": metrics.mae,
                            "rmse": metrics.rmse,
                            "r_squared": metrics.r_squared,
                        }
                        for compound, metrics in validation_result.metrics_by_compound.items()
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"Validation results saved to: {output_file}")
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=args.verbose)
        return 1


def run_undercut_analysis(args: argparse.Namespace) -> int:
    """Run undercut window analysis."""
    try:
        logger.info(f"Analyzing undercut at {args.year} {args.event} lap {args.lap}")

        # Create config
        config = cfg.StrategyConfig(random_seed=args.seed)

        # Load race data
        session = data_loader.load_race_session(args.year, args.event, config)
        race_info = data_loader.get_race_info(session)
        stint_data = data_loader.extract_stints(session, args.driver)

        # Fit models
        deg_models = degrade_model.fit_all_compounds(stint_data, config)

        if not deg_models:
            logger.error("No degradation models could be fitted")
            return 1

        compounds = list(deg_models.keys())
        your_compound = compounds[0]
        new_compound = compounds[min(1, len(compounds) - 1)]
        opponent_compound = compounds[0]

        logger.info(f"Analyzing: {your_compound} -> {new_compound} vs {opponent_compound}")

        # Find optimal window
        recommendation = undercut.find_optimal_undercut_window(
            race_laps=race_info["total_laps"],
            your_compound=your_compound,
            your_stint_age_start=args.lap,
            new_compound=new_compound,
            opponent_compound=opponent_compound,
            opponent_stint_age_start=args.lap,
            degradation_models=deg_models,
            config=config,
        )

        # Print results
        print(f"\n{'='*80}")
        print(f"UNDERCUT ANALYSIS: {args.year} {args.event}")
        print(f"{'='*80}\n")

        print(f"Your Strategy:  {your_compound} -> {new_compound}")
        print(f"Opponent:       {opponent_compound}\n")

        print(f"Optimal Pit Lap:    {recommendation.optimal_lap}")
        print(f"Expected Gain:      {recommendation.expected_gain:+.2f}s")
        print(f"Pit Window:         Lap {recommendation.window_start}-{recommendation.window_end}")
        print(f"Risk Assessment:    {recommendation.risk_assessment.upper()}")

        return 0

    except Exception as e:
        logger.error(f"Undercut analysis failed: {e}", exc_info=args.verbose)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Simulator - Professional Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full strategy optimization
  f1strategy optimize --year 2024 --event "Bahrain" --driver "VER" --n-sims 2000

  # Model validation
  f1strategy validate --year 2024 --event "Bahrain" --driver "VER"

  # Undercut analysis
  f1strategy undercut --year 2024 --event "Bahrain" --driver "VER" --lap 15

Author: João Pedro Cunha
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize command (renamed from 'run')
    opt_parser = subparsers.add_parser("optimize", help="Run full strategy optimization")
    opt_parser.add_argument("--year", type=int, required=True, help="Season year")
    opt_parser.add_argument("--event", type=str, required=True, help="Grand Prix name")
    opt_parser.add_argument("--driver", type=str, required=True, help="Driver code")
    opt_parser.add_argument("--n-sims", type=int, default=1000, help="Number of simulations")
    opt_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    opt_parser.add_argument("--run-id", type=str, help="Custom run identifier")
    opt_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    opt_parser.add_argument("--quick", action="store_true", help="Quick mode: fewer simulations")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate degradation models")
    val_parser.add_argument("--year", type=int, required=True, help="Season year")
    val_parser.add_argument("--event", type=str, required=True, help="Grand Prix name")
    val_parser.add_argument("--driver", type=str, required=True, help="Driver code")
    val_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    val_parser.add_argument("--run-id", type=str, help="Custom run identifier")
    val_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Undercut command
    under_parser = subparsers.add_parser("undercut", help="Analyze undercut/pit windows")
    under_parser.add_argument("--year", type=int, required=True, help="Season year")
    under_parser.add_argument("--event", type=str, required=True, help="Grand Prix name")
    under_parser.add_argument("--driver", type=str, required=True, help="Driver code")
    under_parser.add_argument("--lap", type=int, required=True, help="Current lap for analysis")
    under_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    under_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Legacy 'run' command (alias for 'optimize')
    run_parser = subparsers.add_parser("run", help="Alias for 'optimize' (deprecated)")
    run_parser.add_argument("--year", type=int, required=True)
    run_parser.add_argument("--event", type=str, required=True)
    run_parser.add_argument("--driver", type=str, required=True)
    run_parser.add_argument("--n-sims", type=int, default=1000)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--run-id", type=str)
    run_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Quick mode adjustments
    if hasattr(args, "quick") and args.quick:
        args.n_sims = 500

    # Route to appropriate handler
    if args.command in ["run", "optimize"]:
        return run_strategy_analysis(args)
    elif args.command == "validate":
        return run_validation(args)
    elif args.command == "undercut":
        return run_undercut_analysis(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
