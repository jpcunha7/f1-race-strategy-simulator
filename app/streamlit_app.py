"""Professional F1 Race Strategy Analysis Tool - Streamlit Dashboard.

Multi-page professional interface for race strategists with:
- Model validation and credibility analysis
- Strategy comparison with risk assessment
- Undercut/overcut window analysis
- Sensitivity analysis for robust decision-making

Author: João Pedro Cunha
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import logging

from f1strategy import (
    config,
    data_loader,
    degrade_model,
    simulator,
    optimizer,
    viz,
    validation,
    undercut,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="F1 Race Strategy Analysis",
    page_icon="",
    layout="wide",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #FF1E1E;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False


def sidebar_controls():
    """Professional sidebar controls."""
    st.sidebar.title("F1 Strategy Analyzer")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Race Configuration")
    year = st.sidebar.number_input("Year", 2018, 2025, 2024, help="Season year")
    event = st.sidebar.text_input("Event", "Bahrain", help="Grand Prix name")
    driver = st.sidebar.text_input("Driver Code", "VER", help="Three-letter driver code").upper()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Parameters")

    n_sims = st.sidebar.slider(
        "Monte Carlo Simulations",
        100, 5000, 1000, 100,
        help="More simulations = higher accuracy but slower"
    )

    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42, help="For reproducibility")

    auto_model = st.sidebar.checkbox(
        "Auto-select Degradation Model",
        value=True,
        help="Use cross-validation to select best model type"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Safety Car Settings")

    sc_prob = st.sidebar.slider("SC Probability", 0.0, 1.0, 0.3, 0.05)
    vsc_prob = st.sidebar.slider("VSC Probability", 0.0, 1.0, 0.2, 0.05)

    st.sidebar.markdown("---")

    run_button = st.sidebar.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        help="Execute race strategy analysis"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Author:** João Pedro Cunha")
    st.sidebar.markdown("**Data Source:** FastF1")

    return {
        'year': year,
        'event': event,
        'driver': driver,
        'n_sims': n_sims,
        'seed': seed,
        'auto_model': auto_model,
        'sc_prob': sc_prob,
        'vsc_prob': vsc_prob,
        'run_button': run_button,
    }


def run_full_analysis(params):
    """Execute complete race strategy analysis."""
    try:
        with st.spinner("Loading race data..."):
            cfg = config.StrategyConfig(
                n_simulations=params['n_sims'],
                random_seed=params['seed'],
                auto_select_degradation_model=params['auto_model'],
                safety_car_prob=params['sc_prob'],
                vsc_prob=params['vsc_prob'],
            )

            session = data_loader.load_race_session(params['year'], params['event'], cfg)
            race_info = data_loader.get_race_info(session)
            stint_data = data_loader.extract_stints(session, params['driver'])

        with st.spinner("Fitting and validating degradation models..."):
            deg_models = degrade_model.fit_all_compounds(stint_data, cfg)

            if not deg_models:
                st.error("No degradation models could be fitted from the data")
                return

            # Validate models
            try:
                validation_result = validation.validate_race(
                    params['year'],
                    params['event'],
                    params['driver'],
                    stint_data,
                    cfg,
                )
            except Exception as e:
                st.warning(f"Model validation skipped: {e}")
                validation_result = None

        with st.spinner("Optimizing race strategies..."):
            ranked_strategies, results_dict = optimizer.optimize_strategy(
                deg_models,
                race_info['total_laps'],
                cfg,
                max_strategies_to_test=30,
            )

            comparison_df = simulator.compare_strategies(results_dict)

            # Calculate risk profiles
            risk_profiles = optimizer.calculate_risk_profiles(results_dict, cfg)

        # Store in session state
        st.session_state.analysis_complete = True
        st.session_state.race_info = race_info
        st.session_state.deg_models = deg_models
        st.session_state.ranked_strategies = ranked_strategies
        st.session_state.results_dict = results_dict
        st.session_state.comparison_df = comparison_df
        st.session_state.cfg = cfg
        st.session_state.validation_result = validation_result
        st.session_state.risk_profiles = risk_profiles
        st.session_state.stint_data = stint_data

        st.success("Analysis complete")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)


def page_overview():
    """Page 1: Executive overview and recommendations."""
    st.title("F1 Race Strategy Analysis")

    if not st.session_state.analysis_complete:
        st.info("Configure race parameters in the sidebar and click 'Run Analysis' to begin")

        st.markdown("""
        ### Professional Race Strategy Tool

        This tool provides comprehensive strategy analysis for Formula 1 races:

        **Core Capabilities:**
        - Tire degradation modeling with automatic model selection
        - Monte Carlo simulation for uncertainty quantification
        - Strategy optimization with risk assessment
        - Undercut/overcut window analysis
        - Model validation against historical performance

        **Designed for:**
        - Race engineers and strategists
        - Performance analysts
        - Data scientists in motorsport

        **Technical Approach:**
        - Physics-based degradation modeling
        - Probabilistic scenario analysis
        - Statistical validation and credibility assessment
        """)
        return

    race_info = st.session_state.race_info
    risk_profiles = st.session_state.risk_profiles
    results_dict = st.session_state.results_dict

    # Race information header
    st.subheader(f"{race_info['event_name']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Location", f"{race_info['location']}, {race_info['country']}")
    with col2:
        st.metric("Total Laps", race_info['total_laps'])
    with col3:
        st.metric("Date", race_info['date'])
    with col4:
        st.metric("Simulations", st.session_state.cfg.n_simulations)

    st.markdown("---")

    # Executive summary
    st.subheader("Executive Summary")

    summary_text = optimizer.create_strategy_executive_summary(
        results_dict,
        risk_profiles,
        top_n=3,
        config=st.session_state.cfg,
    )

    st.text(summary_text)

    # Top 3 strategies visualization
    st.subheader("Top Strategy Risk Profiles")
    fig_risk = viz.plot_risk_profiles(risk_profiles, top_n=3, config=st.session_state.cfg)
    st.plotly_chart(fig_risk, use_container_width=True)


def page_validation():
    """Page 2: Model validation and credibility."""
    st.title("Model Validation & Credibility")

    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first (see Overview page)")
        return

    validation_result = st.session_state.validation_result

    if validation_result is None:
        st.warning("Model validation was not performed (insufficient data)")
        return

    st.markdown("""
    Model validation is critical for credibility. This analysis shows how well
    degradation models predict actual lap times on held-out test data.
    """)

    # Validation metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Overall MAE",
            f"{validation_result.overall_mae:.3f}s",
            help="Mean Absolute Error on test set"
        )

    with col2:
        st.metric(
            "Overall RMSE",
            f"{validation_result.overall_rmse:.3f}s",
            help="Root Mean Squared Error on test set"
        )

    # Validation plots
    st.subheader("Validation Diagnostics")
    fig_val = validation.create_validation_plots(
        validation_result,
        config=st.session_state.cfg
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # Detailed metrics table
    st.subheader("Validation Metrics by Compound")
    val_report = validation.generate_validation_report(validation_result)
    st.dataframe(val_report, use_container_width=True)


def page_degradation():
    """Page 3: Degradation model analysis."""
    st.title("Tire Degradation Models")

    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first (see Overview page)")
        return

    deg_models = st.session_state.deg_models

    st.markdown("""
    Tire degradation determines optimal pit strategy. Models are automatically
    selected via cross-validation (linear, quadratic, or piecewise).
    """)

    # Degradation curves
    fig_deg = viz.plot_degradation_curves(deg_models, st.session_state.cfg)
    st.plotly_chart(fig_deg, use_container_width=True)

    # Model details
    st.subheader("Model Parameters")

    for compound, model in deg_models.items():
        with st.expander(f"{compound} - {model.model_type.upper()} Model"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Baseline Lap Time", f"{model.baseline_laptime:.2f}s")
            with col2:
                st.metric("Degradation Rate", f"{model.deg_rate:.3f}s/lap")
            with col3:
                st.metric("R²", f"{model.r_squared:.3f}")
            with col4:
                st.metric("Samples", model.n_samples)

            if model.model_type == "piecewise":
                st.markdown("**Piecewise Parameters:**")
                st.write(f"- Breakpoint: Lap {model.coefficients.get('breakpoint', 'N/A')}")
                st.write(f"- Early rate: {model.coefficients.get('early_rate', 0):.3f}s/lap")
                st.write(f"- Late rate: {model.coefficients.get('late_rate', 0):.3f}s/lap")
            elif model.model_type == "quadratic":
                st.markdown("**Quadratic Parameter:**")
                st.write(f"- Quadratic coef: {model.coefficients.get('a', 0):.6f}s/lap²")


def page_strategy_comparison():
    """Page 4: Strategy comparison and selection."""
    st.title("Strategy Comparison")

    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first (see Overview page)")
        return

    results_dict = st.session_state.results_dict
    comparison_df = st.session_state.comparison_df
    risk_profiles = st.session_state.risk_profiles

    # Strategy comparison chart
    fig_comp = viz.plot_strategy_comparison(comparison_df, st.session_state.cfg)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed comparison table
    st.subheader("Detailed Performance Metrics")
    st.dataframe(comparison_df, use_container_width=True)

    # Distribution plots
    st.subheader("Race Time Distributions")

    tab1, tab2 = st.tabs(["Histogram", "Cumulative Distribution"])

    with tab1:
        fig_hist = viz.plot_race_time_distributions(results_dict, st.session_state.cfg)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        fig_cdf = viz.plot_cumulative_distribution(results_dict, st.session_state.cfg)
        st.plotly_chart(fig_cdf, use_container_width=True)


def page_undercut_analysis():
    """Page 5: Undercut/overcut window analysis."""
    st.title("Pit Window & Undercut Analysis")

    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first (see Overview page)")
        return

    deg_models = st.session_state.deg_models
    race_info = st.session_state.race_info

    st.markdown("""
    Undercut analysis identifies optimal pit timing to gain track position.
    Includes tire warmup penalties and degradation advantages.
    """)

    # Compound selection
    compounds = list(deg_models.keys())

    col1, col2, col3 = st.columns(3)

    with col1:
        your_compound = st.selectbox("Your Current Compound", compounds, index=0)
    with col2:
        new_compound = st.selectbox("New Compound", compounds, index=min(1, len(compounds)-1))
    with col3:
        opponent_compound = st.selectbox("Opponent Compound", compounds, index=0)

    # Stint ages
    col4, col5 = st.columns(2)
    with col4:
        your_age_start = st.number_input("Your Stint Age (start)", 1, 40, 10)
    with col5:
        opp_age_start = st.number_input("Opponent Stint Age (start)", 1, 40, 10)

    if st.button("Analyze Undercut Window"):
        with st.spinner("Analyzing undercut opportunities..."):
            try:
                fig_undercut = undercut.create_undercut_heatmap(
                    race_laps=race_info['total_laps'],
                    your_compound=your_compound,
                    your_stint_age_start=your_age_start,
                    new_compound=new_compound,
                    opponent_compound=opponent_compound,
                    opponent_stint_age_start=opp_age_start,
                    degradation_models=deg_models,
                    config=st.session_state.cfg,
                )

                st.plotly_chart(fig_undercut, use_container_width=True)

                # Find optimal window
                recommendation = undercut.find_optimal_undercut_window(
                    race_laps=race_info['total_laps'],
                    your_compound=your_compound,
                    your_stint_age_start=your_age_start,
                    new_compound=new_compound,
                    opponent_compound=opponent_compound,
                    opponent_stint_age_start=opp_age_start,
                    degradation_models=deg_models,
                    config=st.session_state.cfg,
                )

                st.success(f"**Recommended Pit Window:** Lap {recommendation.optimal_lap}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Gain", f"{recommendation.expected_gain:.2f}s")
                with col2:
                    st.metric("Window", f"Lap {recommendation.window_start}-{recommendation.window_end}")
                with col3:
                    st.metric("Risk Level", recommendation.risk_assessment)

            except Exception as e:
                st.error(f"Undercut analysis failed: {e}")


def page_scenario_analysis():
    """Page 6: Scenario and sensitivity analysis."""
    st.title("Scenario & Sensitivity Analysis")

    if not st.session_state.analysis_complete:
        st.warning("Please run analysis first (see Overview page)")
        return

    st.markdown("""
    Test how strategy rankings change under different scenarios (SC probability,
    pit loss variation, etc.) to assess decision robustness.
    """)

    st.info("Sensitivity analysis available in future updates")


def main():
    """Main application with page navigation."""
    params = sidebar_controls()

    if params['run_button']:
        run_full_analysis(params)

    # Page navigation
    pages = {
        "Overview": page_overview,
        "Degradation Models": page_degradation,
        "Model Validation": page_validation,
        "Strategy Comparison": page_strategy_comparison,
        "Pit Window Explorer": page_undercut_analysis,
        "Scenario Analysis": page_scenario_analysis,
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("", list(pages.keys()), label_visibility="collapsed")

    # Execute selected page
    pages[page]()


if __name__ == "__main__":
    main()
