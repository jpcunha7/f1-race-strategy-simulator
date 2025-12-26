"""Streamlit dashboard for F1 Race Strategy Simulator.

Author: João Pedro Cunha
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import logging

from f1strategy import config, data_loader, degrade_model, simulator, optimizer, viz

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="F1 Race Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False


def sidebar_inputs():
    """Render sidebar controls."""
    st.sidebar.title("🏎️ F1 Strategy Simulator")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Race Selection")
    year = st.sidebar.number_input("Year", 2018, 2025, 2024)
    event = st.sidebar.text_input("Event", "Bahrain")
    driver = st.sidebar.text_input("Driver Code", "VER").upper()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")

    n_sims = st.sidebar.slider("Simulations", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    return {
        'year': year,
        'event': event,
        'driver': driver,
        'n_sims': n_sims,
        'seed': seed,
        'run_button': run_button,
    }


def run_analysis(params):
    """Run the strategy analysis."""
    try:
        with st.spinner("Loading race data..."):
            cfg = config.StrategyConfig(
                n_simulations=params['n_sims'],
                random_seed=params['seed'],
            )

            session = data_loader.load_race_session(params['year'], params['event'], cfg)
            race_info = data_loader.get_race_info(session)

            stint_data = data_loader.extract_stints(session, params['driver'])

        with st.spinner("Fitting degradation models..."):
            deg_models = degrade_model.fit_all_compounds(stint_data, cfg)

            if not deg_models:
                st.error("No degradation models could be fitted from the data")
                return

        with st.spinner("Optimizing strategies..."):
            ranked_strategies, results_dict = optimizer.optimize_strategy(
                deg_models,
                race_info['total_laps'],
                cfg,
                max_strategies_to_test=30,
            )

            comparison_df = simulator.compare_strategies(results_dict)

        # Store in session state
        st.session_state.analysis_complete = True
        st.session_state.race_info = race_info
        st.session_state.deg_models = deg_models
        st.session_state.ranked_strategies = ranked_strategies
        st.session_state.results_dict = results_dict
        st.session_state.comparison_df = comparison_df
        st.session_state.cfg = cfg

        st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)


def main():
    """Main application."""
    params = sidebar_inputs()

    if params['run_button']:
        run_analysis(params)

    st.title("🏎️ F1 Race Strategy Simulator")

    if not st.session_state.analysis_complete:
        st.info("👈 Configure race parameters and click 'Run Analysis' to begin")

        st.markdown("""
        ### About This Tool

        This simulator helps analyze F1 race strategies using:
        - **Tire Degradation Models**: Learned from historical stint data
        - **Monte Carlo Simulation**: Accounts for uncertainty in degradation, pit stops, safety cars
        - **Strategy Optimization**: Finds optimal pit windows and compound choices
        - **Physics-Based**: Models grip loss leading to lap time increase

        **Data Source**: FastF1 (free, open-source F1 data)
        """)
        return

    # Display results
    race_info = st.session_state.race_info
    deg_models = st.session_state.deg_models
    ranked_strategies = st.session_state.ranked_strategies
    results_dict = st.session_state.results_dict
    comparison_df = st.session_state.comparison_df

    # Race info
    st.subheader(f"📍 {race_info['event_name']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Location", f"{race_info['location']}, {race_info['country']}")
    with col2:
        st.metric("Total Laps", race_info['total_laps'])
    with col3:
        st.metric("Date", race_info['date'])

    # Best strategy recommendation
    best_strategy = ranked_strategies[0]
    st.success(f"**💡 Recommended Strategy:** {best_strategy.description}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Strategy Comparison",
        "Degradation Models",
        "Distributions",
        "Data & Export"
    ])

    with tab1:
        st.subheader("Strategy Comparison")
        fig_comp = viz.plot_strategy_comparison(comparison_df, st.session_state.cfg)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.dataframe(comparison_df, use_container_width=True)

    with tab2:
        st.subheader("Tire Degradation Curves")
        fig_deg = viz.plot_degradation_curves(deg_models, st.session_state.cfg)
        st.plotly_chart(fig_deg, use_container_width=True)

        for compound, model in deg_models.items():
            with st.expander(f"{compound} - Model Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Baseline Lap Time", f"{model.baseline_laptime:.2f}s")
                with col2:
                    st.metric("Degradation Rate", f"{model.deg_rate:.3f}s/lap")
                with col3:
                    st.metric("R²", f"{model.r_squared:.3f}")

    with tab3:
        st.subheader("Race Time Distributions")

        fig_hist = viz.plot_race_time_distributions(results_dict, st.session_state.cfg)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Cumulative Distribution")
        fig_cdf = viz.plot_cumulative_distribution(results_dict, st.session_state.cfg)
        st.plotly_chart(fig_cdf, use_container_width=True)

    with tab4:
        st.subheader("Simulation Data")
        st.markdown(f"**Simulations:** {st.session_state.cfg.n_simulations}")
        st.markdown(f"**Random Seed:** {st.session_state.cfg.random_seed}")
        st.markdown(f"**Strategies Tested:** {len(results_dict)}")

        if st.button("Generate HTML Report"):
            from f1strategy import report
            html = report.generate_report(
                deg_models,
                results_dict,
                comparison_df,
                race_info,
                best_strategy.description,
                st.session_state.cfg,
            )

            st.download_button(
                "Download Report",
                html,
                file_name=f"strategy_report_{race_info['event_name']}.html",
                mime="text/html",
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Author:** João Pedro Cunha")
    st.sidebar.markdown("**Data:** FastF1")


if __name__ == "__main__":
    main()
