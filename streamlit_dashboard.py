"""
Streamlit Dashboard for Phase 3 AI Model Evaluation and Signal Monitoring.
Hosts all Phase 3 components in a single interactive web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import asyncio

# Import Phase 3 components
from ai.evaluation.dashboard import ModelEvaluationDashboard, SignalQualityMonitor
from ai.inference.predictor import AIPredictor
from ai.training.trainer import ModelTrainer
from ai.datasets.horizon_configs import HORIZON_CONFIGS

# Import Phase 4 components
from trading.agents.multi_agent_system import multi_agent_system, MultiAgentSignal
from trading.execution.execution_engine import BinaryTradingEngine

# Page configuration
st.set_page_config(
    page_title="BATS Phase 3: AI Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = None

if 'signal_monitor' not in st.session_state:
    st.session_state.signal_monitor = None

if 'ai_predictor' not in st.session_state:
    st.session_state.ai_predictor = None

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def init_components():
    """Initialize dashboard components if not already done."""
    if st.session_state.dashboard is None:
        st.session_state.dashboard = ModelEvaluationDashboard()
    
    if st.session_state.signal_monitor is None:
        st.session_state.signal_monitor = SignalQualityMonitor(min_confidence=0.65)
    
    if st.session_state.ai_predictor is None:
        st.session_state.ai_predictor = AIPredictor()
    
    # Load models if available
    model_dir = Path("models")
    if model_dir.exists():
        horizons = list(HORIZON_CONFIGS.keys())
        st.session_state.ai_predictor.load_models(horizons)
        if any(horizon in st.session_state.ai_predictor.models for horizon in horizons):
            st.session_state.model_trained = True

def load_sample_data():
    """Load sample historical data for demonstration."""
    loader = st.session_state.get('historical_loader')
    if loader is None:
        from ai.training.data_preparation import HistoricalDataLoader
        loader = HistoricalDataLoader()
    
    # Try loading sample data
    data_dir = Path("data/historical")
    csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
    
    if csv_files:
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            # Generate sample timestamps if missing
            df['timestamp'] = pd.date_range(
                start='2026-01-01', 
                periods=len(df), 
                freq='1min'
            )
        return df
    
    # Generate synthetic data if no CSV available
    from ai.training.data_preparation import HistoricalDataLoader
    loader = HistoricalDataLoader()
    df = loader.generate_synthetic_data(n_samples=5000)
    return df

def main():
    init_components()
    
    st.title("📈 BATS Phase 3: AI Prediction Engine Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Model Evaluation", "🎯 Signal Monitor", "⚙️ Parameter Optimization", 
         "🔮 Live Prediction", "🤖 Multi-Agent System", "🛡️ Risk Management"]
    )
    
    # Load data on sidebar
    with st.sidebar.expander("📁 Data Management"):
        st.caption("Loading historical data...")
        df = load_sample_data()
        st.info(f"Loaded {len(df)} data points")
        
        if st.button("Refresh Data"):
            st.rerun()
    
    # Main content based on selection
    if page == "📊 Model Evaluation":
        render_model_evaluation(df)
    
    elif page == "🎯 Signal Monitor":
        render_signal_monitor()
    
    elif page == "⚙️ Parameter Optimization":
        render_parameter_optimization()
    
    elif page == "🔮 Live Prediction":
        render_live_prediction(df)
    
    elif page == "🤖 Multi-Agent System":
        render_multi_agent_system(df)
    
    elif page == "🛡️ Risk Management":
        render_risk_management()

def render_model_evaluation(df):
    """Render the model evaluation page."""
    st.header("Model Evaluation Dashboard")
    
    # Initialize dashboard if needed
    if st.session_state.dashboard is None:
        st.session_state.dashboard = ModelEvaluationDashboard()
    
    # Train models button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment"):
                try:
                    trainer = ModelTrainer()
                    results = trainer.train_all_horizons(
                        df, 
                        model_types=["xgboost", "lstm"]
                    )
                    
                    # Add to dashboard
                    st.session_state.dashboard.add_results(results)
                    
                    # Display summary
                    summary = st.session_state.dashboard.generate_summary_table()
                    st.success("Training complete!")
                    
                    if not summary.empty:
                        st.dataframe(summary, use_container_width=True)
                    
                    # Generate charts
                    fig = st.session_state.dashboard.generate_performance_chart()
                    if 'error' not in fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show calibration
                    cal = st.session_state.dashboard.generate_confidence_calibration_chart()
                    if 'error' not in cal:
                        st.plotly_chart(cal, use_container_width=True)
                    
                    # Trading metrics
                    trad = st.session_state.dashboard.generate_trading_metrics_chart()
                    if 'error' not in trad:
                        st.plotly_chart(trad, use_container_width=True)
                    
                    # Print summary
                    st.session_state.dashboard.print_summary()
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Display existing results
    if st.session_state.dashboard.training_results:
        st.subheader("Model Performance Summary")
        
        # Use cached results
        summary = st.session_state.dashboard.generate_summary_table()
        
        if not summary.empty:
            st.dataframe(
                summary.style.highlight_max(color='#3a7dff', subset=['Accuracy', 'ROC-AUC']),
                use_container_width=True
            )
        else:
            st.info("No model training results available. Click 'Train Models' above.")
    else:
        st.info("No model training results available. Click 'Train Models' above.")

def render_signal_monitor():
    """Render the signal quality monitoring page."""
    st.header("Signal Quality Monitor")
    
    # Initialize signal monitor
    if st.session_state.signal_monitor is None:
        st.session_state.signal_monitor = SignalQualityMonitor(min_confidence=0.65)
    
    monitor = st.session_state.signal_monitor
    
    # Quality metrics overview
    st.subheader("Signal Quality Metrics")
    metrics = monitor.get_quality_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", metrics.get('total_signals', 0))
    with col2:
        st.metric("Approval Rate", f"{metrics.get('approval_rate', 0):.1f}%")
    with col3:
        st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.2f}")
    with col4:
        st.metric("UP Signals", f"{metrics.get('up_signals', 0)} ({metrics.get('up_percentage', 0):.1f}%)")
    
    # Recent signals
    st.subheader("Recent Signal History")
    recent_signals = monitor.signal_history[-10:] if monitor.signal_history else []
    
    if recent_signals:
        recent_df = pd.DataFrame(recent_signals)[
            ['timestamp', 'direction', 'confidence', 'validation_status']
        ]
        st.dataframe(recent_df.tail(10), use_container_width=True)
    else:
        st.info("No signal history yet. Start generating signals.")
    
    # Quality report button
    if st.button("Generate Quality Report"):
        report = monitor.generate_quality_report()
        st.json(report)
    
    # Validation test
    st.subheader("Test Signal Validation")
    with st.form(key="signal_validation_form"):
        col1, col2 = st.columns(2)
        with col1:
            test_direction = st.selectbox("Direction", ["UP", "DOWN", "NEUTRAL"])
            test_confidence = st.slider("Confidence", 0.0, 1.0, 0.75)
        with col2:
            test_horizon = st.selectbox("Horizon", ["30s", "60s", "120s", "300s"])
            test_signal = {
                'direction': test_direction,
                'confidence': test_confidence,
                'horizon': test_horizon
            }
        
        submit = st.form_submit_button("Validate Signal")
        
        if submit:
            is_valid, reason, enriched = monitor.validate_signal(test_signal)
            if is_valid:
                st.success(f"✅ Signal VALIDATED: {reason}")
            else:
                st.warning(f"❌ Signal REJECTED: {reason}")
    
    # Export data
    if st.button("Export Signal Data"):
        if monitor.signal_history:
            export_df = pd.DataFrame(monitor.signal_history)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="signal_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No signal history to export.")

def render_parameter_optimization():
    """Render the parameter optimization page."""
    st.header("Parameter Optimization")
    
    st.caption("Run hyperparameter optimization for selected model and horizon")
    
    # Model and horizon selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "logistic"],
            help="Select the model type to optimize"
        )
    
    with col2:
        horizon = st.selectbox(
            "Horizon",
            list(HORIZON_CONFIGS.keys()),
            help="Select prediction horizon"
        )
    
    # Optimization parameters
    col1, col2 = st.columns(2)
    with col1:
        n_iterations = st.slider(
            "Iterations",
            min_value=10,
            max_value=100,
            value=30,
            help="Number of random search iterations"
        )
    
    with col2:
        test_size = st.slider(
            "Test Size %",
            min_value=10,
            max_value=30,
            value=20,
            help="Percentage of data for testing"
        )
    
    # Run optimization
    if st.button("🔧 Run Optimization", type="primary"):
        with st.spinner("Running optimization search..."):
            try:
                # Prepare data
                from ai.training.data_preparation import HistoricalDataLoader
                loader = HistoricalDataLoader()
                sample_df = loader.generate_synthetic_data(n_samples=5000)
                
                # Prepare training data
                from ai.training.data_preparation import DataPreparationPipeline
                pipeline = DataPreparationPipeline(loader)
                data = pipeline.prepare_training_data(sample_df, horizon)
                
                # Run optimization
                from ai.parameter_optimization.optimize import Optimizer
                
                # Get parameter space
                if model_type == "xgboost":
                    param_space = {
                        "learning_rate": np.linspace(0.01, 0.3, 30).tolist(),
                        "max_depth": list(range(3, 10)),
                        "subsample": np.linspace(0.5, 1.0, 12).tolist(),
                        "colsample_bytree": np.linspace(0.5, 1.0, 12).tolist(),
                        "scale_pos_weight": [0.5, 1, 2, 3, 5],
                        "n_estimators": list(range(50, 300, 25))
                    }
                else:
                    param_space = {
                        "C": np.logspace(-4, 2, 30).tolist(),
                        "penalty": ["l2", "elasticnet"],
                        "solver": ["lbfgs", "saga"],
                        "max_iter": [100, 200, 400]
                    }
                
                # Run optimization
                optimizer = Optimizer(param_space, n_iterations=n_iterations)
                best_params = optimizer.search(
                    data['X_train'], 
                    data['y_train']
                )
                
                # Evaluate best params
                best_model = XGBoostModel if model_type == "xgboost" else LogisticRegressionModel
                model = best_model({})
                metrics = model.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
                
                # Display results
                st.success("Optimization complete!")
                
                # Show best parameters
                st.subheader("Best Parameters Found")
                st.json(best_params)
                
                # Show metrics
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics.accuracy:.4f}")
                with col2:
                    st.metric("ROC-AUC", f"{metrics.roc_auc:.4f}")
                with col3:
                    st.metric("Win Rate", f"{metrics.win_rate:.4f}")
                
                # Store results
                if 'optimization_results' not in st.session_state:
                    st.session_state.optimization_results = {}
                
                st.session_state.optimization_results[f"{model_type}_{horizon}"] = {
                    'best_params': best_params,
                    'metrics': metrics.to_dict(),
                    'model_type': model_type,
                    'horizon': horizon
                }
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display previous results
    if 'optimization_results' in st.session_state:
        st.subheader("Previous Optimization Results")
        
        for key, result in st.session_state.optimization_results.items():
            with st.expander(f"{key}"):
                st.json(result.get('best_params', {}))
                metrics = result.get('metrics', {})
                if isinstance(metrics, dict):
                    st.write(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
                    st.write(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                    st.write(f"Win Rate: {metrics.get('win_rate', 0):.4f}")

def render_live_prediction(df):
    """Render the live prediction page."""
    st.header("Live AI Prediction")
    
    # Initialize AI predictor
    if st.session_state.ai_predictor is None:
        st.session_state.ai_predictor = AIPredictor()
    
    predictor = st.session_state.ai_predictor
    
    # Load models
    if not predictor.loaded:
        horizons = list(HORIZON_CONFIGS.keys())
        predictor.load_models(horizons)
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_horizon = st.selectbox(
            "Select Horizon",
            list(HORIZON_CONFIGS.keys()),
            index=list(HORIZON_CONFIGS.keys()).index("60s")
        )
    
    with col2:
        selected_model = st.radio(
            "Model",
            ["XGBoost", "LSTM"],
            help="Select prediction model"
        )
    
    # Live input features
    st.subheader("Current Market Data")
    
    # Create input form
    with st.form(key="live_prediction_form"):
        st.caption("Enter current market parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            current_price = st.number_input("Current Price", value=1250.40, format="%f")
            rsi = st.slider("RSI", 0.0, 100.0, 62.0)
            ema_9 = st.number_input("EMA 9", value=1250.20, format="%f")
            ema_21 = st.number_input("EMA 21", value=1249.80, format="%f")
            ema_50 = st.number_input("EMA 50", value=1249.50, format="%f")
        
        with col2:
            macd = st.number_input("MACD", value=0.30, format="%f")
            macd_signal = st.number_input("MACD Signal", value=0.10, format="%f")
            macd_hist = st.number_input("MACD Histogram", value=0.20, format="%f")
            volatility = st.slider("Volatility (%)", 0.1, 5.0, 0.72, format="%.2f")
        
        col1, col2 = st.columns(2)
        with col1:
            return_1 = st.number_input("Return 1-min", value=0.18, format="%f")
            return_5 = st.number_input("Return 5-min", value=0.35, format="%f")
            return_30 = st.number_input("Return 30-min", value=0.50, format="%f")
        
        with col2:
            candle_body = st.number_input("Candle Body", value=0.15, format="%f")
            upper_wick = st.number_input("Upper Wick", value=0.04, format="%f")
            lower_wick = st.number_input("Lower Wick", value=0.02, format="%f")
        
        predict_button = st.form_submit_button("🔮 Generate Prediction", type="primary")
    
    if predict_button:
        # Generate prediction
        with st.spinner("Generating AI prediction..."):
            try:
                # Create live signal
                signal = predictor.predict_live_signal(
                    current_price=current_price,
                    rsi=rsi,
                    ema_9=ema_9,
                    ema_21=ema_21,
                    ema_50=ema_50,
                    macd=macd,
                    macd_signal=macd_signal,
                    macd_hist=macd_hist,
                    volatility=volatility,
                    return_1=return_1,
                    return_5=return_5,
                    return_30=return_30,
                    candle_body=candle_body,
                    upper_wick=upper_wick,
                    lower_wick=lower_wick,
                    horizon=selected_horizon
                )
                
                # Display results
                st.subheader("Prediction Results")
                
                # Main prediction display
                col1, col2, col3 = st.columns(3)
                with col1:
                    direction_color = "🟢" if signal['direction'] == 'UP' else "🔴"
                    st.metric(
                        "Direction", 
                        f"{direction_color} {signal['direction']}",
                        f"Confidence: {signal['confidence']:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Probability UP", 
                        f"{signal['probability_up']:.2%}",
                        f"Probability DOWN: {signal['probability_down']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Recommended Stake", 
                        f"{signal['recommended_stake_pct']:.1f}% of balance",
                        f"Duration: {signal['suggested_duration']}"
                    )
                
                # Signal metadata
                with st.expander("Signal Details"):
                    st.write(f"**Horizon:** {signal['horizon']}")
                    st.write(f"**Threshold:** {signal.get('signal_metadata', 0.55):.2f}")
                    st.write(f"**Direction:** {signal['direction']}")
                    
                    # Input features summary
                    st.write("**Input Features Summary:**")
                    st.write(f"- Price: {signal['input_features']['price']}")
                    st.write(f"- RSI: {signal['input_features']['rsi']}")
                    st.write(f"- EMA Trend: {signal['input_features']['ema_trend']}")
                    st.write(f"- MACD Histogram: {signal['input_features']['macd_hist']:.4f}")
                    st.write(f"- Volatility: {signal['input_features']['volatility']:.2f}%")
                    st.write(f"- Recent Returns: {[f'{r:.2%}' for r in signal['input_features']['recent_returns']]}")
                    st.write(f"- Candle Pattern: {signal['input_features']['candle_pattern']}")
                
                # Store prediction in history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'horizon': selected_horizon,
                    'direction': signal['direction'],
                    'confidence': signal['confidence']
                })
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.subheader("Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Plot prediction trends
        fig = px.line(
            history_df,
            x='timestamp',
            y=pd.to_numeric(history_df['direction'].map({'UP': 1, 'DOWN': -1})),
            title="Prediction Direction Over Time",
            labels={'value': 'Direction Score', 'timestamp': 'Time'}
        )
        
        # Add threshold line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        up_count = (history_df['direction'] == 'UP').sum()
        down_count = (history_df['direction'] == 'DOWN').sum()
        total = len(history_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", total)
        with col2:
            st.metric("UP Predictions", up_count, f"{up_count/total*100:.1f}%" if total > 0 else "0%")
        with col3:
            st.metric("DOWN Predictions", down_count, f"{down_count/total*100:.1f}%" if total > 0 else "0%")

if __name__ == "__main__":
    main()


def render_multi_agent_system(df):
    """Render the Phase 4 Multi-Agent Decision System page."""
    st.header("🤖 Phase 4: Multi-Agent Decision System")
    st.markdown("---")
    
    st.markdown("""
    The Multi-Agent Decision System runs 5 specialized agents in parallel:
    
    1. **Technical Agent** — Analyzes RSI, MACD, EMA, Support/Resistance
    2. **Regime Agent** — Determines Trending, Ranging, High/Low Volatility  
    3. **AI Prediction Agent** — Uses trained ML models for direction probability
    4. **Duration Agent** — Recommends optimal contract duration
    5. **Decision Agent** — Final arbiter that combines all signals
    """)
    
    # Initialize session state for multi-agent
    if 'multi_agent_engine' not in st.session_state:
        from trading.execution.execution_engine import BinaryTradingEngine
        st.session_state.multi_agent_engine = BinaryTradingEngine(
            initial_balance=100.0,
            max_risk_pct=2.0,
            max_drawdown_pct=10.0
        )
    
    engine = st.session_state.multi_agent_engine
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        trend_mode = st.selectbox(
            "Market Scenario",
            ["UP Trending", "DOWN Trending", "High Volatility", "Ranging", "Low Volatility"],
            index=0
        )
    
    with col2:
        num_bars = st.slider("Data Points", 50, 500, 200, 50)
    
    with col3:
        if st.button("🔄 Run Multi-Agent Analysis", type="primary"):
            st.session_state.run_analysis = True
    
    # Run analysis
    if st.session_state.get('run_analysis', False):
        with st.spinner("Running all 5 agents..."):
            # Create test data based on scenario
            test_df = create_test_scenario_data(df, trend_mode, num_bars)
            current_price = test_df['close'].iloc[-1]
            
            # Run multi-agent analysis
            decision = engine.process_multi_agent_signal(test_df, current_price)
            st.session_state.last_decision = decision
            st.session_state.test_data = test_df
            
            # Execute if trade is pending
            if decision.get('trade_status') == 'PENDING_EXECUTION':
                trade_result = asyncio.run(engine.execute_decision_trade(decision, current_price))
                st.session_state.last_trade = trade_result
        
        st.session_state.run_analysis = False
        st.rerun()
    
    # Display results
    if 'last_decision' in st.session_state:
        decision = st.session_state.last_decision
        
        # Decision display
        st.markdown("### 📋 Decision Output")
        
        # Create visual decision tree
        decision_text = format_decision_tree(decision)
        st.code(decision_text, language=None)
        
        # Agent details in columns
        st.markdown("### 🔍 Agent Breakdown")
        
        if 'agent_contributions' in decision:
            agents = decision['agent_contributions']
            
            cols = st.columns(4)
            
            # Technical Agent
            with cols[0]:
                tech = agents['technical']
                st.markdown("**🔧 Technical Agent**")
                st.metric("Direction", tech['direction'])
                st.metric("Confidence", f"{tech['confidence']:.0%}")
            
            # Regime Agent
            with cols[1]:
                regime = agents['regime']
                st.markdown("**🌊 Regime Agent**")
                st.metric("Regime", regime['regime'])
                st.metric("Trend", regime['trend_direction'])
                st.metric("Volatility", regime['volatility'])
            
            # AI Agent
            with cols[2]:
                ai = agents['ai']
                st.markdown("**🧠 AI Agent**")
                st.metric("Direction", ai['direction'])
                st.metric("Confidence", f"{ai['confidence']:.0%}")
                st.metric("UP Probability", f"{ai['probabilities']['UP']:.0%}")
                st.metric("DOWN Probability", f"{ai['probabilities']['DOWN']:.0%}")
            
            # Duration Agent
            with cols[3]:
                dur = agents['duration']
                st.markdown("**⏱️ Duration Agent**")
                st.metric("Duration", dur['duration_label'])
                st.metric("Confidence", f"{dur['confidence']:.0%}")
                st.caption(dur['reason'])
        
        # Trade execution result
        if 'last_trade' in st.session_state:
            trade = st.session_state.last_trade
            st.markdown("### 💰 Trade Execution Result")
            
            if trade.get('status') == 'EXECUTED':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Result", trade['result'])
                with col2:
                    st.metric("Stake", f"${trade['stake_amount']:.2f}")
                with col3:
                    st.metric("Payout", f"${trade['payout_amount']:.2f}")
                with col4:
                    st.metric("Balance", f"${trade['balance_after']:.2f}")
            else:
                st.warning(f"Trade not executed: {trade.get('reason', 'Unknown')}")
    
    # Equity curve
    if hasattr(engine, 'equity_curve') and len(engine.equity_curve) > 1:
        st.markdown("### 📈 Equity Curve")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=engine.equity_curve,
            mode='lines',
            name='Balance',
            line=dict(color='green' if engine.equity_curve[-1] >= engine.equity_curve[0] else 'red')
        ))
        fig.add_hline(y=engine.equity_curve[0], line_dash="dash", line_color="gray", annotation_text="Start")
        fig.update_layout(
            xaxis_title="Trades",
            yaxis_title="Balance ($)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    if 'test_data' in st.session_state:
        with st.expander("📊 View Test Data"):
            st.dataframe(st.session_state.test_data.tail(10), use_container_width=True)


def create_test_scenario_data(base_df, scenario: str, n_bars: int):
    """Create test data for different market scenarios."""
    import numpy as np
    from datetime import datetime, timedelta
    
    base_price = base_df['close'].iloc[-1] if len(base_df) > 0 else 100.0
    
    if scenario == "UP Trending":
        trend = np.linspace(0, 8, n_bars)
        vol = 0.015
    elif scenario == "DOWN Trending":
        trend = np.linspace(0, -8, n_bars)
        vol = 0.015
    elif scenario == "High Volatility":
        trend = np.zeros(n_bars)
        vol = 0.04
    elif scenario == "Ranging":
        # Oscillate around base
        t = np.arange(n_bars)
        trend = 2 * np.sin(t * 0.1) + 1 * np.sin(t * 0.2)
        vol = 0.01
    elif scenario == "Low Volatility":
        trend = np.linspace(0, 2, n_bars)
        vol = 0.005
    else:
        trend = np.zeros(n_bars)
        vol = 0.015
    
    noise = np.random.normal(0, vol, n_bars)
    prices = base_price + trend + np.cumsum(noise)
    
    dates = pd.date_range(start=datetime.now() - timedelta(minutes=n_bars*5), periods=n_bars, freq='5min')
    
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, vol/2, n_bars)),
        'low': prices - np.abs(np.random.normal(0, vol/2, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    
    return test_df


def format_decision_tree(decision: dict) -> str:
    """Format decision as ASCII tree."""
    action = decision.get('decision_action', 'UNKNOWN')
    
    lines = [
        "┌─────────────────────────────────────────┐",
        "│       MULTI-AGENT DECISION SYSTEM       │",
        "└─────────────────────────────────────────┘",
        "",
        "┌───────────────────────┐",
        f"│ TECHNICAL AGENT       │",
        f"│ {decision['agent_contributions']['technical']['direction']:>4} — {decision['agent_contributions']['technical']['confidence']:.0%}              │",
        "└───────────┬───────────┘",
        "            │",
        "┌───────────▼───────────┐",
        f"│ REGIME AGENT          │",
        f"│ {decision['agent_contributions']['regime']['regime']:<15} │",
        "└───────────┬───────────┘",
        "            │",
        "┌───────────▼───────────┐",
        "│ AI AGENT              │",
        f"│ UP  — {decision['agent_contributions']['ai']['probabilities']['UP']:.0%}              │",
        f"│ DOWN — {decision['agent_contributions']['ai']['probabilities']['DOWN']:.0%}              │",
        "└───────────┬───────────┘",
        "            │",
        "            ▼",
        "       DECISION",
    ]
    
    if action == "TRADE_UP":
        lines.extend([
            "       TRADE UP",
            f"       {decision.get('duration_label', '60s')}"
        ])
    elif action == "TRADE_DOWN":
        lines.extend([
            "       TRADE DOWN",
            f"       {decision.get('duration_label', '60s')}"
        ])
    else:
        lines.extend([
            "       NO TRADE",
            f"       Reason: {decision.get('reject_reason', 'Unknown')}"
        ])
    
    return "\n".join(lines)


def render_risk_management():
    """Render Phase 5 Risk Management System page."""
    st.header("🛡️ Phase 5: Risk Management System")
    st.markdown("---")
    
    st.markdown("""
    **Risk Management sits between the Decision Agent and Broker execution.**
    
    Every trade must pass 7 risk checks:
    1. **Confidence Threshold** — AI confidence must be ≥ 75%
    2. **Stake Size** — Stake must not exceed max risk % of balance
    3. **Daily Loss Limit** — Daily losses capped at $10 (pauses bot)
    4. **Consecutive Losses** — After 3 losses, bot pauses
    5. **Max Open Trades** — Maximum 3 simultaneous trades
    6. **Cooldown** — 60 seconds wait after each trade
    7. **Drawdown** — Max drawdown before stopping (10%)
    """)
    
    # Initialize risk manager if not exists
    if 'risk_manager' not in st.session_state:
        from trading.execution.execution_engine import RiskManager
        st.session_state.risk_manager = RiskManager(
            balance=100.0,
            max_risk_pct=1.0,
            max_drawdown_pct=10.0,
            daily_loss_limit=10.0,
            max_open_trades=3,
            confidence_threshold=0.75,
            cooldown_seconds=60
        )
    
    rm = st.session_state.risk_manager
    
    # Risk status display
    st.markdown("### 📊 Current Risk Status")
    status = rm.get_risk_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Balance", f"${status['balance']:.2f}")
    with col2:
        st.metric("Daily P&L", f"${status['daily_pnl']:.2f}")
        if status['daily_pnl'] < 0:
            st.caption(f"Loss: ${status['daily_loss_amount']:.2f}")
    with col3:
        st.metric("Trades Today", status['trades_today'])
        st.caption(f"Win Rate: {status['win_rate_today']:.0f}%")
    with col4:
        if status['is_paused']:
            st.error("⏸️ PAUSED")
            st.caption(status['pause_reason'])
        else:
            st.success("✅ TRADING")
    
    # Risk checks display
    st.markdown("### ✅ Risk Checks Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        conf_pass = status['confidence_threshold'] >= 0.75
        st.checkbox(f"Confidence ≥ 75%", conf_pass)
        max_stake = status['balance'] * (rm.max_risk_pct / 100)
        st.checkbox(f"Stake ≤ ${max_stake:.2f}", True)
    
    with col2:
        loss_ok = status['daily_loss_amount'] < status['daily_loss_limit']
        st.checkbox(f"Daily Loss OK", loss_ok)
        loss_ok2 = status['consecutive_losses'] < 3
        st.checkbox(f"Consecutive Losses < 3", loss_ok2)
    
    with col3:
        trades_ok = status['open_trades'] < status['max_open_trades']
        st.checkbox(f"Open Trades OK", trades_ok)
        cooldown_ok = status['seconds_to_cooldown'] == 0
        st.checkbox(f"Cooldown Complete", cooldown_ok)
    
    # Risk parameters
    st.markdown("### ⚙️ Risk Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Max Stake:** {rm.max_risk_pct}% of balance")
        st.write(f"**Daily Loss Limit:** ${rm.daily_loss_limit:.2f}")
        st.write(f"**Consecutive Loss Limit:** 3")
    with col2:
        st.write(f"**Max Open Trades:** {rm.max_open_trades}")
        st.write(f"**Confidence Threshold:** {rm.confidence_threshold:.0%}")
        st.write(f"**Cooldown:** {rm.cooldown_seconds}s")
    with col3:
        st.write(f"**Current Drawdown:** {status['current_drawdown']:.2f}%")
        st.write(f"**Max Drawdown:** {rm.max_drawdown_pct:.1f}%")
        st.write(f"**Open Trades:** {status['open_trades']}/{rm.max_open_trades}")
    
    # Test validation
    st.markdown("### 🧪 Test Risk Validation")
    
    col1, col2 = st.columns(2)
    with col1:
        test_confidence = st.slider("AI Confidence", 0.0, 1.0, 0.80, 0.05)
        test_stake = st.number_input("Recommended Stake ($)", 0.0, 100.0, 2.0, 0.5)
    
    with col2:
        test_direction = st.selectbox("Direction", ["UP", "DOWN"])
        
        if st.button("Validate Trade", type="primary"):
            result = rm.validate_trade(
                signal_direction=test_direction,
                signal_confidence=test_confidence,
                entry_price=100.0,
                recommended_stake=test_stake,
                current_balance=rm.balance
            )
            
            if result['is_valid']:
                st.success("✅ **APPROVED**")
            else:
                st.error("❌ **REJECTED**")
            
            st.write(f"**Reason:** {result['reason']}")
            
            if result.get('checks'):
                st.write("**Checks:**")
                for check, passed in result['checks'].items():
                    icon = "✅" if passed else "❌"
                    st.write(f"  {icon} {check.replace('_', ' ').title()}")
    
    # Reset button
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reset Daily Stats"):
            rm.reset_daily_stats()
            st.success("Daily stats reset!")
            st.rerun()
    with col2:
        if status['is_paused'] and st.button("▶️ Resume Trading"):
            rm.resume_trading()
            st.success("Trading resumed!")
            st.rerun()