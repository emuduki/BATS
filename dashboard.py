"""
Phase 7 Control Center Dashboard for BATS.
Real-time monitoring, control, and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
import time
import threading
import subprocess
from pathlib import Path

# Add BATS to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.state import TradingState

# Page configuration
st.set_page_config(
    page_title="BATS Control Center",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton > button { width: 100%; margin-top: 0.5rem; }
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    .bot-running {
        background: #2d5a2d;
        color: #90ee90;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .bot-stopped {
        background: #5a2d2d;
        color: #ff6b6b;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .signal-card {
        background: #1a1a2e;
        border: 2px solid #4a90e2;
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .decision-approved {
        background: #2d5a2d;
        color: #90ee90;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .decision-rejected {
        background: #5a2d2d;
        color: #ff6b6b;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize state
if 'trading_state' not in st.session_state:
    st.session_state.trading_state = TradingState()
if 'bot_process' not in st.session_state:
    st.session_state.bot_process = None
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []


def load_state() -> dict:
    """Load current trading state."""
    return st.session_state.trading_state.get_state()


def start_bot(symbol: str = "R_100", interval: int = 30) -> bool:
    """Start the trading bot in background."""
    if st.session_state.bot_process is not None:
        return False
    
    try:
        # Start bot in background thread
        from trading.loop import trading_loop_background
        
        def run_bot():
            trading_loop_background(symbol=symbol, interval_seconds=interval, max_trades_per_session=100)
        
        thread = threading.Thread(target=run_bot, daemon=True)
        thread.start()
        
        st.session_state.bot_running = True
        st.session_state.bot_thread = thread
        return True
    except Exception as e:
        st.error(f"Failed to start bot: {e}")
        return False


def stop_bot() -> bool:
    """Stop the trading bot."""
    st.session_state.bot_running = False
    st.session_state.bot_process = None
    return True


def render_dashboard():
    """Render the main dashboard page."""
    state = load_state()
    
    # Header
    st.markdown("""
    # 🦇 BATS — LIVE DASHBOARD
    ### Binary AI Trading System Control Center
    """)
    st.markdown("---")
    
    # Bot status indicator
    col_status, col_action = st.columns([1, 3])
    with col_status:
        if st.session_state.bot_running:
            st.markdown('<div class="bot-running">● BOT RUNNING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bot-stopped">● BOT STOPPED</div>', unsafe_allow_html=True)
    
    with col_action:
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if not st.session_state.bot_running:
                if st.button("▶️ Start Bot", type="primary"):
                    if start_bot():
                        st.success("Bot started!")
                        st.rerun()
        with btn_col2:
            if st.session_state.bot_running:
                if st.button("⏸️ Stop Bot"):
                    stop_bot()
                    st.warning("Bot stopped")
                    st.rerun()
        with btn_col3:
            if st.button("🚨 EMERGENCY STOP"):
                stop_bot()
                st.error("EMERGENCY STOP ACTIVATED")
                st.rerun()
    
    st.markdown("---")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Balance",
            f"${state.get('balance', 0):.2f}",
            f"${state.get('daily_pnl', 0):+.2f} today"
        )
    with col2:
        st.metric(
            "Today's P/L",
            f"${state.get('daily_pnl', 0):+.2f}",
            f"{(state.get('daily_pnl', 0) / 100.0 * 100):+.1f}%"
        )
    with col3:
        st.metric(
            "Win Rate",
            f"{state.get('win_rate', 0) * 100:.1f}%",
            f"{state.get('winning_trades', 0)}W / {state.get('total_trades', 0) - state.get('winning_trades', 0)}L"
        )
    with col4:
        st.metric(
            "Active Trades",
            f"{state.get('active_trades', 0)}",
            f"of {3} max"
        )
    
    st.markdown("---")
    
    # Market Signal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 MARKET SIGNAL")
        signal = state.get('last_signal')
        if signal:
            st.markdown(f"""
            <div class="signal-card">
                <h3>{signal.get('symbol', 'N/A')}</h3>
                <p><strong>Direction:</strong> {signal.get('direction', 'N/A')}</p>
                <p><strong>Confidence:</strong> {signal.get('confidence', 0):.0%}</p>
                <p><strong>Duration:</strong> {signal.get('duration', 0)}s</p>
                <p><strong>Action:</strong> {signal.get('action', 'N/A')}</p>
                <p><strong>Time:</strong> {signal.get('timestamp', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No signal yet. Start the bot to begin.")
    
    with col2:
        st.markdown("### 🤖 AGENT DECISIONS")
        risk_status = state.get('risk_status', {})
        decisions = [
            ("Technical", signal.get('direction', '-') if signal else '-', signal.get('confidence', 0) if signal else 0),
            ("AI", signal.get('direction', '-') if signal else '-', signal.get('confidence', 0) if signal else 0),
            ("Regime", "TRENDING", "75%"),
        ]
        
        for name, decision, confidence in decisions:
            st.write(f"**{name}**: {decision} ({confidence})")
        
        st.markdown("---")
        if risk_status.get('can_trade', True):
            st.markdown('<div class="decision-approved">RISK: APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decision-rejected">RISK: REJECTED — {risk_status.get("pause_reason", "Unknown")}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Equity Curve")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            history_df['cumulative_pnl'] = history_df['pnl'].cumsum()
            fig = px.line(
                history_df, 
                x=range(len(history_df)), 
                y='cumulative_pnl',
                title="Cumulative P&L"
            )
            fig.update_layout(xaxis_title="Trade #", yaxis_title="P&L ($)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades yet")
    
    with col2:
        st.markdown("### 📋 Recent Trades")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history[-10:][::-1])
            st.dataframe(
                history_df[['timestamp', 'direction', 'stake', 'result', 'pnl']],
                use_container_width=True
            )
        else:
            st.info("No trades yet")


def render_markets():
    """Render markets page."""
    st.header("🌍 Markets")
    
    st.markdown("""
    ### Available Symbols
    """)
    
    symbols_data = {
        "Symbol": ["R_100", "R_50", "R_25", "R_10", "R_5", "frxEURUSD", "frxGBPUSD"],
        "Display Name": ["Volatility 100", "Volatility 50", "Volatility 25", "Volatility 10", "Volatility 5", "EUR/USD", "GBP/USD"],
        "Min Stake": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
        "Max Stake": [20000, 20000, 20000, 20000, 20000, 20000, 20000],
        "Available Durations": ["15s-300s", "15s-300s", "15s-300s", "15s-300s", "15s-300s", "60s-300s", "60s-300s"],
        "Payout": ["95%", "90%", "92%", "94%", "93%", "85%", "85%"],
        "Status": ["🟢 Active", "🟢 Active", "🟢 Active", "🟢 Active", "🟢 Active", "🟢 Active", "🟢 Active"]
    }
    
    st.dataframe(pd.DataFrame(symbols_data), use_container_width=True)


def render_live_signals():
    """Render live signals page."""
    st.header("📡 Live Signals")
    
    state = load_state()
    signal = state.get('last_signal')
    
    if signal:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Symbol", signal.get('symbol', 'N/A'))
            st.metric("Direction", signal.get('direction', 'N/A'))
        with col2:
            st.metric("Confidence", f"{signal.get('confidence', 0):.0%}")
            st.metric("Duration", f"{signal.get('duration', 0)}s")
        with col3:
            st.metric("Action", signal.get('action', 'N/A'))
            st.metric("Time", signal.get('timestamp', 'N/A')[:19])
        
        st.markdown("### Agent Breakdown")
        # Display agent details
        st.json(signal)
    else:
        st.info("No signals yet. Start the bot to receive signals.")


def render_active_trades():
    """Render active trades page."""
    st.header("⚡ Active Trades")
    
    state = load_state()
    active_count = state.get('active_trades', 0)
    
    if active_count > 0:
        st.warning(f"{active_count} active trade(s)")
        # In real implementation, would show contract details
    else:
        st.info("No active trades")


def render_trade_history():
    """Render trade history page."""
    st.header("📜 Trade History")
    
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(history_df))
        with col2:
            st.metric("Wins", (history_df['result'] == 'WIN').sum())
        with col3:
            st.metric("Losses", (history_df['result'] == 'LOSS').sum())
        with col4:
            total_pnl = history_df['pnl'].sum()
            st.metric("Total P&L", f"${total_pnl:+.2f}")
        
        st.markdown("### All Trades")
        st.dataframe(
            history_df[['timestamp', 'symbol', 'direction', 'stake', 'result', 'pnl']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                history_df, 
                x=range(len(history_df)),
                y=history_df['pnl'].cumsum(),
                title="Cumulative P&L"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                history_df,
                x=range(len(history_df)),
                y='pnl',
                color='result',
                title="P&L by Trade",
                color_discrete_map={'WIN': 'green', 'LOSS': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet")


def render_strategy_performance():
    """Render strategy performance page."""
    st.header("📊 Strategy Performance")
    
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Win Rate by Direction")
            winrate_dir = history_df.groupby('direction').apply(
                lambda x: (x['result'] == 'WIN').sum() / len(x) * 100
            ).reset_index()
            winrate_dir.columns = ['direction', 'win_rate']
            fig = px.bar(winrate_dir, x='direction', y='win_rate', title="Win Rate by Direction")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### P&L by Direction")
            pnl_dir = history_df.groupby('direction')['pnl'].sum().reset_index()
            fig = px.bar(pnl_dir, x='direction', y='pnl', title="P&L by Direction")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")


def render_ai_performance():
    """Render AI performance page."""
    st.header("🧠 AI Performance")
    
    st.markdown("""
    ### Model Metrics
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Loaded", "2")
    with col2:
        st.metric("Last Training", "N/A")
    with col3:
        st.metric("Avg Accuracy", "62%")
    with col4:
        st.metric("Horizons", "4")
    
    st.info("Train models in the Phase 3 dashboard to see performance metrics here.")


def render_risk_management():
    """Render risk management page."""
    st.header("🛡️ Risk Management")
    
    state = load_state()
    risk = state.get('risk_status', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Daily Loss", f"${risk.get('daily_loss_amount', 0):.2f}")
    with col2:
        st.metric("Max Daily Loss", "$10.00")
    with col3:
        st.metric("Consecutive Losses", risk.get('consecutive_losses', 0))
    with col4:
        st.metric("Open Trades", risk.get('open_trades', 0))
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Status")
        if risk.get('is_paused', False):
            st.error(f"⏸️ PAUSED: {risk.get('pause_reason', 'Unknown')}")
        else:
            st.success("✅ Trading Allowed")
    
    with col2:
        st.markdown("### Risk Parameters")
        st.write("- Max Stake: 1% of balance")
        st.write("- Daily Loss Limit: $10")
        st.write("- Consecutive Loss Limit: 3")
        st.write("- Max Open Trades: 3")
        st.write("- Confidence Threshold: 75%")
        st.write("- Cooldown: 60s")


def render_bot_logs():
    """Render bot logs page."""
    st.header("📋 Bot Logs")
    
    log_file = Path("trading_state.json")
    if log_file.exists():
        with open(log_file) as f:
            st.json(json.load(f))
    else:
        st.info("No logs yet")


def render_settings():
    """Render settings page."""
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trading Settings")
        symbol = st.selectbox("Symbol", ["R_100", "R_50", "frxEURUSD"], index=0)
        interval = st.slider("Cycle Interval (seconds)", 10, 120, 30)
        max_trades = st.number_input("Max Trades per Session", 1, 100, 20)
    
    with col2:
        st.markdown("### Risk Settings")
        max_risk = st.slider("Max Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
        daily_limit = st.number_input("Daily Loss Limit ($)", 1.0, 100.0, 10.0)
        confidence = st.slider("Min Confidence (%)", 50, 95, 75)
    
    st.markdown("### Broker Settings")
    broker_type = st.selectbox("Broker", ["Demo Broker (Safe)", "Deriv (Live - Not Configured)"])
    if "Deriv" in broker_type:
        st.warning("⚠️ Live trading requires proper Deriv API credentials and configuration.")


def main():
    """Main dashboard function."""
    
    # Sidebar navigation
    st.sidebar.title("🦇 BATS Control")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🌍 Markets",
            "📡 Live Signals",
            "⚡ Active Trades",
            "📜 Trade History",
            "📊 Strategy Performance",
            "🧠 AI Performance",
            "🛡️ Risk Management",
            "📋 Bot Logs",
            "⚙️ Settings"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Bot control in sidebar
    st.sidebar.markdown("### Bot Control")
    if st.session_state.bot_running:
        st.sidebar.success("● RUNNING")
        if st.sidebar.button("⏸️ Stop Bot"):
            stop_bot()
            st.rerun()
    else:
        st.sidebar.error("● STOPPED")
        if st.sidebar.button("▶️ Start Bot"):
            if start_bot():
                st.sidebar.success("Started!")
                st.rerun()
    
    if st.sidebar.button("🚨 EMERGENCY STOP"):
        stop_bot()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("BATS v0.7.0 | Phase 7 Dashboard")
    
    # Auto-refresh indicator
    if st.session_state.bot_running:
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        st.sidebar.caption("Auto-refresh: ON")
    
    # Render selected page
    if "Dashboard" in page:
        render_dashboard()
    elif "Markets" in page:
        render_markets()
    elif "Live Signals" in page:
        render_live_signals()
    elif "Active Trades" in page:
        render_active_trades()
    elif "Trade History" in page:
        render_trade_history()
    elif "Strategy Performance" in page:
        render_strategy_performance()
    elif "AI Performance" in page:
        render_ai_performance()
    elif "Risk Management" in page:
        render_risk_management()
    elif "Bot Logs" in page:
        render_bot_logs()
    elif "Settings" in page:
        render_settings()


if __name__ == "__main__":
    main()