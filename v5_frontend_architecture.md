1. Tab Restructuring & Sidebar Migration
The layout topology of app.py is being inverted to prioritize the exploratory tools.

New Tab Order: [ "⚡ Open Simulation", "📊 Match Analysis", "🧠 Roster Optimizer", "📋 VFL Players" ]

Stateful Form Migration: The global settings (Match ID, Series Format, Veto Mode) currently housed in st.sidebar must be migrated into an st.expander or control panel at the very top of the "Match Analysis" tab. The sidebar should be reserved exclusively for global app navigation or left completely empty for a cleaner, full-width UI.

2. Actual vs. Predicted Validation (Match Analysis)
Since the pipeline now ingests historical matches via the VLR.gg scraper, the "Match Analysis" tab can act as a backtesting visualizer. When a user inputs a past Match ID, the UI must fetch the actual parsed results from data/raw/match_{id}.json and render them adjacent to the engine's predicted results.

Display Actual Series Score vs. Predicted Series Score.

Display Actual Map Vetoes vs. Predicted Vetoes.

Display Actual Agent Compositions vs. Predicted Compositions.