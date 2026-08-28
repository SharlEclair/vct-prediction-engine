# Repository Open-Source Transition Guidelines

## 1. Security & Credentials (Zero-Tolerance)
*   **API Keys & Secrets:** No `.env` files, hardcoded keys (Gemini, OpenAI), or access tokens may exist in the commit history or active working tree.
*   **Local Paths:** Remove any hardcoded local system paths (e.g., `C:/Users/.../Desktop/vct-prediction-model`) from documentation and scripts. Replace with relative paths (e.g., `./data/`).

## 2. Data Segregation & Storage
*   **Raw Data:** The `data/raw/` directory (wikitext dumps, HTML scrapes) must be excluded.
*   **Processed Data:** The `data/processed/` directory (JSON databases, telemetry) must be excluded to avoid licensing issues with scraped VLR/Wiki data and repository bloat.
*   **Mock Data:** Provide minimal, anonymized sample schemas or mock JSON files so users can test the PyTorch graph and MILP optimizer without needing the full 2026 database.

## 3. Environment & Caching Clutter
*   Exclude all Python virtual environments (`venv/`, `env/`, `.conda/`).
*   Exclude all bytecode and cache directories (`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`).
*   Exclude OS-specific metadata (`.DS_Store`, `Thumbs.db`).

## 4. Documentation Standards
*   **README.md:** Must include setup instructions, environment variable templates (without real keys), and an architectural overview.
*   **LICENSE:** An open-source license (e.g., MIT, Apache 2.0) must be explicitly defined.
*   **Clean History:** Ensure the `.git` history does not contain previously committed data files or `.env` files that were later deleted. (If history is dirty, a git history rewrite or squashing to a fresh initial commit is required before publishing).