"""
Project scaffolding script for FashionMNIST-Analysis.

Run this once after cloning the repository to create any missing
directories that are excluded from version control (e.g. data/, logs/).

The script is idempotent: existing folders are silently skipped.

Usage:
    python setup_project.py
"""

import os

# ---------------------------------------------------------------------------
# Directory structure
# Note: src/ subpackages and notebooks/ are tracked in git and not listed
# here. Only runtime directories that are gitignored need to be created.
# ---------------------------------------------------------------------------
folders = [
    "data/raw",
    "data/processed",
    "models/all_models",
    "models/best_model_weights",
    "logs",
    "checkpoints",
    "tests",
    "eda",
    "figures/EDA_plots",
    "figures/modeling_plots",
    "figures/evaluation_plots",
    "figures/Traditional_ML_Algo_plots",
    "notebooks",
    "results/fine_tuning_results",
    "results/Traditional_ML_Algo_results",
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
]

# Minimal .gitignore added only when the file does not yet exist
gitignore_content = """\
# Python cache
__pycache__/
*.pyc

# Virtual environments
venv/
env/
envf/

# Jupyter checkpoints
.ipynb_checkpoints/

# Logs and secrets
*.log
*.env
"""

# Function to create directories
def create_folders():
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created folder: {folder}")
        else:
            print(f"🔍 Folder already exists, skipping: {folder}")

# Function to create essential files
def create_files():
    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                if file == ".gitignore":
                    f.write(gitignore_content)
                print(f"✅ Created file: {file}")
        else:
            print(f"🔍 File already exists, skipping: {file}")

# Main function
def main():
    print("\n📂 Setting up project structure...")
    create_folders()
    create_files()
    print("\n🎉 Project setup complete!")

if __name__ == "__main__":
    main()