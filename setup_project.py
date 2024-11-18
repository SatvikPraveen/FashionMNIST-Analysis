import os

# Define the project structure
folders = [
    "data",
    "data_preparation",
    "models",
    "models/all_models",
    "models/best_model",
    "models/best_model_weights",
    "src",
    "tests",
    "eda",
    "figures",
    "figures/EDA_plots",
    "figures/modeling_plots",
    "figures/fine_tuning_plots",
    "figures/evaluation_plots"
    "notebooks"
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore"
]

# Updated .gitignore content
gitignore_content = """
# Python cache and compiled files
__pycache__/
*.pyc

# Virtual environment folder (updated to match your environment name)
envf/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Log files and environment configuration
*.log
*.env

# Ignore specific temporary or backup files
*.bak
*.swp

# Uncomment these lines if you want to ignore figures and processed CSV files
# figures/
# data_preparation/
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