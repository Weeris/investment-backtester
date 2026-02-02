# GitHub Setup Instructions

To push this code to your GitHub repository, you need to authenticate with GitHub using either:

## Option 1: Personal Access Token (Recommended)
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. Copy the token
4. Run: `git remote set-url origin https://<token>@github.com/Weeris/investment-backtester.git`
5. Then run: `git push -u origin main`

## Option 2: SSH Key
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to ssh-agent: `eval "$(ssh-agent -s)"` and `ssh-add ~/.ssh/id_ed25519`
3. Copy public key to GitHub: `cat ~/.ssh/id_ed25519.pub`
4. Add to GitHub: Settings → SSH and GPG keys → New SSH key
5. Change remote: `git remote set-url origin git@github.com:Weeris/investment-backtester.git`
6. Then run: `git push -u origin main`

## Current Status
Repository is initialized and code is ready to push. You just need to authenticate with GitHub.

The files in this repository:
- app.py: Streamlit web application for investment backtesting
- requirements.txt: Python dependencies
- Procfile: For Heroku deployment
- setup.sh: Setup script for deployment
- README.md: Documentation
- .streamlit/config.toml: UI configuration for Streamlit