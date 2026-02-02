#!/bin/bash
# Setup script for Heroku deployment

echo "Setting up Investment Backtesting Platform..."

# Create cache directory
mkdir -p data_cache

# Install any additional dependencies if needed
pip install -r requirements.txt

echo "Setup complete!"