#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${YELLOW}==> $1${NC}"
    echo "----------------------------------------"
}

# Function to handle errors
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if Python is installed
section "Checking Python installation"
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 is required but not installed. Please install Python 3.8 or higher."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
echo "Python $PYTHON_VERSION is installed"

# Set up virtual environment
section "Setting up virtual environment"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || error_exit "Failed to create virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate  # Linux/Mac
else
    source venv/Scripts/activate  # Windows
fi

# Upgrade pip
section "Upgrading pip"
pip install --upgrade pip || error_exit "Failed to upgrade pip"

# Install dependencies
section "Installing dependencies"
pip install -r requirements.txt || error_exit "Failed to install dependencies"

# Run tests for individual modules
section "Running module tests"

# Test config module
echo -e "\n${GREEN}Testing config module...${NC}"
python3 -c "from config import stock_config; print('Config module test passed')" || \
    error_exit "Config module test failed"

# Test data module
echo -e "\n${GREEN}Testing data module...${NC}"
python3 -c "from data import fetcher; print('Data module test passed')" || \
    error_exit "Data module test failed"

# Test analysis modules
echo -e "\n${GREEN}Testing analysis modules...${NC}"
python3 -c "from analysis import technical, fundamental; print('Analysis modules test passed')" || \
    error_exit "Analysis modules test failed"

# Run integration tests
section "Running integration tests"
if [ -f "test_integration.py" ]; then
    python3 -m pytest test_integration.py -v || \
        error_exit "Integration tests failed"
else
    echo -e "${YELLOW}Warning: test_integration.py not found, skipping integration tests${NC}"
fi

# Run the main application
section "Running main application"
echo -e "${GREEN}Starting market research analysis...${NC}"
python3 main.py --group indian_it || \
    error_exit "Failed to run main application"

echo -e "\n${GREEN}All tasks completed successfully!${NC}"

# Deactivate virtual environment when done
deactivate
