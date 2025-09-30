# Virtual Environment Setup for Sleep Detection Module

This module uses a virtual environment to isolate dependencies and ensure consistent behavior across different systems.

## Quick Start

### Option 1: Automatic Setup
```bash
python setup_and_run.py
```
The setup script will automatically create and configure the virtual environment.

### Option 2: Manual Setup

#### For macOS/Linux:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Or use the provided script
./activate_venv.sh

# Install requirements
pip install -r requirements.txt
```

#### For Windows:
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Command Prompt)
.venv\Scripts\activate.bat

# Or activate (PowerShell)
.venv\Scripts\Activate.ps1

# Or use the provided script
activate_venv.bat

# Install requirements
pip install -r requirements.txt
```

## Running the System

Once the virtual environment is activated, you can run any of the scripts directly:

```bash
# Web interface
python sleep_web_ui.py

# Command line system
python sleep_detection_system.py

# Visual demo
python visual_demo.py

# System validation
python validate_system.py

# Manual tests
python manual_test_scenarios.py
```

## Virtual Environment Benefits

- **Isolation**: Dependencies don't conflict with system packages
- **Reproducibility**: Consistent environment across different machines
- **Clean Installation**: Easy to remove by deleting the `.venv` folder
- **Version Control**: Specific package versions for stability

## Deactivating

To deactivate the virtual environment:
```bash
deactivate
```

## Troubleshooting

### Virtual Environment Not Found
If you get errors about missing virtual environment:
```bash
python setup_and_run.py
# Select option 7 to setup environment
```

### Permission Issues (macOS/Linux)
Make sure the activation script is executable:
```bash
chmod +x activate_venv.sh
```

### Python Version Issues
Ensure you're using Python 3.8 or higher:
```bash
python --version
# or
python3 --version
```

## Directory Structure

```
sleep_detection_module/
├── .venv/                 # Virtual environment (created automatically)
├── activate_venv.sh       # Activation script (macOS/Linux)
├── activate_venv.bat      # Activation script (Windows)
├── requirements.txt       # Python dependencies
├── setup_and_run.py      # Main setup and run script
└── ...                   # Other module files
```