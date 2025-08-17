#!/bin/bash
# Setup script for rag-advanced project

set -e  # Exit on error

echo "🚀 Setting up rag-advanced project..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    pip install uv
fi

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Activate based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Unix-like
    source .venv/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -e ".[dev]"

# Setup environment
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file from example..."
    cp .env.example .env
    echo "ℹ️ Please edit the .env file with your configuration"
fi

echo "✅ Setup complete!"
echo "   To activate the virtual environment, run:"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi

echo "\n🚀 Start the app with: streamlit run app/ui/streamlit_app.py"
