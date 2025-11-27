#!/bin/bash
# Script to build and publish AgentFlow to PyPI

# 1. Install build tools
pip install --upgrade build twine

# 2. Build the package
python3 -m build

# 3. Check the package
twine check dist/*

# 4. Upload to PyPI (will ask for username/password or token)
# Username: __token__
# Password: <your-pypi-token>
echo "Ready to upload to PyPI."
echo "Run: twine upload dist/*"
