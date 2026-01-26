#!/bin/bash
# Setup CARLA 0.9.15 locally
# This script downloads and sets up CARLA in your local directory

# Set local CARLA installation path (adjust as needed)
CARLA_INSTALL_DIR="${HOME}/software/carla0915"

echo "Setting up CARLA 0.9.15 in ${CARLA_INSTALL_DIR}"

# Create directories
mkdir -p "${CARLA_INSTALL_DIR}"
cd "${CARLA_INSTALL_DIR}"

# Download CARLA 0.9.15
echo "Downloading CARLA 0.9.15..."
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# Extract CARLA
echo "Extracting CARLA..."
tar -xvf CARLA_0.9.15.tar.gz

# Clean up tar file
rm CARLA_0.9.15.tar.gz

# Download and import additional maps
echo "Downloading additional maps..."
cd Import
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz

# Import assets
cd ..
echo "Importing assets (this may take a while)..."
bash ImportAssets.sh

echo ""
echo "✅ CARLA setup completed!"
echo "CARLA_ROOT: ${CARLA_INSTALL_DIR}"
echo ""
echo "Add this to your environment:"
echo "export CARLA_ROOT=${CARLA_INSTALL_DIR}"
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla"
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"

