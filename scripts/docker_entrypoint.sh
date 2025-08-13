#!/bin/bash

echo "🚀 Starting MRI AI Service Container..."
echo "======================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to safely source a file if it exists
safe_source() {
    if [ -f "$1" ]; then
        echo "📁 Sourcing: $1"
        source "$1"
        return 0
    else
        echo "⚠️  Not found: $1"
        return 1
    fi
}

# Try to source FSL configuration
echo "🔧 Setting up FSL environment..."
FSL_SOURCED=false

# List of possible FSL config locations
FSL_CONFIG_PATHS=(
    "${FSLDIR}/etc/fslconf/fsl.sh"
    "${FSLDIR}/fslpython/envs/fslpython/etc/fslconf/fsl.sh"
    "/usr/local/fsl/etc/fslconf/fsl.sh"
    "/usr/local/fsl/fslpython/envs/fslpython/etc/fslconf/fsl.sh"
    "/opt/fsl/etc/fslconf/fsl.sh"
    "/usr/share/fsl/etc/fslconf/fsl.sh"
)

for fsl_path in "${FSL_CONFIG_PATHS[@]}"; do
    if safe_source "$fsl_path"; then
        FSL_SOURCED=true
        echo "✅ FSL configuration loaded from: $fsl_path"
        break
    fi
done

if [ "$FSL_SOURCED" = false ]; then
    echo "⚠️  No FSL configuration file found. Using environment variables only."
    echo "   FSLDIR: ${FSLDIR:-'Not set'}"
    if [ -n "$FSLDIR" ]; then
        echo "   FSL directory exists: $([ -d "$FSLDIR" ] && echo 'Yes' || echo 'No')"
        echo "   FSL bin directory: $([ -d "$FSLDIR/bin" ] && echo 'Yes' || echo 'No')"
    fi
fi

# Verify FSL commands are available
echo "🔍 Verifying FSL installation..."
if command_exists bet; then
    echo "✅ FSL bet command available"
    echo "   FSL version: $(bet 2>&1 | head -1 | grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' || echo 'Unknown')"
else
    echo "❌ FSL bet command not found in PATH"
    echo "   Current PATH: $PATH"
    if [ -n "$FSLDIR" ] && [ -d "$FSLDIR/bin" ]; then
        echo "   Adding FSLDIR/bin to PATH..."
        export PATH="$FSLDIR/bin:$PATH"
        if command_exists bet; then
            echo "✅ FSL commands now available"
        else
            echo "❌ FSL commands still not available"
        fi
    fi
fi

# Try to source FreeSurfer configuration
echo "🔧 Setting up FreeSurfer environment..."
FREESURFER_PATHS=(
    "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
    "/opt/freesurfer/SetUpFreeSurfer.sh"
    "/usr/local/freesurfer/SetUpFreeSurfer.sh"
)

FREESURFER_SOURCED=false
for fs_path in "${FREESURFER_PATHS[@]}"; do
    if safe_source "$fs_path"; then
        FREESURFER_SOURCED=true
        echo "✅ FreeSurfer configuration loaded from: $fs_path"
        break
    fi
done

if [ "$FREESURFER_SOURCED" = false ]; then
    echo "⚠️  FreeSurfer configuration not found"
    echo "   FREESURFER_HOME: ${FREESURFER_HOME:-'Not set'}"
fi

# Verify ANTs
echo "🔍 Verifying ANTs installation..."
if command_exists antsRegistration; then
    echo "✅ ANTs available"
else
    echo "⚠️  ANTs not found in PATH"
    if [ -n "$ANTSPATH" ] && [ -d "$ANTSPATH" ]; then
        echo "   Adding ANTSPATH to PATH..."
        export PATH="$ANTSPATH:$PATH"
        if command_exists antsRegistration; then
            echo "✅ ANTs now available"
        else
            echo "❌ ANTs still not available"
        fi
    fi
fi

# Verify Python and required packages
echo "🔍 Verifying Python environment..."
if command_exists python3; then
    echo "✅ Python3 available: $(python3 --version)"
    
    # Check key packages
    python3 -c "import nibabel; print('✅ nibabel available')" 2>/dev/null || echo "❌ nibabel not available"
    python3 -c "import numpy; print('✅ numpy available')" 2>/dev/null || echo "❌ numpy not available"
    python3 -c "import yaml; print('✅ yaml available')" 2>/dev/null || echo "❌ yaml not available"
else
    echo "❌ Python3 not available"
fi

# Run the container initialization script
echo "🔧 Running container initialization..."
if [ -f "/app/scripts/docker_init.py" ]; then
    python3 /app/scripts/docker_init.py
    if [ $? -eq 0 ]; then
        echo "✅ Container initialization completed"
    else
        echo "❌ Container initialization failed"
        exit 1
    fi
else
    echo "⚠️  Container init script not found: /app/scripts/docker_init.py"
    echo "   Creating minimal directory structure..."
    mkdir -p /app/data/{input,output} /app/logs
fi

# Final environment summary
echo ""
echo "📋 Environment Summary:"
echo "   Working directory: $(pwd)"
echo "   FSL available: $(command_exists bet && echo 'Yes' || echo 'No')"
echo "   ANTs available: $(command_exists antsRegistration && echo 'Yes' || echo 'No')"
echo "   FreeSurfer available: $(command_exists mri_convert && echo 'Yes' || echo 'No')"
echo "   Python available: $(command_exists python3 && echo 'Yes' || echo 'No')"

# Start the main application
echo ""
echo "🌐 Starting MRI AI Service Web Application..."
echo "============================================="

# Change to app directory
cd /app || {
    echo "❌ Cannot change to /app directory"
    exit 1
}

# Start the Flask application
if [ -f "webapp/app.py" ]; then
    exec python3 webapp/app.py
else
    echo "❌ webapp/app.py not found!"
    echo "Available files in /app:"
    ls -la
    exit 1
fi