#!/bin/bash

# Setup script for enabling remote path access with Docker

echo "Setting up system for remote path access with Docker..."
echo "======================================================"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo privileges. Please run: sudo $0"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"

echo "Setting up for user: $ACTUAL_USER"

# 1. Install sshfs if not present
echo ""
echo "1. Checking sshfs installation..."
if ! command -v sshfs &> /dev/null; then
    echo "Installing sshfs..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y sshfs
    elif command -v yum &> /dev/null; then
        yum install -y fuse-sshfs
    else
        echo "Error: Cannot determine package manager. Please install sshfs manually."
        exit 1
    fi
else
    echo "✓ sshfs is already installed"
fi

# 2. Enable user_allow_other in fuse.conf
echo ""
echo "2. Configuring FUSE to allow other users..."
if [ -f /etc/fuse.conf ]; then
    # Backup original
    cp /etc/fuse.conf /etc/fuse.conf.backup.$(date +%Y%m%d_%H%M%S)
    
    # Enable user_allow_other
    if grep -q "^#user_allow_other" /etc/fuse.conf; then
        sed -i 's/^#user_allow_other/user_allow_other/' /etc/fuse.conf
        echo "✓ Enabled user_allow_other in /etc/fuse.conf"
    elif grep -q "^user_allow_other" /etc/fuse.conf; then
        echo "✓ user_allow_other is already enabled"
    else
        echo "user_allow_other" >> /etc/fuse.conf
        echo "✓ Added user_allow_other to /etc/fuse.conf"
    fi
else
    echo "user_allow_other" > /etc/fuse.conf
    echo "✓ Created /etc/fuse.conf with user_allow_other"
fi

# 3. Add user to fuse group
echo ""
echo "3. Adding user to fuse group..."
if getent group fuse > /dev/null 2>&1; then
    usermod -a -G fuse $ACTUAL_USER
    echo "✓ Added $ACTUAL_USER to fuse group"
else
    groupadd fuse
    usermod -a -G fuse $ACTUAL_USER
    echo "✓ Created fuse group and added $ACTUAL_USER"
fi

# 4. Set up SSH key if requested
echo ""
echo "4. SSH Key Setup"
read -p "Do you want to set up SSH key authentication for bigdata.nsu.ru? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Switch to actual user for SSH operations
    sudo -u $ACTUAL_USER bash <<EOF
        # Check if SSH key exists
        if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then
            echo "Generating SSH key..."
            ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
        fi
        
        echo ""
        echo "Copying SSH key to bigdata.nsu.ru..."
        echo "You'll be prompted for your password on the remote server:"
        ssh-copy-id roppert@bigdata.nsu.ru
EOF
else
    echo "Skipping SSH key setup"
fi

# 5. Test permissions
echo ""
echo "5. Testing configuration..."

# Create test directory
TEST_DIR="/tmp/fuse_test_$$"
mkdir -p $TEST_DIR

# Try to mount with allow_other as the actual user
sudo -u $ACTUAL_USER bash -c "
    echo 'Testing FUSE mount with allow_other...'
    echo 'test' > $TEST_DIR/testfile
    mkdir -p $TEST_DIR/mount
    
    # Try to mount the directory to itself using bindfs or a simple fuse filesystem
    if command -v bindfs &> /dev/null; then
        bindfs -o allow_other $TEST_DIR $TEST_DIR/mount 2>/dev/null
        if [ \$? -eq 0 ]; then
            echo '✓ FUSE allow_other is working correctly'
            fusermount -u $TEST_DIR/mount 2>/dev/null
        else
            echo '✗ FUSE allow_other test failed - you may need to logout/login for group changes'
        fi
    else
        echo '⚠ bindfs not installed, skipping allow_other test'
    fi
"

# Cleanup
rm -rf $TEST_DIR

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "IMPORTANT: You need to logout and login again for group changes to take effect!"
echo ""
echo "After logging back in, you can use:"
echo "  ./launch_docker.sh -r    # To use remote paths"
echo ""
echo "If you still encounter issues:"
echo "1. Make sure you've logged out and back in"
echo "2. Verify you're in the fuse group: groups | grep fuse"
echo "3. Check that /etc/fuse.conf contains: user_allow_other"
echo "4. Ensure SSH key authentication works: ssh roppert@bigdata.nsu.ru"
echo "=============================================="