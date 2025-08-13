#!/usr/bin/env python3
"""
Docker initialization script that handles dynamic path configuration
"""
import os
import yaml
import sys
from pathlib import Path

def setup_config():
    """Setup configuration from environment variables or mounted config"""
    
    # Check if paths are provided via environment variables
    env_input = os.environ.get('MRI_INPUT_PATH')
    env_output = os.environ.get('MRI_OUTPUT_PATH') 
    env_template = os.environ.get('MRI_TEMPLATE_PATH')
    
    if env_input and env_output:
        print(f"🔧 Using environment variable paths:")
        print(f"   Input: {env_input}")
        print(f"   Output: {env_output}")
        print(f"   Template: {env_template or '/app/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii'}")
        
        # Update config with environment paths
        config_path = '/app/config/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths in config to point to mounted directories
            if 'paths' not in config:
                config['paths'] = {}
                
            config['paths']['raw_input_dir'] = env_input
            config['paths']['output_base_dir'] = env_output
            config['paths']['template_path'] = env_template or '/app/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii'
            
            # Ensure subdirectories are relative to output
            if 'subdirs' not in config['paths']:
                config['paths']['subdirs'] = {}
            
            # Write updated config back
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print("✅ Configuration updated with environment variables")
        else:
            print("⚠️  Config file not found, creating minimal config")
            # Create minimal config
            config = {
                'paths': {
                    'raw_input_dir': env_input,
                    'output_base_dir': env_output,
                    'template_path': env_template or '/app/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii',
                    'subdirs': {
                        'bids_dicom': 'bids_data_dicom',
                        'dicom_checks': 'dciodvfy_reports',
                        'dicom_meta': 'dicom_metadata',
                        'bids_nifti': 'bids_data_nifti',
                        'validation_reports': 'validation_results',
                        'fast_qc_reports': 'bids_quality_metrics',
                        'mriqc_output': 'mriqc_output',
                        'mriqc_interpret': 'mriqc_interpretation',
                        'transforms': 'transformations',
                        'preprocessed': 'preprocessed_data',
                        'segmentation_masks': 'segmentation_masks',
                        'logs': 'logs'
                    }
                }
            }
            
            os.makedirs('/app/config', exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print("✅ Minimal configuration created")
    else:
        print("📁 Using mounted configuration file")
        
        # Verify the mounted config exists
        config_path = '/app/config/config.yaml'
        if not os.path.exists(config_path):
            print("❌ No configuration found! Either:")
            print("   1. Mount a config file to /app/config/config.yaml, OR")
            print("   2. Provide MRI_INPUT_PATH and MRI_OUTPUT_PATH environment variables")
            sys.exit(1)
            
        # Load and display current config paths
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            paths = config.get('paths', {})
            print(f"   Input: {paths.get('raw_input_dir', 'Not specified')}")
            print(f"   Output: {paths.get('output_base_dir', 'Not specified')}")
            print(f"   Template: {paths.get('template_path', 'Not specified')}")
        except Exception as e:
            print(f"⚠️  Could not read config file: {e}")
        
    # Ensure required directories exist inside container
    required_dirs = [
        '/app/data/input',
        '/app/data/output', 
        '/app/logs'
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    print("✅ Required directories ensured")

def verify_mounts():
    """Verify that expected mount points exist and are accessible"""
    mount_checks = [
        ('/app/data/input', 'Input data directory'),
        ('/app/data/output', 'Output data directory'),
        ('/app/config/config.yaml', 'Configuration file'),
        ('/app/mni_templates', 'MNI templates directory'),
        ('/app/logs', 'Logs directory')
    ]
    
    print("🔍 Verifying mounts:")
    all_good = True
    
    for path, description in mount_checks:
        if os.path.exists(path):
            if os.path.isdir(path):
                files_count = len(os.listdir(path)) if os.access(path, os.R_OK) else "Access denied"
                print(f"   ✅ {description}: {path} ({files_count} items)")
            else:
                print(f"   ✅ {description}: {path} (file)")
        else:
            print(f"   ❌ {description}: {path} (missing)")
            all_good = False
    
    if not all_good:
        print("⚠️  Some expected mounts are missing. The service may not work correctly.")
    else:
        print("✅ All mounts verified")
    
    return all_good

if __name__ == '__main__':
    print("🚀 Docker container initialization starting...")
    print("=" * 50)
    
    # Setup configuration
    setup_config()
    
    # Verify mounts
    verify_mounts()
    
    print("=" * 50)
    print("✅ Docker initialization complete!")
    print("🌐 Starting MRI AI Service...")
    print("")