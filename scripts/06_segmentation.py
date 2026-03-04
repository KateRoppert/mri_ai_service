# -*- coding: utf-8 -*-
"""
Batch MRI segmentation script for BIDS-formatted datasets.

This script processes multiple subjects/sessions from a BIDS directory structure,
sending NIfTI files to a segmentation server and organizing outputs in BIDS format.

Key components:
- Config: Manages YAML configuration including modality mappings
- SubjectSession: Data structure for a single subject/session
- BIDSScanner: Discovers and validates subjects in BIDS directories
- SegmentationInput: Validates complete modality sets
- SegmentationClient: Handles HTTP communication with segmentation server
- SegmentationRunner: Orchestrates batch processing workflow
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import yaml
import requests
from dataclasses import dataclass
from contextlib import ExitStack
from collections import defaultdict
import subprocess
import atexit
import socket
import getpass
import shutil
import aiohttp
import asyncio
import time
from datetime import datetime 
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from pipeline_validator import InputOutputValidator

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None, console_level: str = "INFO"):
    """Configures the main application logger."""
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to set up file logger at {log_file}: {e}")

# --- Server Availability Check ---

def check_server_availability(server_url: str, timeout: int = 5) -> bool:
    """
    Checks if the segmentation server is accessible.
    
    Args:
        server_url: URL of the server to check
        timeout: Request timeout in seconds
    
    Returns:
        True if server responds, False otherwise
    """
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=timeout)
        if response.status_code == 200:
            logger.debug(f"Server is accessible at {server_url}")
            return True
        else:
            logger.debug(f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"Server not accessible at {server_url}: {e}")
        return False

def create_ssh_tunnel(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    remote_port: int,
    local_port: int,
    ssh_key: Optional[Path] = None,
    password: Optional[str] = None
) -> subprocess.Popen:
    """
    Creates an SSH tunnel to access remote server.
    
    Args:
        ssh_host: SSH server hostname/IP
        ssh_port: SSH server port
        ssh_user: SSH username
        remote_port: Remote port where service runs (e.g., 5000)
        local_port: Local port for tunnel (e.g., 15000)
        ssh_key: Optional path to SSH key file
        password: Optional password (will prompt if not provided and no key)
    
    Returns:
        SSH process object
    """
    logger.info("=" * 60)
    logger.info("CREATING SSH TUNNEL")
    logger.info("=" * 60)
    logger.info(f"SSH: {ssh_user}@{ssh_host}:{ssh_port}")
    logger.info(f"Tunnel: localhost:{local_port} -> remote:{remote_port}")
    
    # Check if local port is available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', local_port))
    sock.close()
    
    if result == 0:
        raise RuntimeError(
            f"Local port {local_port} is already in use. "
            "Please choose a different local_tunnel_port or close the conflicting process."
        )
    
    # Build base SSH command
    ssh_cmd = [
        "ssh",
        "-N",  # Don't execute remote command
        "-L", f"{local_port}:localhost:{remote_port}",  # Local forward
        "-p", str(ssh_port),
        f"{ssh_user}@{ssh_host}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=3"
    ]
    
    # Add SSH key if specified
    if ssh_key:
        ssh_cmd.extend(["-i", str(ssh_key)])
        logger.info(f"Using SSH key: {ssh_key}")
    
    # Determine authentication method
    use_sshpass = False
    if not ssh_key:
        # No key specified - need password
        if password is None:
            # Prompt for password
            password = getpass.getpass(f"Enter SSH password for {ssh_user}@{ssh_host}: ")
        
        # Check if sshpass is available
        if shutil.which("sshpass"):
            use_sshpass = True
            logger.info("Using sshpass for password authentication")
        else:
            logger.warning(
                "sshpass not found. SSH will prompt for password in terminal.\n"
                "For better experience, install sshpass:\n"
                "  Ubuntu/Debian: sudo apt install sshpass\n"
                "  macOS: brew install hudochenkov/sshpass/sshpass"
            )
    
    # Build final command with sshpass if needed
    if use_sshpass and password:
        final_cmd = ["sshpass", "-p", password] + ssh_cmd
        # Don't log password!
        log_cmd = ["sshpass", "-p", "****"] + ssh_cmd
    else:
        final_cmd = ssh_cmd
        log_cmd = ssh_cmd
    
    logger.info(f"Executing: {' '.join(log_cmd[:10])}...")
    
    try:
        # Start SSH tunnel
        tunnel_process = subprocess.Popen(
            final_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if not use_sshpass else None
        )
        
        # Wait for tunnel to establish
        import time
        logger.info("Waiting for tunnel to establish...")
        time.sleep(3)
        
        # Check if tunnel is alive
        if tunnel_process.poll() is not None:
            stdout, stderr = tunnel_process.communicate()
            error_msg = stderr.decode('utf-8', errors='ignore')
            
            # Common error messages
            if "Permission denied" in error_msg:
                raise RuntimeError(
                    "SSH authentication failed. Check your password/key.\n"
                    f"Details: {error_msg[:200]}"
                )
            elif "Connection refused" in error_msg:
                raise RuntimeError(
                    f"Cannot connect to {ssh_host}:{ssh_port}. "
                    "Check if the server is accessible.\n"
                    f"Details: {error_msg[:200]}"
                )
            else:
                raise RuntimeError(
                    f"SSH tunnel failed to start.\n"
                    f"stdout: {stdout.decode('utf-8', errors='ignore')[:200]}\n"
                    f"stderr: {error_msg[:200]}"
                )
        
        # Verify tunnel works
        logger.info(f"Verifying tunnel on port {local_port}...")
        for attempt in range(5):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            try:
                result = sock.connect_ex(('localhost', local_port))
                sock.close()
                
                if result == 0:
                    logger.info("✓ SSH tunnel established successfully")
                    return tunnel_process
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        
        # If we got here, tunnel didn't work
        tunnel_process.terminate()
        raise RuntimeError(
            f"SSH tunnel created but port {local_port} is not accessible after 5 attempts"
        )
        
    except FileNotFoundError as e:
        if "sshpass" in str(e):
            raise RuntimeError(
                "sshpass not found. Please install it:\n"
                "  Ubuntu/Debian: sudo apt install sshpass\n"
                "  macOS: brew install hudochenkov/sshpass/sshpass\n"
                "Or use SSH key authentication instead."
            )
        else:
            raise RuntimeError(
                "SSH client not found. Please install OpenSSH:\n"
                "  Ubuntu/Debian: sudo apt install openssh-client\n"
                "  macOS: built-in\n"
                "  Windows: install OpenSSH or use WSL"
            )


def close_ssh_tunnel(process: subprocess.Popen):
    """Closes SSH tunnel gracefully."""
    if process and process.poll() is None:
        logger.info("Closing SSH tunnel...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("✓ SSH tunnel closed")
        except subprocess.TimeoutExpired:
            logger.warning("SSH tunnel did not close gracefully, forcing...")
            process.kill()
            process.wait()

# --- Core Classes ---

class Config:
    """Loads and provides access to configuration parameters."""
    
    def __init__(self, config_path: Path):
        self._config_path = config_path
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._data = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def get_active_profile(self) -> dict:
        """
        Retrieves the active profile configuration.
        
        Returns:
            Dictionary with profile settings
        
        Raises:
            ValueError: If active profile not found or invalid
        """
        active = self._data.get('segmentation', {}).get('active_profile')
        
        if not active:
            # Fallback to legacy format
            logger.warning("No active_profile found in config, using legacy format")
            legacy_url = self._data.get('executables', {}).get('aiaa_server_url')
            if not legacy_url:
                raise ValueError(
                    "Neither 'segmentation.active_profile' nor 'executables.aiaa_server_url' "
                    "found in config. Please update your configuration file."
                )
            
            return {
                'connection_type': 'direct',
                'server_url': legacy_url,
                'timeout': 1200,
                'description': 'Legacy configuration'
            }
        
        profiles = self._data.get('segmentation', {}).get('profiles', {})
        
        if active not in profiles:
            raise ValueError(
                f"Active profile '{active}' not found in config. "
                f"Available profiles: {list(profiles.keys())}"
            )
        
        return profiles[active]

    def get_server_url(self) -> str:
        """Retrieves the server URL from active profile."""
        profile = self.get_active_profile()
        return profile.get('server_url', 'http://localhost:5000')

    def get_model_name(self) -> str:
        """Retrieves the segmentation model name from the config."""
        model = self._data.get('segmentation', {}).get('model_name')
        if not model:
            raise ValueError("'segmentation.model_name' not found in config.")
        return model

    def get_modality_map(self) -> dict[str, str]:
        """
        Retrieves the modality input mapping from the config.
        
        Returns:
            Dictionary mapping server keys to BIDS suffixes.
            Example: {"t1": "T1w", "t1c": "ce-gd_T1w", "t2": "T2w", "flair": "FLAIR"}
        """
        modality_map = self._data.get('segmentation', {}).get('modality_input_map')
        if not modality_map:
            raise ValueError("'segmentation.modality_input_map' not found in config.")
        
        # Validate that all required keys are present
        required_keys = {"t1", "t1c", "t2", "t2fl"}
        if not required_keys.issubset(modality_map.keys()):
            missing = required_keys - modality_map.keys()
            raise ValueError(f"Missing required modality keys in config: {missing}")
        
        return modality_map
    
    def get_connection_type(self) -> str:
        """Returns connection type: 'direct' or 'ssh_tunnel'"""
        profile = self.get_active_profile()
        return profile.get('connection_type', 'direct')
    
    def get_ssh_config(self) -> Optional[dict]:
        """Returns full SSH configuration including ports."""
        profile = self.get_active_profile()
        
        if not profile or profile.get('connection_type') != 'ssh_tunnel':
            return None
        
        ssh_block = profile.get('ssh', {})
        key_file = ssh_block.get('key_file')
        if key_file:
            key_file = Path(key_file).expanduser()
        
        return {
            'host': ssh_block.get('host'),
            'port': ssh_block.get('port'),
            'user': ssh_block.get('user'),
            'key_file': key_file,
            'remote_port': profile.get('remote_port'),  # ← Из профиля!
            'local_port': profile.get('local_tunnel_port')  # ← Из профиля!
        }
    
    def get_profile_description(self) -> str:
        """Returns human-readable description of active profile."""
        profile = self.get_active_profile()
        return profile.get('description', 'No description')


@dataclass
class SubjectSession:
    """
    Data structure for a single subject/session with all required information.
    """
    subject_id: str
    session_id: Optional[str]
    modality_files: dict[str, Path]  # Keys: t1, t1c, t2, flair
    output_mask_path: Path
    
    def get_identifier(self) -> str:
        """Returns a human-readable identifier for logging."""
        if self.session_id:
            return f"{self.subject_id}_{self.session_id}"
        return self.subject_id
    
    def has_all_modalities(self) -> bool:
        """Checks if all required modalities are present."""
        required = {"t1", "t1c", "t2", "t2fl"}
        return required.issubset(self.modality_files.keys())


class BIDSScanner:
    """
    Scans a BIDS directory structure and discovers subject/session combinations.
    """
    def __init__(self, input_dir: Path, output_dir: Path, modality_map: dict[str, str]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.modality_map = modality_map
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    def scan(self, max_subjects: Optional[int] = None) -> list[SubjectSession]:
        """
        Scans the input directory and returns a list of SubjectSession objects.
        
        Args:
            max_subjects: Maximum number of subjects to process (for testing)
        
        Returns:
            List of SubjectSession objects with complete metadata
        """
        sessions = []
        subject_dirs = sorted([d for d in self.input_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('sub-')])
        
        logger.info(f"Found {len(subject_dirs)} subject directories in {self.input_dir}")
        
        for subject_dir in subject_dirs:
            if max_subjects and len(sessions) >= max_subjects:
                logger.info(f"Reached max_subjects limit ({max_subjects}). Stopping scan.")
                break
            
            subject_id = subject_dir.name
            sessions.extend(self._scan_subject(subject_dir, subject_id))
        
        logger.info(f"Total sessions discovered: {len(sessions)}")
        return sessions
    
    def _scan_subject(self, subject_dir: Path, subject_id: str) -> list[SubjectSession]:
        """Scans a single subject directory for sessions."""
        sessions = []
        
        # Check for session subdirectories
        session_dirs = sorted([d for d in subject_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('ses-')])
        
        if session_dirs:
            # Process each session
            for session_dir in session_dirs:
                session_id = session_dir.name
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    logger.warning(f"No 'anat' directory found for {subject_id}/{session_id}")
                    continue
                
                session = self._create_session(subject_id, session_id, anat_dir)
                if session:
                    sessions.append(session)
        else:
            # No sessions - check for direct anat directory
            anat_dir = subject_dir / "anat"
            if anat_dir.exists():
                session = self._create_session(subject_id, None, anat_dir)
                if session:
                    sessions.append(session)
            else:
                logger.warning(f"No 'anat' directory found for {subject_id}")
        
        return sessions
    
    def _create_session(
        self, 
        subject_id: str, 
        session_id: Optional[str], 
        anat_dir: Path
    ) -> Optional[SubjectSession]:
        """
        Creates a SubjectSession object by finding all modality files.
        
        Returns None if not all modalities are found.
        """
        identifier = f"{subject_id}_{session_id}" if session_id else subject_id
        
        # Find files for each modality
        modality_files = {}
        for server_key, bids_suffix in self.modality_map.items():
            # Pattern: sub-XXX_ses-YYY_<bids_suffix>.nii.gz or sub-XXX_<bids_suffix>.nii.gz
            if session_id:
                pattern = f"{subject_id}_{session_id}_{bids_suffix}.nii.gz"
            else:
                pattern = f"{subject_id}_{bids_suffix}.nii.gz"
            
            matches = list(anat_dir.glob(pattern))
            
            if matches:
                modality_files[server_key] = matches[0]
            else:
                logger.debug(f"{identifier}: Missing modality '{server_key}' (pattern: {pattern})")
        
        # Check if we have all required modalities
        required = {"t1", "t1c", "t2", "t2fl"}
        if not required.issubset(modality_files.keys()):
            missing = required - modality_files.keys()
            logger.warning(f"⚠️  Skipping {identifier}: Missing modalities {missing}")
            return None
        
        # Determine output path (mirrors BIDS structure)
        if session_id:
            output_anat_dir = self.output_dir / subject_id / session_id / "anat"
            base_filename = f"{subject_id}_{session_id}"
        else:
            output_anat_dir = self.output_dir / subject_id / "anat"
            base_filename = subject_id
        
        # Pick one modality file to derive the output name from (e.g., T1w)
        reference_file = modality_files["t1"]
        # Extract the full suffix after subject/session ID
        # e.g., sub-001_ses-01_T1w.nii.gz -> T1w.nii.gz
        ref_name = reference_file.name
        suffix_start = ref_name.find(self.modality_map["t1"])
        if suffix_start != -1:
            original_suffix = ref_name[suffix_start:]
            # Insert _segmask before .nii.gz
            output_name = original_suffix.replace('.nii.gz', '_segmask.nii.gz')
        else:
            # Fallback if pattern not found
            output_name = f"{self.modality_map['t1']}_segmask.nii.gz"
        
        output_mask_path = output_anat_dir / f"{base_filename}_{output_name}"
        
        logger.debug(f"✓ {identifier}: All modalities found")
        
        return SubjectSession(
            subject_id=subject_id,
            session_id=session_id,
            modality_files=modality_files,
            output_mask_path=output_mask_path
        )
    
    def filter_existing(
        self, 
        sessions: list[SubjectSession]
    ) -> tuple[list[SubjectSession], int]:
        """
        Filter out sessions that already have output mask files.
        
        Args:
            sessions: List of SubjectSession objects to filter
        
        Returns:
            Tuple of (filtered_sessions, skipped_count)
        """
        filtered = []
        skipped = 0
        
        for session in sessions:
            if session.output_mask_path.exists():
                identifier = session.get_identifier()
                logger.debug(f"⊙ Skipping {identifier}: Output mask already exists at {session.output_mask_path}")
                skipped += 1
            else:
                filtered.append(session)
        
        if skipped > 0:
            logger.info(f"⊙ Skipped {skipped} sessions with existing output masks")
        
        return filtered, skipped


@dataclass
class SegmentationInput:
    """
    Validates and prepares input files for segmentation.
    No fallback logic - all modalities must be present.
    """
    t1: Path
    t1c: Path
    t2: Path
    flair: Path
    
    def validate(self) -> bool:
        """Validates that all files exist."""
        all_files = [self.t1, self.t1c, self.t2, self.flair]
        
        for file_path in all_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        return True
    
    def prepare_for_server(self) -> dict[str, Path]:
        """Returns a dictionary ready for server submission."""
        if not self.validate():
            raise FileNotFoundError("Cannot prepare files for server: validation failed")
        
        files = {
            "t1": self.t1,
            "t1c": self.t1c,
            "t2": self.t2,
            "t2fl": self.flair,
        }
        
        logger.debug("Files prepared for server:")
        for mod, path in files.items():
            logger.debug(f"  - {mod.upper()}: {path.name}")
        
        return files


class SegmentationClient:
    """Handles all HTTP communication with the segmentation server."""
    def __init__(self, server_url: str, timeout: int = 1200):
        self.server_url = server_url
        self.timeout = timeout

    def segment(
        self,
        files_to_send: dict[str, Path],
        model_name: str,
        client_id: str,
        output_path: Path
    ) -> bool:
        """
        Sends files to the server for segmentation and saves the resulting mask.
        
        Returns:
            True on success, False on failure.
        """
        # Use query parameters for net and client_id (more reliable with multipart)
        inference_url = f"{self.server_url}/v1/inference?net={model_name}&client_id={client_id}"
        logger.debug(f"Sending request to: {inference_url}")

        try:
            # ExitStack cleanly manages opening multiple files
            with ExitStack() as stack:
                # Server expects field names: file_t1, file_t1c, file_t2, file_t2fl
                files_multipart = {
                    'file_t1': (
                        files_to_send["t1"].name, 
                        stack.enter_context(open(files_to_send["t1"], 'rb'))
                    ),
                    'file_t1c': (
                        files_to_send["t1c"].name,
                        stack.enter_context(open(files_to_send["t1c"], 'rb'))
                    ),
                    'file_t2': (
                        files_to_send["t2"].name,
                        stack.enter_context(open(files_to_send["t2"], 'rb'))
                    ),
                    'file_t2fl': (
                        files_to_send["t2fl"].name,
                        stack.enter_context(open(files_to_send["t2fl"], 'rb'))
                    ),
                }

                logger.info("  → Uploading files to server...")
                response = requests.post(
                    inference_url,
                    files=files_multipart,  # Note: no data parameter
                    timeout=self.timeout
                )
                response.raise_for_status()

                logger.info("  → Saving segmentation mask...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f_mask:
                    f_mask.write(response.content)
                
                logger.info(f"  ✓ Success: {output_path}")
                return True

        except requests.exceptions.Timeout:
            logger.error(f"  ✗ Request timed out after {self.timeout} seconds")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"  ✗ Connection error: {e}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"  ✗ HTTP Error {e.response.status_code}: {e.response.reason}")
            if hasattr(e.response, 'text'):
                logger.error(f"  Server response: {e.response.text[:500]}")
            return False
        except Exception as e:
            logger.exception(f"  ✗ Unexpected error during segmentation: {e}")
            return False

class AsyncSegmentationClient:
    """Асинхронный клиент для параллельной отправки запросов на сегментацию"""
    
    def __init__(self, server_url: str, timeout: int = 1800):
        self.server_url = server_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def segment_async(
        self,
        files_to_send: dict,
        model_name: str,
        client_id: str,
        output_path: Path
    ) -> bool:
        """Асинхронная отправка запроса на сегментацию"""
        try:
            # Формируем multipart данные
            data = aiohttp.FormData()
            data.add_field('model', model_name)
            data.add_field('client_id', client_id)
            
            # Открываем файлы
            files_handles = []
            for key, filepath in files_to_send.items():
                f = open(filepath, 'rb')
                files_handles.append(f)
                data.add_field(f'file_{key}', f, filename=filepath.name)
            
            # Отправляем запрос
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.server_url}/v1/inference_async",
                    data=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Server error: {error_text}")
                        return False
                    
                    result = await response.json()
                    task_id = result.get('task_id')
                    
                    if not task_id:
                        logger.error("No task_id in response")
                        return False
                    
                    # Закрываем файлы
                    for f in files_handles:
                        f.close()
                    
                    # Ожидаем завершения
                    success = await self._wait_for_completion(session, task_id, output_path)
                    return success
        
        except Exception as e:
            logger.error(f"Async segmentation error: {e}")
            return False
        
    async def segment_async_with_status(
        self,
        files_to_send: dict,
        model_name: str,
        client_id: str,
        output_path: Path
    ) -> tuple[bool, dict]:
        """
        Асинхронная отправка запроса на сегментацию с возвратом финального статуса.
        
        Returns:
            Tuple of (success: bool, final_task_status: dict)
        """
        try:
            # Формируем multipart данные
            data = aiohttp.FormData()
            data.add_field('model', model_name)
            data.add_field('client_id', client_id)
            
            # Открываем файлы
            files_handles = []
            for key, filepath in files_to_send.items():
                f = open(filepath, 'rb')
                files_handles.append(f)
                data.add_field(f'file_{key}', f, filename=filepath.name)
            
            # Отправляем запрос
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.server_url}/v1/inference_async",
                    data=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Server error: {error_text}")
                        return False, {}
                    
                    result = await response.json()
                    task_id = result.get('task_id')
                    
                    if not task_id:
                        logger.error("No task_id in response")
                        return False, {}
                    
                    # Закрываем файлы
                    for f in files_handles:
                        f.close()
                    
                    # Ожидаем завершения и получаем финальный статус
                    success, final_status = await self._wait_for_completion_with_status(
                        session, task_id, output_path
                    )
                    return success, final_status
        
        except Exception as e:
            logger.error(f"Async segmentation error: {e}")
            return False, {}
    
    async def _wait_for_completion_with_status(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        output_path: Path,
        poll_interval: float = 2.0
    ) -> tuple[bool, dict]:
        """
        Ожидание завершения задачи с опросом статуса и возвратом финального статуса.
        
        Returns:
            Tuple of (success: bool, final_status: dict)
        """
        status_url = f"{self.server_url}/get_status/{task_id}"
        final_status = {}
        
        while True:
            try:
                async with session.get(status_url) as response:
                    if response.status != 200:
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    status_data = await response.json()
                    current_status = status_data.get('status')
                    
                    if current_status == 'completed':
                        # Сохраняем финальный статус
                        final_status = status_data
                        
                        # Скачиваем результат
                        download_url = status_data.get('download_url')
                        if download_url:
                            success = await self._download_result(
                                session,
                                f"{self.server_url}{download_url}",
                                output_path
                            )
                            return success, final_status
                        else:
                            logger.error(f"No download_url for task {task_id}")
                            return False, final_status
                    
                    elif current_status == 'failed':
                        error = status_data.get('error', 'Unknown error')
                        logger.error(f"Task {task_id} failed: {error}")
                        return False, status_data
                    
                    await asyncio.sleep(poll_interval)
            
            except Exception as e:
                logger.error(f"Error polling status: {e}")
                await asyncio.sleep(poll_interval)
    
    async def _wait_for_completion(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        output_path: Path,
        poll_interval: float = 2.0
    ) -> bool:
        """Ожидание завершения задачи с опросом статуса"""
        status_url = f"{self.server_url}/get_status/{task_id}"
        
        while True:
            try:
                async with session.get(status_url) as response:
                    if response.status != 200:
                        await asyncio.sleep(poll_interval)
                        continue
                    
                    status_data = await response.json()
                    current_status = status_data.get('status')
                    
                    if current_status == 'completed':
                        # Скачиваем результат
                        download_url = status_data.get('download_url')
                        if download_url:
                            return await self._download_result(
                                session,
                                f"{self.server_url}{download_url}",
                                output_path
                            )
                        else:
                            logger.error(f"No download_url for task {task_id}")
                            return False
                    
                    elif current_status == 'failed':
                        error = status_data.get('error', 'Unknown error')
                        logger.error(f"Task {task_id} failed: {error}")
                        return False
                    
                    await asyncio.sleep(poll_interval)
            
            except Exception as e:
                logger.error(f"Error polling status: {e}")
                await asyncio.sleep(poll_interval)
    
    async def _download_result(
        self,
        session: aiohttp.ClientSession,
        url: str,
        output_path: Path
    ) -> bool:
        """Скачивание результата"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Download failed: {response.status}")
                    return False
                
                with open(output_path, 'wb') as f:
                    f.write(await response.read())
                
                logger.info(f"  ✓ Saved: {output_path}")
                return True
        
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

@dataclass
class ProcessingStats:
    """Tracks processing statistics."""
    total: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    
    def log_summary(self):
        """Logs a summary of processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total sessions to process: {self.total}")
        logger.info(f"Successfully processed:    {self.successful}")
        logger.info(f"Failed:                    {self.failed}")
        if self.skipped > 0:
            logger.info(f"Skipped (already exists):  {self.skipped}")
        logger.info("=" * 60)


class SegmentationRunner:
    """Orchestrates the batch processing of multiple subjects/sessions."""
    
    def __init__(self, config: 'Config', args):
        """
        Initialize the runner.
        
        Args:
            config: Configuration object
            args: Command-line arguments
        """
        self.config = config
        self.args = args
        self.stats = ProcessingStats()
        
        # Benchmark components
        self.monitor = None
        self.benchmark_logger = None
        self.pipeline_start_time = None
        self.gpu_metrics_collected = []
        
        # Initialize benchmark if enabled
        if args.benchmark:
            # Determine results directory
            if args.results_dir:
                results_dir = args.results_dir
            else:
                results_dir = args.output_dir / "benchmark_results" / "segmentation"
            
            # Initialize logger
            self.benchmark_logger = BenchmarkLogger(results_dir)
            logger.info(f"Benchmark mode enabled. Results will be saved to: {results_dir}")
            
            # Initialize performance monitor
            self.monitor = PerformanceMonitor(enabled=True)

    def run(self) -> bool:
        """Executes the full batch segmentation process."""
        logger.info("=" * 60)
        logger.info("BATCH MRI SEGMENTATION - STARTING")
        logger.info("=" * 60)
        logger.info(f"Input directory:  {self.args.input_dir}")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Config file:      {self.args.config}")
        logger.info(f"Max concurrent:   {self.args.max_concurrent}")

        # Start timing and monitoring for benchmark
        self.pipeline_start_time = time.time()
        if self.monitor:
            self.monitor.start()
            logger.info("Performance monitoring started")
        
        # Get and display profile information
        try:
            active_profile_name = self.config._data.get('segmentation', {}).get('active_profile', 'unknown')
            profile = self.config.get_active_profile()
            logger.info("=" * 60)
            logger.info(f"Active profile:   {active_profile_name}")
            logger.info(f"Description:      {self.config.get_profile_description()}")
            logger.info(f"Connection:       {self.config.get_connection_type()}")
            logger.info(f"Server URL:       {profile.get('server_url')}")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not load profile info: {e}")
        
        ssh_tunnel_process = None
        
        try:
            # Check if SSH tunnel is needed
            connection_type = self.config.get_connection_type()
            
            if connection_type == 'ssh_tunnel':
                ssh_cfg = self.config.get_ssh_config()
                if not ssh_cfg:
                    raise ValueError(
                        "SSH tunnel requested but SSH config not found in profile.\n"
                        "Please check your configuration file."
                    )
                
                # Create SSH tunnel
                ssh_tunnel_process = create_ssh_tunnel(
                    ssh_host=ssh_cfg['host'],
                    ssh_port=ssh_cfg['port'],
                    ssh_user=ssh_cfg['user'],
                    remote_port=ssh_cfg['remote_port'],
                    local_port=ssh_cfg['local_port'],
                    ssh_key=ssh_cfg['key_file'],
                    password=None
                )
                
                atexit.register(lambda: close_ssh_tunnel(ssh_tunnel_process))
            
            server_url = self.config.get_server_url()
            
            # Check server availability
            logger.info("=" * 60)
            logger.info("CHECKING SERVER AVAILABILITY")
            logger.info("=" * 60)
            
            if not check_server_availability(server_url):
                raise ConnectionError(
                    f"Server not accessible at {server_url}.\n"
                    f"Please ensure the server is running on the remote machine."
                )
            
            logger.info(f"✓ Server is accessible at {server_url}")
            
            # Scan for subjects/sessions
            logger.info("=" * 60)
            logger.info("SCANNING INPUT DIRECTORY")
            logger.info("=" * 60)
            
            scanner = BIDSScanner(
                input_dir=self.args.input_dir,
                output_dir=self.args.output_dir,
                modality_map=self.config.get_modality_map()
            )
            
            sessions = scanner.scan(max_subjects=self.args.max_subjects)
            
            if not sessions:
                logger.warning("No valid sessions found to process!")
                return False
            
            # Filter existing sessions if requested
            if self.args.skip_existing:
                logger.info("=" * 60)
                logger.info("FILTERING EXISTING SESSIONS")
                logger.info("=" * 60)
                logger.info(f"Sessions discovered: {len(sessions)}")
                
                sessions, skipped_count = scanner.filter_existing(sessions)
                self.stats.skipped = skipped_count
                
                logger.info(f"Sessions to process: {len(sessions)}")
                logger.info(f"Sessions skipped:    {skipped_count}")
                logger.info("=" * 60)
                
                if not sessions:
                    logger.info("All sessions already processed. Nothing to do.")

                    # Run validation if enabled (even when nothing processed)
                    if self.args.validate:
                        self._run_validation()

                    return True 
            
            self.stats.total = len(sessions)
            
            # Run async processing
            success = asyncio.run(self._process_sessions_async(sessions))

            # Save benchmark metrics if enabled
            if self.args.benchmark and self.monitor and self.benchmark_logger:
                self._save_benchmark_metrics(sessions)

            # Log final statistics
            logger.info("")
            self.stats.log_summary()

            # Run validation if enabled
            if self.args.validate:
                self._run_validation()
            
            return success
        
        except Exception as e:
            logger.critical(f"Critical error in batch processing: {e}")
            logger.exception("Full traceback:")
            return False
        
        finally:
            if ssh_tunnel_process:
                close_ssh_tunnel(ssh_tunnel_process)

    async def _process_sessions_async(self, sessions: List) -> bool:
        """Асинхронная обработка сессий с ограничением параллелизма"""
        logger.info("=" * 60)
        logger.info("PROCESSING SESSIONS")
        logger.info("=" * 60)
        
        client = AsyncSegmentationClient(server_url=self.config.get_server_url())
        model_name = self.config.get_model_name()
        
        # Семафор для ограничения параллелизма
        semaphore = asyncio.Semaphore(self.args.max_concurrent)
        
        async def process_one(idx: int, session) -> bool:
            """Обработка одной сессии с семафором"""
            async with semaphore:
                identifier = session.get_identifier()
                logger.info(f"[{idx}/{self.stats.total}] Processing: {identifier}")
                
                try:
                    seg_input = SegmentationInput(
                        t1=session.modality_files["t1"],
                        t1c=session.modality_files["t1c"],
                        t2=session.modality_files["t2"],
                        flair=session.modality_files["t2fl"]
                    )
                    
                    files_to_send = seg_input.prepare_for_server()
                    client_id = f"{identifier}_{idx}"
                    
                    # Get task status after processing
                    success, task_status = await client.segment_async_with_status(
                        files_to_send=files_to_send,
                        model_name=model_name,
                        client_id=client_id,
                        output_path=session.output_mask_path
                    )
                    
                    if success:
                        self.stats.successful += 1
                        
                        # compute volume report
                        try:
                            from compute_volumes import compute_and_save_volume_report
                            compute_and_save_volume_report(session.output_mask_path)
                        except Exception as e:
                            logger.warning(f"  Volume report failed for {identifier}: {e}")
                        
                        # Collect GPU metrics if benchmarking
                        if self.args.benchmark and task_status and 'gpu_metrics' in task_status:
                            self.gpu_metrics_collected.append(task_status['gpu_metrics'])
                    else:
                        self.stats.failed += 1
                        logger.error(f"  Failed to process {identifier}")
                    
                    return success
                
                except Exception as e:
                    self.stats.failed += 1
                    logger.error(f"  ✗ Error processing {identifier}: {e}")
                    logger.exception(f"  Full traceback for {identifier}:")
                    return False
        
        # Запускаем все задачи параллельно
        tasks = [process_one(idx, session) for idx, session in enumerate(sessions, 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log final statistics
        self.stats.log_summary()
        
        return self.stats.failed == 0
    

    def _save_benchmark_metrics(self, sessions: List):
        """Save benchmark metrics after processing."""
        # Stop monitoring
        self.monitor.stop()
        logger.info("Performance monitoring stopped")
        
        # Calculate total time
        total_time = time.time() - self.pipeline_start_time
        
        # Get system metrics
        system_metrics = self.monitor.get_metrics()
        
        # Determine server name
        if self.args.server_name:
            server_name = self.args.server_name
        else:
            # Auto-detect from config
            server_name = self.config._data.get('segmentation', {}).get('active_profile', 'unknown')
        
        # Generate experiment ID
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"seg_{server_name}_c{self.args.max_concurrent}_{timestamp_str}"
        
        # Create metrics object
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            stage="segmentation",
            parallelism_type="gpu_concurrent",
            parallelism_level=self.args.max_concurrent,
            server_name=server_name,
            gpu_count=self.args.gpu_count,
            gpu_ids=self.args.gpu_ids,
            total_series=self.stats.total,
            successful=self.stats.successful,
            failed=self.stats.failed,
            skipped=0,  # segmentation doesn't skip, only fails
            total_time=total_time,
            time_per_series=total_time / self.stats.total if self.stats.total > 0 else 0,
            throughput=self.stats.total / total_time if total_time > 0 else 0,
            cpu_avg=system_metrics.get('cpu_avg'),
            cpu_max=system_metrics.get('cpu_max'),
            memory_avg_mb=system_metrics.get('memory_avg_mb'),
            memory_peak_mb=system_metrics.get('memory_peak_mb')
        )

        # Aggregate GPU metrics from all tasks
        if self.gpu_metrics_collected:
            utilization_values = [m.get('utilization_avg', 0) for m in self.gpu_metrics_collected if m.get('utilization_avg')]
            memory_values = [m.get('memory_used_mb_avg', 0) for m in self.gpu_metrics_collected if m.get('memory_used_mb_avg')]
            temp_values = [m.get('temperature_avg', 0) for m in self.gpu_metrics_collected if m.get('temperature_avg')]
            
            utilization_max_values = [m.get('utilization_max', 0) for m in self.gpu_metrics_collected if m.get('utilization_max')]
            memory_max_values = [m.get('memory_used_mb_max', 0) for m in self.gpu_metrics_collected if m.get('memory_used_mb_max')]
            temp_max_values = [m.get('temperature_max', 0) for m in self.gpu_metrics_collected if m.get('temperature_max')]
            
            if utilization_values:
                metrics.gpu_utilization_avg = sum(utilization_values) / len(utilization_values)
                metrics.gpu_utilization_max = max(utilization_max_values) if utilization_max_values else None
            
            if memory_values:
                metrics.gpu_memory_used_mb_avg = sum(memory_values) / len(memory_values)
                metrics.gpu_memory_used_mb_max = max(memory_max_values) if memory_max_values else None
            
            if temp_values:
                metrics.gpu_temperature_avg = sum(temp_values) / len(temp_values)
                metrics.gpu_temperature_max = max(temp_max_values) if temp_max_values else None
            
            logger.info("GPU metrics aggregated from all tasks:")
            logger.info(f"  Utilization: avg={metrics.gpu_utilization_avg:.1f}%, max={metrics.gpu_utilization_max:.1f}%")
            logger.info(f"  Memory: avg={metrics.gpu_memory_used_mb_avg:.1f}MB, max={metrics.gpu_memory_used_mb_max:.1f}MB")
            logger.info(f"  Temperature: avg={metrics.gpu_temperature_avg:.1f}°C, max={metrics.gpu_temperature_max:.1f}°C")
        
        # Calculate speedup and efficiency vs baseline
        if self.args.max_concurrent == 1:
            # This IS the baseline for this server
            metrics.speedup = 1.0
            metrics.efficiency = 1.0
        else:
            # Compare to baseline for this server
            baseline_time = self.benchmark_logger.get_baseline_time(
                stage="segmentation",
                server_name=server_name
            )
            if baseline_time and baseline_time > 0:
                metrics.speedup = baseline_time / metrics.time_per_series
                metrics.efficiency = metrics.speedup / metrics.parallelism_level
            else:
                logger.warning(
                    f"No baseline found for server '{server_name}'. "
                    f"Run with --max-concurrent 1 first to establish baseline."
                )
        
        # Log metrics
        self.benchmark_logger.log_metrics(metrics)
        
        logger.info("=" * 60)
        logger.info("BENCHMARK METRICS")
        logger.info("=" * 60)
        logger.info(f"Experiment ID:     {experiment_id}")
        logger.info(f"Server:            {server_name}")
        logger.info(f"GPU count:         {self.args.gpu_count or 'not specified'}")
        logger.info(f"GPU IDs:           {self.args.gpu_ids or 'not specified'}")
        logger.info(f"Max concurrent:    {self.args.max_concurrent}")
        logger.info(f"Total series:      {self.stats.total}")
        logger.info(f"Successful:        {self.stats.successful}")
        logger.info(f"Failed:            {self.stats.failed}")
        logger.info(f"Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"Time per series:   {metrics.time_per_series:.2f}s")
        logger.info(f"Throughput:        {metrics.throughput:.3f} series/sec")
        if metrics.gpu_utilization_avg:
            logger.info(f"GPU Utilization:   avg={metrics.gpu_utilization_avg:.1f}%, max={metrics.gpu_utilization_max:.1f}%")
            logger.info(f"GPU Memory:        avg={metrics.gpu_memory_used_mb_avg:.1f}MB, max={metrics.gpu_memory_used_mb_max:.1f}MB")
            logger.info(f"GPU Temperature:   avg={metrics.gpu_temperature_avg:.1f}°C, max={metrics.gpu_temperature_max:.1f}°C")
        if metrics.speedup:
            logger.info(f"Speedup:           {metrics.speedup:.2f}x")
            logger.info(f"Efficiency:        {metrics.efficiency:.2%}")
        logger.info("=" * 60)

    def _scan_output_structure(self) -> dict:
        """
        Scan output directory for segmentation masks.
        
        Returns:
            Structure dict compatible with InputOutputValidator:
            {patient_id: {session_id: {'segmask'}}}
        """
        from collections import defaultdict
        
        structure = defaultdict(lambda: defaultdict(set))
        
        logger.debug("Scanning output directory for segmentation masks...")
        
        # Scan output directory for BIDS structure
        for subject_dir in sorted(self.args.output_dir.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            # Check for session directories
            session_dirs = sorted([d for d in subject_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith('ses-')])
            
            if session_dirs:
                # Multi-session structure
                for session_dir in session_dirs:
                    session_id = session_dir.name.replace("ses-", "")
                    anat_dir = session_dir / "anat"
                    
                    if anat_dir.exists():
                        # Look for segmentation masks
                        mask_files = list(anat_dir.glob("*_segmask.nii.gz"))
                        if mask_files:
                            structure[patient_id][session_id].add('segmask')
                            logger.debug(f"  Found mask: {patient_id}/ses-{session_id}")
            else:
                # Single-session structure
                anat_dir = subject_dir / "anat"
                if anat_dir.exists():
                    mask_files = list(anat_dir.glob("*_segmask.nii.gz"))
                    if mask_files:
                        # Use empty string for session_id in single-session case
                        structure[patient_id]["001"].add('segmask')
                        logger.debug(f"  Found mask: {patient_id}")
        
        # Convert to regular dict
        return {
            patient_id: {
                session_id: modalities 
                for session_id, modalities in sessions.items()
            }
            for patient_id, sessions in structure.items()
        }
    
    def _run_validation(self):
        """
        Validate input-output correspondence and generate incomplete data report.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION: Input-Output Correspondence")
        logger.info("=" * 60)
        
        try:
            # Initialize validator
            validator = InputOutputValidator(logger=logger)
            
            # Scan input structure (NIfTI files)
            logger.info("Scanning input directory structure...")
            input_structure = validator.scan_structure(
                directory=self.args.input_dir,
                format_type='bids-nifti'
            )
            
            logger.info(f"  Found {len(input_structure)} subjects in input")
            
            # Scan output structure (segmentation masks)
            logger.info("Scanning output directory structure...")
            output_structure = self._scan_output_structure()
            
            logger.info(f"  Found {len(output_structure)} subjects in output")
            
            # Validate segmentation completion
            logger.info("Validating segmentation completion...")
            comparison = validator.validate_segmentation_completion(
                input_structure=input_structure,
                output_structure=output_structure
            )
            
            # Generate report
            report_path = validator.generate_incomplete_report(
                comparison_result=comparison,
                stage_name="06_segmentation",
                output_dir=self.args.output_dir,
                filename="segmentation_incomplete_data.json"
            )
            
            # Log summary
            stats = comparison['statistics']
            logger.info("")
            logger.info("Validation Results:")
            logger.info("-" * 60)
            logger.info(f"  Total patients:       {stats['total_patients']}")
            logger.info(f"  Complete patients:    {stats['complete_patients']}")
            logger.info(f"  Incomplete patients:  {stats['incomplete_patients']}")
            logger.info(f"  Total sessions:       {stats['total_sessions']}")
            logger.info(f"  Complete sessions:    {stats['complete_sessions']}")
            logger.info(f"  Incomplete sessions:  {stats['incomplete_sessions']}")
            logger.info(f"  Success rate:         {stats['success_rate_percent']}%")
            logger.info("-" * 60)
            logger.info(f"  Report saved to: {report_path}")
            logger.info("=" * 60)
            
            if stats['incomplete_patients'] > 0:
                logger.warning(
                    f"⚠️  Found {stats['incomplete_patients']} patients with incomplete data. "
                    f"Check {report_path} for details."
                )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            logger.exception("Full validation traceback:")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch MRI segmentation for BIDS-formatted datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with preprocessed NIfTI files in BIDS format"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for segmentation masks"
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file (default: output_dir/segmentation.log)"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "segmentation_config.yaml",
        help="Path to segmentation configuration file"
    )
    parser.add_argument(
        "--console-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum number of concurrent segmentation requests"
    )
    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking mode (collect performance metrics)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for benchmark results (default: output_dir/benchmark_results)"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server name for benchmark (default: auto-detect from config active_profile)"
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs on server (for benchmark metadata)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="GPU IDs used on server, e.g., '0,1,2' (for benchmark metadata)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions that already have output segmentation masks"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate input-output correspondence after processing"
    )
    
    args = parser.parse_args()

    # Determine log file path
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = args.output_dir / "segmentation.log"

    setup_logging(log_file, args.console_log_level)

    try:
        app_config = Config(args.config)
        runner = SegmentationRunner(config=app_config, args=args)
        is_successful = runner.run()
        sys.exit(0 if is_successful else 1)
        
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.critical("A fatal exception occurred.", exc_info=True)
        sys.exit(1)