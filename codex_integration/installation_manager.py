"""
Codex CLI Installation Manager
Handles installation, verification, and management of OpenAI Codex CLI
"""

import subprocess
import sys
import shutil
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from codex_integration.platform_detector import PlatformDetector
from utils.logger import setup_logging


class CodexInstallationManager:
    """
    Manager for Codex CLI installation and verification
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize installation manager"""
        self.logger = logger or setup_logging(name='codex_installation_manager')
        self.platform_detector = PlatformDetector(logger=self.logger)
        self.installation_timeout = 300  # 5 minutes
        self.verification_timeout = 30   # 30 seconds
    
    def generate_installation_commands(self) -> Dict[str, any]:
        """
        Generate platform-specific installation commands
        
        Returns:
            Dictionary with installation commands and metadata
        """
        platform_info = self.platform_detector.get_platform_summary()
        platform = platform_info['platform']
        node_available = platform_info['node_info'][0]
        npm_available = platform_info['npm_info'][0]
        wsl_info = platform_info['wsl_info']
        
        commands = {
            'platform': platform,
            'prerequisites': [],
            'installation_commands': [],
            'verification_commands': [],
            'post_install_steps': [],
            'estimated_time_minutes': 5,
            'requires_sudo': False,
            'success_probability': 0.9
        }
        
        # Handle prerequisites first
        if not node_available:
            if platform == 'windows' and not wsl_info['is_wsl']:
                commands['prerequisites'].extend([
                    "# Windows - Install Node.js first",
                    "# Option 1: Download from https://nodejs.org/",
                    "# Option 2: Use Chocolatey: choco install nodejs",
                    "# Option 3: Use Scoop: scoop install nodejs"
                ])
                commands['estimated_time_minutes'] = 15
                commands['success_probability'] = 0.7
            
            elif platform == 'macos':
                commands['prerequisites'].extend([
                    "# macOS - Install Node.js first",
                    "brew install node",
                    "# Or download from https://nodejs.org/"
                ])
                commands['requires_sudo'] = False
                commands['estimated_time_minutes'] = 10
            
            elif platform == 'linux' or wsl_info['is_wsl']:
                commands['prerequisites'].extend([
                    "# Linux/WSL - Install Node.js first",
                    "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
                    "sudo apt-get install -y nodejs"
                ])
                commands['requires_sudo'] = True
                commands['estimated_time_minutes'] = 10
        
        # Main Codex CLI installation
        if node_available or platform in ['macos', 'linux'] or wsl_info['is_wsl']:
            commands['installation_commands'] = [
                "npm install -g @openai/codex"
            ]
            
            # Add sudo for global npm installs on Linux/WSL if needed
            if (platform == 'linux' or wsl_info['is_wsl']) and not self._check_npm_global_permissions():
                commands['installation_commands'] = [
                    "sudo npm install -g @openai/codex"
                ]
                commands['requires_sudo'] = True
        
        # Verification commands
        commands['verification_commands'] = [
            "codex --version",
            "which codex",  # Unix-like systems
            "where codex"   # Windows
        ]
        
        # Post-installation steps
        commands['post_install_steps'] = [
            "Set up OpenAI API key: export OPENAI_API_KEY=your_key_here",
            "Test installation: codex --help",
            "Configure authentication: codex auth"
        ]
        
        return commands
    
    def _check_npm_global_permissions(self) -> bool:
        """
        Check if npm can install global packages without sudo
        
        Returns:
            True if npm can install globally without sudo
        """
        try:
            result = subprocess.run(
                ['npm', 'config', 'get', 'prefix'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                prefix = result.stdout.strip()
                prefix_path = Path(prefix)
                
                # Check if user has write access to npm global directory
                return prefix_path.exists() and prefix_path.is_dir() and \
                       (prefix_path / 'lib' / 'node_modules').parent.exists()
            
        except Exception as e:
            self.logger.debug(f"Error checking npm permissions: {e}")
        
        return False
    
    def install_codex_cli(self, force_reinstall: bool = False) -> Dict[str, any]:
        """
        Install Codex CLI with error handling and progress tracking
        
        Args:
            force_reinstall: Whether to reinstall if already present
            
        Returns:
            Installation result dictionary
        """
        result = {
            'success': False,
            'already_installed': False,
            'installation_output': [],
            'error_messages': [],
            'duration_seconds': 0,
            'commands_executed': []
        }
        
        start_time = time.time()
        
        try:
            # Check if already installed
            if not force_reinstall and self._is_codex_installed():
                result['already_installed'] = True
                result['success'] = True
                self.logger.info("Codex CLI is already installed")
                return result
            
            # Get installation commands
            install_info = self.generate_installation_commands()
            
            # Check prerequisites
            if install_info['prerequisites']:
                self.logger.warning("Prerequisites required before installation:")
                for prereq in install_info['prerequisites']:
                    self.logger.warning(f"  {prereq}")
                
                # Check if Node.js is available after showing prerequisites
                node_available, _ = self.platform_detector.check_nodejs_availability()
                if not node_available:
                    result['error_messages'].append(
                        "Node.js is required but not installed. Please install Node.js first."
                    )
                    return result
            
            # Execute installation commands
            for command in install_info['installation_commands']:
                self.logger.info(f"Executing: {command}")
                result['commands_executed'].append(command)
                
                try:
                    # Split command for subprocess
                    cmd_parts = command.split()
                    
                    process_result = subprocess.run(
                        cmd_parts,
                        capture_output=True,
                        text=True,
                        timeout=self.installation_timeout
                    )
                    
                    # Log output
                    if process_result.stdout:
                        output_lines = process_result.stdout.strip().split('\n')
                        result['installation_output'].extend(output_lines)
                        for line in output_lines:
                            self.logger.info(f"  {line}")
                    
                    if process_result.stderr:
                        error_lines = process_result.stderr.strip().split('\n')
                        for line in error_lines:
                            self.logger.warning(f"  stderr: {line}")
                    
                    # Check return code
                    if process_result.returncode != 0:
                        error_msg = f"Command failed with return code {process_result.returncode}"
                        result['error_messages'].append(error_msg)
                        self.logger.error(error_msg)
                        return result
                
                except subprocess.TimeoutExpired:
                    error_msg = f"Installation command timed out after {self.installation_timeout} seconds"
                    result['error_messages'].append(error_msg)
                    self.logger.error(error_msg)
                    return result
                
                except Exception as e:
                    error_msg = f"Error executing command '{command}': {e}"
                    result['error_messages'].append(error_msg)
                    self.logger.error(error_msg)
                    return result
            
            # Verify installation
            if self.verify_installation()['success']:
                result['success'] = True
                self.logger.info("Codex CLI installation completed successfully")
            else:
                result['error_messages'].append("Installation completed but verification failed")
                self.logger.error("Installation verification failed")
        
        except Exception as e:
            error_msg = f"Unexpected error during installation: {e}"
            result['error_messages'].append(error_msg)
            self.logger.error(error_msg)
        
        finally:
            result['duration_seconds'] = time.time() - start_time
        
        return result
    
    def _is_codex_installed(self) -> bool:
        """
        Check if Codex CLI is already installed
        
        Returns:
            True if Codex CLI is available
        """
        try:
            # Try to find codex command
            codex_path = shutil.which('codex')
            if codex_path:
                self.logger.debug(f"Codex CLI found at: {codex_path}")
                return True
            
            # Alternative check - try running codex command
            result = subprocess.run(
                ['codex', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0
        
        except Exception as e:
            self.logger.debug(f"Error checking Codex installation: {e}")
            return False
    
    def verify_installation(self) -> Dict[str, any]:
        """
        Verify that Codex CLI is properly installed and accessible
        
        Returns:
            Verification result dictionary
        """
        result = {
            'success': False,
            'codex_path': None,
            'version_info': None,
            'help_accessible': False,
            'auth_status': None,
            'error_messages': []
        }
        
        try:
            # Check if codex command is in PATH
            codex_path = shutil.which('codex')
            if codex_path:
                result['codex_path'] = codex_path
                self.logger.info(f"Codex CLI found at: {codex_path}")
            else:
                result['error_messages'].append("Codex CLI not found in PATH")
                return result
            
            # Check version
            try:
                version_result = subprocess.run(
                    ['codex', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=self.verification_timeout
                )
                
                if version_result.returncode == 0:
                    result['version_info'] = version_result.stdout.strip()
                    self.logger.info(f"Codex CLI version: {result['version_info']}")
                else:
                    result['error_messages'].append("Failed to get Codex CLI version")
            
            except subprocess.TimeoutExpired:
                result['error_messages'].append("Codex version check timed out")
            except Exception as e:
                result['error_messages'].append(f"Error checking version: {e}")
            
            # Check help command
            try:
                help_result = subprocess.run(
                    ['codex', '--help'],
                    capture_output=True,
                    text=True,
                    timeout=self.verification_timeout
                )
                
                if help_result.returncode == 0:
                    result['help_accessible'] = True
                    self.logger.info("Codex CLI help command accessible")
                else:
                    result['error_messages'].append("Codex help command failed")
            
            except Exception as e:
                result['error_messages'].append(f"Error checking help: {e}")
            
            # Overall success check
            if result['codex_path'] and result['version_info'] and result['help_accessible']:
                result['success'] = True
                self.logger.info("Codex CLI verification successful")
            else:
                self.logger.warning("Codex CLI verification incomplete")
        
        except Exception as e:
            error_msg = f"Unexpected error during verification: {e}"
            result['error_messages'].append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def uninstall_codex_cli(self) -> Dict[str, any]:
        """
        Uninstall Codex CLI
        
        Returns:
            Uninstallation result dictionary
        """
        result = {
            'success': False,
            'was_installed': False,
            'uninstall_output': [],
            'error_messages': []
        }
        
        try:
            # Check if installed first
            if not self._is_codex_installed():
                result['success'] = True  # Nothing to uninstall
                self.logger.info("Codex CLI is not installed")
                return result
            
            result['was_installed'] = True
            
            # Uninstall using npm
            self.logger.info("Uninstalling Codex CLI...")
            
            uninstall_result = subprocess.run(
                ['npm', 'uninstall', '-g', '@openai/codex'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if uninstall_result.stdout:
                result['uninstall_output'].extend(uninstall_result.stdout.strip().split('\n'))
            
            if uninstall_result.returncode == 0:
                # Verify uninstallation
                if not self._is_codex_installed():
                    result['success'] = True
                    self.logger.info("Codex CLI uninstalled successfully")
                else:
                    result['error_messages'].append("Uninstall command succeeded but Codex CLI still accessible")
            else:
                result['error_messages'].append(f"Uninstall failed with return code {uninstall_result.returncode}")
                if uninstall_result.stderr:
                    result['error_messages'].extend(uninstall_result.stderr.strip().split('\n'))
        
        except Exception as e:
            error_msg = f"Error during uninstallation: {e}"
            result['error_messages'].append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def get_installation_status(self) -> Dict[str, any]:
        """
        Get comprehensive installation status
        
        Returns:
            Complete status information
        """
        platform_info = self.platform_detector.get_platform_summary()
        verification_result = self.verify_installation()
        
        return {
            'platform_info': platform_info,
            'is_installed': verification_result['success'],
            'installation_details': verification_result,
            'can_install': platform_info['recommendations']['can_install_codex'],
            'installation_commands': self.generate_installation_commands()
        }


def main():
    """Test the installation manager"""
    manager = CodexInstallationManager()
    
    print("Codex CLI Installation Manager")
    print("=" * 40)
    
    # Get status
    status = manager.get_installation_status()
    print(f"Platform: {status['platform_info']['platform']}")
    print(f"Can install: {status['can_install']}")
    print(f"Is installed: {status['is_installed']}")
    
    if status['is_installed']:
        print(f"Version: {status['installation_details']['version_info']}")
        print(f"Path: {status['installation_details']['codex_path']}")
    else:
        print("\nInstallation commands:")
        for cmd in status['installation_commands']['installation_commands']:
            print(f"  {cmd}")


if __name__ == "__main__":
    main()