"""
Platform Detection Utility
Detects operating system and development environment for Codex CLI installation
"""

import platform
import subprocess
import sys
import shutil
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

from utils.logger import setup_logging


class PlatformDetector:
    """
    Utility class for detecting platform and development environment
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize platform detector with logging"""
        self.logger = logger or setup_logging(name='platform_detector')
        self._platform_info = None
        self._node_info = None
        self._wsl_info = None
    
    def detect_platform(self) -> str:
        """
        Detect the current operating system platform
        
        Returns:
            Platform string: 'windows', 'macos', 'linux', or 'unknown'
        """
        try:
            system = platform.system().lower()
            
            if system == 'windows':
                return 'windows'
            elif system == 'darwin':
                return 'macos'
            elif system == 'linux':
                return 'linux'
            else:
                self.logger.warning(f"Unknown platform detected: {system}")
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Error detecting platform: {e}")
            return 'unknown'
    
    def check_nodejs_availability(self) -> Tuple[bool, Optional[str]]:
        """
        Check if Node.js is available on the system
        
        Returns:
            Tuple of (is_available, version_string)
        """
        try:
            # Check for node command
            result = subprocess.run(
                ['node', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"Node.js found: {version}")
                return True, version
            else:
                self.logger.warning("Node.js command failed")
                return False, None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Node.js version check timed out")
            return False, None
        except FileNotFoundError:
            self.logger.info("Node.js not found in PATH")
            return False, None
        except Exception as e:
            self.logger.error(f"Error checking Node.js: {e}")
            return False, None
    
    def check_npm_availability(self) -> Tuple[bool, Optional[str]]:
        """
        Check if npm is available on the system
        
        Returns:
            Tuple of (is_available, version_string)
        """
        try:
            # Check for npm command
            result = subprocess.run(
                ['npm', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"npm found: {version}")
                return True, version
            else:
                self.logger.warning("npm command failed")
                return False, None
                
        except subprocess.TimeoutExpired:
            self.logger.error("npm version check timed out")
            return False, None
        except FileNotFoundError:
            self.logger.info("npm not found in PATH")
            return False, None
        except Exception as e:
            self.logger.error(f"Error checking npm: {e}")
            return False, None
    
    def detect_wsl_environment(self) -> Dict[str, any]:
        """
        Detect if running in Windows Subsystem for Linux (WSL)
        
        Returns:
            Dictionary with WSL detection information
        """
        wsl_info = {
            'is_wsl': False,
            'wsl_version': None,
            'distribution': None,
            'windows_version': None
        }
        
        try:
            # Check if we're in WSL
            if self.detect_platform() == 'linux':
                # Check for WSL-specific files and environment variables
                wsl_indicators = [
                    Path('/proc/version'),
                    Path('/mnt/c'),  # Windows C: drive mount
                ]
                
                # Check /proc/version for Microsoft/WSL
                if Path('/proc/version').exists():
                    with open('/proc/version', 'r') as f:
                        version_info = f.read().lower()
                        if 'microsoft' in version_info or 'wsl' in version_info:
                            wsl_info['is_wsl'] = True
                            
                            # Try to determine WSL version
                            if 'wsl2' in version_info:
                                wsl_info['wsl_version'] = '2'
                            elif 'microsoft' in version_info:
                                wsl_info['wsl_version'] = '1'
                
                # Check for Windows drives mounted
                if Path('/mnt/c').exists():
                    wsl_info['is_wsl'] = True
                
                # Get distribution info
                if wsl_info['is_wsl']:
                    try:
                        result = subprocess.run(
                            ['lsb_release', '-d'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            wsl_info['distribution'] = result.stdout.strip()
                    except:
                        pass
            
            if wsl_info['is_wsl']:
                self.logger.info(f"WSL environment detected: {wsl_info}")
            
        except Exception as e:
            self.logger.error(f"Error detecting WSL environment: {e}")
        
        return wsl_info
    
    def get_installation_recommendations(self) -> Dict[str, any]:
        """
        Get platform-specific installation recommendations
        
        Returns:
            Dictionary with installation guidance
        """
        platform_name = self.detect_platform()
        node_available, node_version = self.check_nodejs_availability()
        npm_available, npm_version = self.check_npm_availability()
        wsl_info = self.detect_wsl_environment()
        
        recommendations = {
            'platform': platform_name,
            'node_available': node_available,
            'node_version': node_version,
            'npm_available': npm_available,
            'npm_version': npm_version,
            'wsl_info': wsl_info,
            'installation_steps': [],
            'warnings': [],
            'can_install_codex': False
        }
        
        # Platform-specific recommendations
        if platform_name == 'windows':
            if wsl_info['is_wsl']:
                recommendations['installation_steps'] = [
                    "You're in WSL - this is the recommended environment for Codex CLI on Windows",
                    "Install Node.js in WSL: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
                    "sudo apt-get install -y nodejs",
                    "npm install -g @openai/codex"
                ]
            else:
                recommendations['warnings'].append(
                    "Windows detected - WSL is recommended for best Codex CLI compatibility"
                )
                recommendations['installation_steps'] = [
                    "Option 1 (Recommended): Install WSL2 and use Linux environment",
                    "Option 2: Install Node.js from https://nodejs.org/",
                    "Then run: npm install -g @openai/codex"
                ]
        
        elif platform_name == 'macos':
            if not node_available:
                recommendations['installation_steps'] = [
                    "Install Node.js using Homebrew: brew install node",
                    "Or download from: https://nodejs.org/",
                    "npm install -g @openai/codex"
                ]
            else:
                recommendations['installation_steps'] = [
                    "npm install -g @openai/codex"
                ]
        
        elif platform_name == 'linux':
            if not node_available:
                recommendations['installation_steps'] = [
                    "Install Node.js: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
                    "sudo apt-get install -y nodejs",
                    "npm install -g @openai/codex"
                ]
            else:
                recommendations['installation_steps'] = [
                    "npm install -g @openai/codex"
                ]
        
        # Check if we can install Codex
        if node_available and npm_available:
            recommendations['can_install_codex'] = True
        elif platform_name in ['macos', 'linux'] or wsl_info['is_wsl']:
            recommendations['can_install_codex'] = True  # Can install Node.js first
        
        return recommendations
    
    def get_platform_summary(self) -> Dict[str, any]:
        """
        Get comprehensive platform information summary
        
        Returns:
            Complete platform analysis
        """
        return {
            'platform': self.detect_platform(),
            'python_version': sys.version,
            'node_info': self.check_nodejs_availability(),
            'npm_info': self.check_npm_availability(),
            'wsl_info': self.detect_wsl_environment(),
            'recommendations': self.get_installation_recommendations()
        }


def main():
    """Test the platform detector"""
    detector = PlatformDetector()
    summary = detector.get_platform_summary()
    
    print("Platform Detection Summary:")
    print("=" * 40)
    print(f"Platform: {summary['platform']}")
    print(f"Python: {summary['python_version']}")
    print(f"Node.js: {summary['node_info']}")
    print(f"npm: {summary['npm_info']}")
    print(f"WSL: {summary['wsl_info']}")
    print("\nInstallation Recommendations:")
    for step in summary['recommendations']['installation_steps']:
        print(f"  - {step}")
    
    if summary['recommendations']['warnings']:
        print("\nWarnings:")
        for warning in summary['recommendations']['warnings']:
            print(f"  ! {warning}")


if __name__ == "__main__":
    main()