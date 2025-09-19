"""
Unit tests for platform detection utility
"""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sys
from pathlib import Path

from codex_integration.platform_detector import PlatformDetector


class TestPlatformDetector(unittest.TestCase):
    """Test cases for PlatformDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PlatformDetector()
    
    @patch('platform.system')
    def test_detect_platform_windows(self, mock_system):
        """Test Windows platform detection"""
        mock_system.return_value = 'Windows'
        result = self.detector.detect_platform()
        self.assertEqual(result, 'windows')
    
    @patch('platform.system')
    def test_detect_platform_macos(self, mock_system):
        """Test macOS platform detection"""
        mock_system.return_value = 'Darwin'
        result = self.detector.detect_platform()
        self.assertEqual(result, 'macos')
    
    @patch('platform.system')
    def test_detect_platform_linux(self, mock_system):
        """Test Linux platform detection"""
        mock_system.return_value = 'Linux'
        result = self.detector.detect_platform()
        self.assertEqual(result, 'linux')
    
    @patch('platform.system')
    def test_detect_platform_unknown(self, mock_system):
        """Test unknown platform detection"""
        mock_system.return_value = 'FreeBSD'
        result = self.detector.detect_platform()
        self.assertEqual(result, 'unknown')
    
    @patch('subprocess.run')
    def test_check_nodejs_available_success(self, mock_run):
        """Test successful Node.js detection"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'v18.17.0\n'
        mock_run.return_value = mock_result
        
        available, version = self.detector.check_nodejs_availability()
        self.assertTrue(available)
        self.assertEqual(version, 'v18.17.0')
    
    @patch('subprocess.run')
    def test_check_nodejs_not_available(self, mock_run):
        """Test Node.js not available"""
        mock_run.side_effect = FileNotFoundError()
        
        available, version = self.detector.check_nodejs_availability()
        self.assertFalse(available)
        self.assertIsNone(version)
    
    @patch('subprocess.run')
    def test_check_npm_available_success(self, mock_run):
        """Test successful npm detection"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '9.8.1\n'
        mock_run.return_value = mock_result
        
        available, version = self.detector.check_npm_availability()
        self.assertTrue(available)
        self.assertEqual(version, '9.8.1')
    
    @patch('subprocess.run')
    def test_check_npm_timeout(self, mock_run):
        """Test npm check timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired('npm', 10)
        
        available, version = self.detector.check_npm_availability()
        self.assertFalse(available)
        self.assertIsNone(version)
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    @patch('codex_integration.platform_detector.PlatformDetector.detect_platform')
    def test_detect_wsl_environment_wsl2(self, mock_platform, mock_open, mock_exists):
        """Test WSL2 environment detection"""
        mock_platform.return_value = 'linux'
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = 'Linux version 5.4.0-microsoft-standard-WSL2'
        
        wsl_info = self.detector.detect_wsl_environment()
        self.assertTrue(wsl_info['is_wsl'])
        self.assertEqual(wsl_info['wsl_version'], '2')
    
    @patch('pathlib.Path.exists')
    @patch('codex_integration.platform_detector.PlatformDetector.detect_platform')
    def test_detect_wsl_environment_not_wsl(self, mock_platform, mock_exists):
        """Test non-WSL Linux environment"""
        mock_platform.return_value = 'linux'
        mock_exists.return_value = False
        
        wsl_info = self.detector.detect_wsl_environment()
        self.assertFalse(wsl_info['is_wsl'])
        self.assertIsNone(wsl_info['wsl_version'])
    
    @patch('codex_integration.platform_detector.PlatformDetector.detect_platform')
    @patch('codex_integration.platform_detector.PlatformDetector.check_nodejs_availability')
    @patch('codex_integration.platform_detector.PlatformDetector.check_npm_availability')
    def test_get_installation_recommendations_macos_ready(self, mock_npm, mock_node, mock_platform):
        """Test installation recommendations for macOS with Node.js ready"""
        mock_platform.return_value = 'macos'
        mock_node.return_value = (True, 'v18.17.0')
        mock_npm.return_value = (True, '9.8.1')
        
        recommendations = self.detector.get_installation_recommendations()
        self.assertTrue(recommendations['can_install_codex'])
        self.assertIn('npm install -g @openai/codex', recommendations['installation_steps'])
    
    @patch('codex_integration.platform_detector.PlatformDetector.detect_platform')
    @patch('codex_integration.platform_detector.PlatformDetector.check_nodejs_availability')
    @patch('codex_integration.platform_detector.PlatformDetector.detect_wsl_environment')
    def test_get_installation_recommendations_windows_no_wsl(self, mock_wsl, mock_node, mock_platform):
        """Test installation recommendations for Windows without WSL"""
        mock_platform.return_value = 'windows'
        mock_node.return_value = (False, None)
        mock_wsl.return_value = {'is_wsl': False}
        
        recommendations = self.detector.get_installation_recommendations()
        self.assertIn('WSL is recommended', recommendations['warnings'][0])
        self.assertIn('Install WSL2', recommendations['installation_steps'][0])


if __name__ == '__main__':
    unittest.main()