"""
Unit tests for Codex CLI installation manager
"""

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import time

from codex_integration.installation_manager import CodexInstallationManager


class TestCodexInstallationManager(unittest.TestCase):
    """Test cases for CodexInstallationManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = CodexInstallationManager()
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._check_npm_global_permissions')
    @patch('codex_integration.platform_detector.PlatformDetector.get_platform_summary')
    def test_generate_installation_commands_macos_ready(self, mock_platform, mock_npm_perms):
        """Test installation command generation for macOS with Node.js ready"""
        mock_platform.return_value = {
            'platform': 'macos',
            'node_info': (True, 'v18.17.0'),
            'npm_info': (True, '9.8.1'),
            'wsl_info': {'is_wsl': False}
        }
        mock_npm_perms.return_value = True
        
        commands = self.manager.generate_installation_commands()
        
        self.assertEqual(commands['platform'], 'macos')
        self.assertIn('npm install -g @openai/codex', commands['installation_commands'])
        self.assertFalse(commands['requires_sudo'])
        self.assertEqual(len(commands['prerequisites']), 0)
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._check_npm_global_permissions')
    @patch('codex_integration.platform_detector.PlatformDetector.get_platform_summary')
    def test_generate_installation_commands_linux_no_node(self, mock_platform, mock_npm_perms):
        """Test installation command generation for Linux without Node.js"""
        mock_platform.return_value = {
            'platform': 'linux',
            'node_info': (False, None),
            'npm_info': (False, None),
            'wsl_info': {'is_wsl': False}
        }
        mock_npm_perms.return_value = False
        
        commands = self.manager.generate_installation_commands()
        
        self.assertEqual(commands['platform'], 'linux')
        self.assertTrue(len(commands['prerequisites']) > 0)
        self.assertIn('curl -fsSL https://deb.nodesource.com/setup_lts.x', commands['prerequisites'][1])
        self.assertTrue(commands['requires_sudo'])
        self.assertGreater(commands['estimated_time_minutes'], 5)
    
    @patch('subprocess.run')
    def test_check_npm_global_permissions_success(self, mock_run):
        """Test npm global permissions check success"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '/usr/local\n'
        mock_run.return_value = mock_result
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            result = self.manager._check_npm_global_permissions()
            self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_check_npm_global_permissions_failure(self, mock_run):
        """Test npm global permissions check failure"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'npm')
        
        result = self.manager._check_npm_global_permissions()
        self.assertFalse(result)
    
    @patch('shutil.which')
    def test_is_codex_installed_found(self, mock_which):
        """Test Codex CLI detection when installed"""
        mock_which.return_value = '/usr/local/bin/codex'
        
        result = self.manager._is_codex_installed()
        self.assertTrue(result)
    
    @patch('shutil.which')
    @patch('subprocess.run')
    def test_is_codex_installed_not_found(self, mock_run, mock_which):
        """Test Codex CLI detection when not installed"""
        mock_which.return_value = None
        mock_run.side_effect = FileNotFoundError()
        
        result = self.manager._is_codex_installed()
        self.assertFalse(result)
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._is_codex_installed')
    def test_install_codex_cli_already_installed(self, mock_is_installed):
        """Test installation when Codex CLI is already installed"""
        mock_is_installed.return_value = True
        
        result = self.manager.install_codex_cli(force_reinstall=False)
        
        self.assertTrue(result['success'])
        self.assertTrue(result['already_installed'])
        self.assertEqual(len(result['commands_executed']), 0)
    
    @patch('codex_integration.installation_manager.CodexInstallationManager.verify_installation')
    @patch('codex_integration.installation_manager.CodexInstallationManager._is_codex_installed')
    @patch('subprocess.run')
    @patch('codex_integration.platform_detector.PlatformDetector.check_nodejs_availability')
    def test_install_codex_cli_success(self, mock_node_check, mock_run, mock_is_installed, mock_verify):
        """Test successful Codex CLI installation"""
        mock_is_installed.return_value = False
        mock_node_check.return_value = (True, 'v18.17.0')
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'Successfully installed @openai/codex'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        mock_verify.return_value = {'success': True}
        
        result = self.manager.install_codex_cli()
        
        self.assertTrue(result['success'])
        self.assertFalse(result['already_installed'])
        self.assertGreater(len(result['commands_executed']), 0)
        self.assertIn('npm install -g @openai/codex', result['commands_executed'][0])
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._is_codex_installed')
    @patch('subprocess.run')
    @patch('codex_integration.platform_detector.PlatformDetector.check_nodejs_availability')
    def test_install_codex_cli_command_failure(self, mock_node_check, mock_run, mock_is_installed):
        """Test installation failure due to command error"""
        mock_is_installed.return_value = False
        mock_node_check.return_value = (True, 'v18.17.0')
        
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ''
        mock_result.stderr = 'Permission denied'
        mock_run.return_value = mock_result
        
        result = self.manager.install_codex_cli()
        
        self.assertFalse(result['success'])
        self.assertGreater(len(result['error_messages']), 0)
        self.assertIn('failed with return code 1', result['error_messages'][0])
    
    @patch('shutil.which')
    @patch('subprocess.run')
    def test_verify_installation_success(self, mock_run, mock_which):
        """Test successful installation verification"""
        mock_which.return_value = '/usr/local/bin/codex'
        
        # Mock version check
        version_result = MagicMock()
        version_result.returncode = 0
        version_result.stdout = 'codex 1.0.0'
        
        # Mock help check
        help_result = MagicMock()
        help_result.returncode = 0
        help_result.stdout = 'Usage: codex [options]'
        
        mock_run.side_effect = [version_result, help_result]
        
        result = self.manager.verify_installation()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['codex_path'], '/usr/local/bin/codex')
        self.assertEqual(result['version_info'], 'codex 1.0.0')
        self.assertTrue(result['help_accessible'])
    
    @patch('shutil.which')
    def test_verify_installation_not_found(self, mock_which):
        """Test verification when Codex CLI is not found"""
        mock_which.return_value = None
        
        result = self.manager.verify_installation()
        
        self.assertFalse(result['success'])
        self.assertIsNone(result['codex_path'])
        self.assertIn('not found in PATH', result['error_messages'][0])
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._is_codex_installed')
    @patch('subprocess.run')
    def test_uninstall_codex_cli_success(self, mock_run, mock_is_installed):
        """Test successful Codex CLI uninstallation"""
        # First call returns True (installed), second call returns False (uninstalled)
        mock_is_installed.side_effect = [True, False]
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'removed @openai/codex'
        mock_run.return_value = mock_result
        
        result = self.manager.uninstall_codex_cli()
        
        self.assertTrue(result['success'])
        self.assertTrue(result['was_installed'])
        self.assertIn('removed @openai/codex', result['uninstall_output'])
    
    @patch('codex_integration.installation_manager.CodexInstallationManager._is_codex_installed')
    def test_uninstall_codex_cli_not_installed(self, mock_is_installed):
        """Test uninstallation when Codex CLI is not installed"""
        mock_is_installed.return_value = False
        
        result = self.manager.uninstall_codex_cli()
        
        self.assertTrue(result['success'])
        self.assertFalse(result['was_installed'])


if __name__ == '__main__':
    unittest.main()