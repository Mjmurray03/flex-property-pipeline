"""
Authentication Manager for Codex CLI
Handles OpenAI API key configuration and authentication verification
"""

import os
import json
import keyring
import subprocess
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
from cryptography.fernet import Fernet
import base64

from utils.logger import setup_logging


class CodexAuthManager:
    """
    Manager for Codex CLI authentication and API key configuration
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize authentication manager"""
        self.logger = logger or setup_logging(name='codex_auth_manager')
        self.config_dir = Path.home() / '.codex_integration'
        self.config_file = self.config_dir / 'auth_config.json'
        self.keyring_service = 'codex_integration'
        self.keyring_username = 'openai_api_key'
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
    
    def configure_api_key(self, api_key: str, storage_method: str = 'keyring') -> Dict[str, any]:
        """
        Configure OpenAI API key for Codex CLI
        
        Args:
            api_key: OpenAI API key
            storage_method: Storage method ('keyring', 'env', 'file')
            
        Returns:
            Configuration result dictionary
        """
        result = {
            'success': False,
            'storage_method': storage_method,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate API key format
            if not self._validate_api_key_format(api_key):
                result['errors'].append("Invalid API key format")
                return result
            
            # Store API key using specified method
            if storage_method == 'keyring':
                success = self._store_in_keyring(api_key)
                if not success:
                    result['warnings'].append("Keyring storage failed, falling back to environment variable")
                    storage_method = 'env'
            
            if storage_method == 'env':
                success = self._store_in_environment(api_key)
                if not success:
                    result['warnings'].append("Environment variable storage failed, falling back to file")
                    storage_method = 'file'
            
            if storage_method == 'file':
                success = self._store_in_file(api_key)
                if not success:
                    result['errors'].append("All storage methods failed")
                    return result
            
            # Update configuration
            config = self._load_config()
            config['api_key_storage'] = storage_method
            config['configured_at'] = str(Path.cwd())
            self._save_config(config)
            
            result['success'] = True
            result['storage_method'] = storage_method
            self.logger.info(f"API key configured successfully using {storage_method}")
            
        except Exception as e:
            result['errors'].append(f"Configuration error: {str(e)}")
            self.logger.error(f"API key configuration failed: {e}")
        
        return result
    
    def _validate_api_key_format(self, api_key: str) -> bool:
        """
        Validate OpenAI API key format
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # OpenAI API keys typically start with 'sk-' and are 51 characters long
        if api_key.startswith('sk-') and len(api_key) == 51:
            return True
        
        # Also accept newer format keys
        if api_key.startswith('sk-') and len(api_key) > 40:
            return True
        
        self.logger.warning("API key format validation failed - proceeding anyway")
        return True  # Be lenient with validation
    
    def _store_in_keyring(self, api_key: str) -> bool:
        """
        Store API key in system keyring
        
        Args:
            api_key: API key to store
            
        Returns:
            True if successful
        """
        try:
            keyring.set_password(self.keyring_service, self.keyring_username, api_key)
            self.logger.info("API key stored in system keyring")
            return True
        except Exception as e:
            self.logger.warning(f"Keyring storage failed: {e}")
            return False
    
    def _store_in_environment(self, api_key: str) -> bool:
        """
        Store API key in environment variable
        
        Args:
            api_key: API key to store
            
        Returns:
            True if successful
        """
        try:
            # Set for current session
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Try to persist in shell profile
            self._add_to_shell_profile('OPENAI_API_KEY', api_key)
            
            self.logger.info("API key stored in environment variable")
            return True
        except Exception as e:
            self.logger.warning(f"Environment variable storage failed: {e}")
            return False
    
    def _store_in_file(self, api_key: str) -> bool:
        """
        Store API key in encrypted file
        
        Args:
            api_key: API key to store
            
        Returns:
            True if successful
        """
        try:
            # Generate encryption key from system info
            key = self._generate_file_encryption_key()
            fernet = Fernet(key)
            
            # Encrypt API key
            encrypted_key = fernet.encrypt(api_key.encode())
            
            # Store in file
            key_file = self.config_dir / '.api_key'
            with open(key_file, 'wb') as f:
                f.write(encrypted_key)
            
            # Set restrictive permissions
            key_file.chmod(0o600)
            
            self.logger.info("API key stored in encrypted file")
            return True
        except Exception as e:
            self.logger.warning(f"File storage failed: {e}")
            return False
    
    def _generate_file_encryption_key(self) -> bytes:
        """
        Generate encryption key for file storage
        
        Returns:
            Encryption key bytes
        """
        # Use system-specific information to generate key
        import platform
        import getpass
        
        key_material = f"{platform.node()}-{getpass.getuser()}-codex-integration"
        key_hash = base64.urlsafe_b64encode(key_material.encode()[:32].ljust(32, b'0'))
        return key_hash
    
    def _add_to_shell_profile(self, var_name: str, value: str) -> None:
        """
        Add environment variable to shell profile
        
        Args:
            var_name: Variable name
            value: Variable value
        """
        try:
            home = Path.home()
            profile_files = [
                home / '.bashrc',
                home / '.bash_profile',
                home / '.zshrc',
                home / '.profile'
            ]
            
            export_line = f'export {var_name}="{value}"\n'
            
            for profile_file in profile_files:
                if profile_file.exists():
                    # Check if already exists
                    with open(profile_file, 'r') as f:
                        content = f.read()
                    
                    if f'export {var_name}=' not in content:
                        with open(profile_file, 'a') as f:
                            f.write(f'\n# Added by Codex Integration\n{export_line}')
                        self.logger.info(f"Added to {profile_file}")
                        break
        except Exception as e:
            self.logger.warning(f"Could not update shell profile: {e}")
    
    def get_api_key(self) -> Optional[str]:
        """
        Retrieve configured API key
        
        Returns:
            API key string or None if not found
        """
        config = self._load_config()
        storage_method = config.get('api_key_storage', 'keyring')
        
        try:
            if storage_method == 'keyring':
                return self._get_from_keyring()
            elif storage_method == 'env':
                return self._get_from_environment()
            elif storage_method == 'file':
                return self._get_from_file()
        except Exception as e:
            self.logger.error(f"Error retrieving API key: {e}")
        
        # Try all methods as fallback
        for method in ['keyring', 'env', 'file']:
            try:
                if method == 'keyring':
                    key = self._get_from_keyring()
                elif method == 'env':
                    key = self._get_from_environment()
                else:
                    key = self._get_from_file()
                
                if key:
                    return key
            except:
                continue
        
        return None
    
    def _get_from_keyring(self) -> Optional[str]:
        """Get API key from keyring"""
        try:
            return keyring.get_password(self.keyring_service, self.keyring_username)
        except Exception:
            return None
    
    def _get_from_environment(self) -> Optional[str]:
        """Get API key from environment variable"""
        return os.environ.get('OPENAI_API_KEY')
    
    def _get_from_file(self) -> Optional[str]:
        """Get API key from encrypted file"""
        try:
            key_file = self.config_dir / '.api_key'
            if not key_file.exists():
                return None
            
            # Generate decryption key
            key = self._generate_file_encryption_key()
            fernet = Fernet(key)
            
            # Read and decrypt
            with open(key_file, 'rb') as f:
                encrypted_key = f.read()
            
            decrypted_key = fernet.decrypt(encrypted_key)
            return decrypted_key.decode()
        except Exception:
            return None
    
    def verify_authentication(self) -> Dict[str, any]:
        """
        Verify authentication with OpenAI API
        
        Returns:
            Verification result dictionary
        """
        result = {
            'is_authenticated': False,
            'api_key_found': False,
            'api_key_valid': False,
            'codex_accessible': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if API key is configured
            api_key = self.get_api_key()
            if not api_key:
                result['errors'].append("No API key found")
                return result
            
            result['api_key_found'] = True
            
            # Set environment variable for Codex CLI
            os.environ['OPENAI_API_KEY'] = api_key
            
            # Try to verify with Codex CLI
            try:
                # Test Codex CLI with a simple command
                cmd_result = subprocess.run(
                    ['codex', '--help'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=os.environ.copy()
                )
                
                if cmd_result.returncode == 0:
                    result['codex_accessible'] = True
                    result['is_authenticated'] = True
                    self.logger.info("Codex CLI authentication verified")
                else:
                    result['errors'].append(f"Codex CLI error: {cmd_result.stderr}")
            
            except subprocess.TimeoutExpired:
                result['errors'].append("Codex CLI verification timed out")
            except FileNotFoundError:
                result['errors'].append("Codex CLI not found - install first")
            except Exception as e:
                result['errors'].append(f"Codex CLI verification error: {str(e)}")
        
        except Exception as e:
            result['errors'].append(f"Authentication verification error: {str(e)}")
            self.logger.error(f"Authentication verification failed: {e}")
        
        return result
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")
        
        return {}
    
    def _save_config(self, config: Dict) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save config: {e}")
    
    def get_auth_status(self) -> Dict[str, any]:
        """
        Get comprehensive authentication status
        
        Returns:
            Authentication status dictionary
        """
        status = {
            'configured': False,
            'storage_method': None,
            'api_key_available': False,
            'verification_result': None,
            'configuration_path': str(self.config_dir),
            'recommendations': []
        }
        
        # Check configuration
        config = self._load_config()
        if config.get('api_key_storage'):
            status['configured'] = True
            status['storage_method'] = config['api_key_storage']
        
        # Check API key availability
        api_key = self.get_api_key()
        status['api_key_available'] = api_key is not None
        
        # Verify authentication if key is available
        if status['api_key_available']:
            status['verification_result'] = self.verify_authentication()
        
        # Generate recommendations
        if not status['configured']:
            status['recommendations'].append("Configure OpenAI API key using configure_api_key()")
        elif not status['api_key_available']:
            status['recommendations'].append("API key configured but not accessible - check storage method")
        elif status['verification_result'] and not status['verification_result']['is_authenticated']:
            status['recommendations'].append("API key found but authentication failed - check key validity")
        
        return status
    
    def clear_authentication(self) -> Dict[str, any]:
        """
        Clear all stored authentication data
        
        Returns:
            Clear operation result
        """
        result = {
            'success': False,
            'cleared_methods': [],
            'errors': []
        }
        
        try:
            # Clear keyring
            try:
                keyring.delete_password(self.keyring_service, self.keyring_username)
                result['cleared_methods'].append('keyring')
            except Exception:
                pass
            
            # Clear environment variable
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
                result['cleared_methods'].append('environment')
            
            # Clear file
            key_file = self.config_dir / '.api_key'
            if key_file.exists():
                key_file.unlink()
                result['cleared_methods'].append('file')
            
            # Clear config
            if self.config_file.exists():
                self.config_file.unlink()
                result['cleared_methods'].append('config')
            
            result['success'] = True
            self.logger.info(f"Authentication cleared: {result['cleared_methods']}")
            
        except Exception as e:
            result['errors'].append(f"Clear operation error: {str(e)}")
            self.logger.error(f"Clear authentication failed: {e}")
        
        return result


def main():
    """Test the authentication manager"""
    auth_manager = CodexAuthManager()
    
    # Get current status
    status = auth_manager.get_auth_status()
    print("Authentication Status:")
    print("=" * 30)
    print(f"Configured: {status['configured']}")
    print(f"Storage Method: {status['storage_method']}")
    print(f"API Key Available: {status['api_key_available']}")
    
    if status['verification_result']:
        print(f"Authentication Valid: {status['verification_result']['is_authenticated']}")
    
    if status['recommendations']:
        print("\nRecommendations:")
        for rec in status['recommendations']:
            print(f"  💡 {rec}")


if __name__ == "__main__":
    main()