#!/usr/bin/env python3
"""Additional tests for environment module to improve coverage."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightweight_rag.environment import (
    detect_environment,
    get_default_paths,
    adapt_config_for_environment,
    is_running_in_subprocess,
    get_environment_info
)


class TestEnvironmentDetection:
    """Test environment detection in different contexts."""
    
    def test_detect_local_environment(self):
        """Test detection of local development environment."""
        with patch.dict(os.environ, {}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "local"
            assert "platform" in env_info
            assert "working_dir" in env_info
            assert "home_dir" in env_info
    
    def test_detect_codespace_environment(self):
        """Test detection of GitHub Codespaces environment."""
        with patch.dict(os.environ, {"CODESPACES": "true", "GITHUB_WORKSPACE": "/workspace"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "codespace"
            assert env_info.get("workspace_dir") == "/workspace"
    
    def test_detect_docker_environment(self):
        """Test detection of Docker container environment."""
        with patch.dict(os.environ, {"DOCKER_CONTAINER": "true"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "docker"
    
    def test_detect_docker_via_file(self):
        """Test detection of Docker via /.dockerenv file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                env_info = detect_environment()
                
                # Should detect as docker
                assert env_info["type"] in ["docker", "local"]
    
    def test_detect_ci_environment(self):
        """Test detection of CI/CD environment."""
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "ci"
    
    def test_detect_github_actions(self):
        """Test detection of GitHub Actions environment."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "ci"
    
    def test_detect_gitlab_ci(self):
        """Test detection of GitLab CI environment."""
        with patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "ci"
    
    def test_detect_jenkins(self):
        """Test detection of Jenkins environment."""
        with patch.dict(os.environ, {"JENKINS_URL": "http://jenkins"}, clear=True):
            env_info = detect_environment()
            
            assert env_info["type"] == "ci"
    
    def test_interactive_detection(self):
        """Test detection of interactive vs non-interactive mode."""
        env_info = detect_environment()
        
        assert "is_interactive" in env_info
        assert isinstance(env_info["is_interactive"], bool)


class TestPathResolution:
    """Test path resolution scenarios."""
    
    def test_default_paths_local(self):
        """Test default paths for local environment."""
        env_info = {"type": "local", "platform": "linux", "temp_dir": "/tmp"}
        paths = get_default_paths(env_info)
        
        assert "pdf_dir" in paths
        assert "cache_dir" in paths
        assert paths["pdf_dir"] == "pdfs"
        assert paths["cache_dir"] == ".rag_cache"
    
    def test_default_paths_codespace(self):
        """Test default paths for Codespace environment."""
        env_info = {
            "type": "codespace",
            "workspace_dir": "/workspaces",
            "temp_dir": "/tmp"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            paths = get_default_paths(env_info)
            
            assert "/workspaces" in paths["pdf_dir"]
            assert "/workspaces" in paths["cache_dir"]
    
    def test_default_paths_codespace_fallback(self):
        """Test default paths for Codespace when workspace doesn't exist."""
        env_info = {
            "type": "codespace",
            "workspace_dir": "/nonexistent",
            "temp_dir": "/tmp"
        }
        
        with patch('pathlib.Path.exists', return_value=False):
            paths = get_default_paths(env_info)
            
            assert paths["pdf_dir"] == "pdfs"
            assert paths["cache_dir"] == ".rag_cache"
    
    def test_default_paths_docker(self):
        """Test default paths for Docker environment."""
        env_info = {"type": "docker", "temp_dir": "/tmp"}
        
        with patch('pathlib.Path.exists', return_value=False):
            paths = get_default_paths(env_info)
            
            assert paths["pdf_dir"] == "pdfs"
            assert "/tmp" in paths["cache_dir"]
    
    def test_default_paths_docker_with_data_dir(self):
        """Test default paths for Docker with /data directory."""
        env_info = {"type": "docker", "temp_dir": "/tmp"}
        
        with patch('pathlib.Path.exists', return_value=True):
            paths = get_default_paths(env_info)
            
            assert "/data" in paths["pdf_dir"]
            assert "/data" in paths["cache_dir"]
    
    def test_default_paths_ci(self):
        """Test default paths for CI environment."""
        env_info = {"type": "ci", "temp_dir": "/tmp"}
        paths = get_default_paths(env_info)
        
        assert paths["pdf_dir"] == "pdfs"
        assert "/tmp" in paths["cache_dir"]
    
    def test_default_paths_auto_detect(self):
        """Test default paths with auto-detection."""
        paths = get_default_paths()
        
        assert "pdf_dir" in paths
        assert "cache_dir" in paths


class TestConfigAdaptation:
    """Test config adaptation logic."""
    
    def test_adapt_config_empty(self):
        """Test adapting empty config."""
        config = {}
        
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {"type": "local", "platform": "linux", "temp_dir": "/tmp"}
            adapted = adapt_config_for_environment(config)
            
            assert "paths" in adapted
            assert "performance" in adapted
    
    def test_adapt_config_ci_resources(self):
        """Test config adaptation for CI environment with conservative resources."""
        config = {}
        
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {"type": "ci", "platform": "linux", "temp_dir": "/tmp"}
            adapted = adapt_config_for_environment(config)
            
            assert adapted["performance"]["api_semaphore_size"] == 2
            assert adapted["performance"]["pdf_thread_workers"] == 1
    
    def test_adapt_config_codespace_resources(self):
        """Test config adaptation for Codespace environment."""
        config = {}
        
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {
                "type": "codespace",
                "workspace_dir": "/workspaces",
                "temp_dir": "/tmp"
            }
            adapted = adapt_config_for_environment(config)
            
            assert adapted["performance"]["api_semaphore_size"] == 3
            assert adapted["performance"]["pdf_thread_workers"] == 2
    
    def test_adapt_config_preserve_existing(self):
        """Test that existing config values are preserved."""
        config = {
            "paths": {"pdf_dir": "/custom/path"},
            "performance": {"api_semaphore_size": 10}
        }
        
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {"type": "local", "temp_dir": "/tmp"}
            
            # Mock path exists to preserve custom path
            with patch('pathlib.Path.exists', return_value=True):
                adapted = adapt_config_for_environment(config)
                
                # Custom values should be preserved
                assert adapted["paths"]["pdf_dir"] == "/custom/path"
                assert adapted["performance"]["api_semaphore_size"] == 10
    
    def test_adapt_config_path_fallback(self):
        """Test path adaptation when configured paths don't exist."""
        config = {
            "paths": {"pdf_dir": "pdfs", "cache_dir": ".rag_cache"}
        }
        
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {"type": "local", "temp_dir": "/tmp"}
            
            # Paths don't exist, should use defaults
            with patch('pathlib.Path.exists', return_value=False):
                adapted = adapt_config_for_environment(config)
                
                assert "pdf_dir" in adapted["paths"]
                assert "cache_dir" in adapted["paths"]


class TestSubprocessDetection:
    """Test subprocess detection."""
    
    def test_subprocess_detection_tty(self):
        """Test subprocess detection with TTY."""
        with patch('sys.stdin.isatty', return_value=True):
            with patch('sys.argv', ['script.py']):
                result = is_running_in_subprocess()
                
                # Interactive with no --json flag
                assert result is False
    
    def test_subprocess_detection_no_tty(self):
        """Test subprocess detection without TTY."""
        with patch('sys.stdin.isatty', return_value=False):
            result = is_running_in_subprocess()
            
            # Non-interactive
            assert result is True
    
    def test_subprocess_detection_json_flag(self):
        """Test subprocess detection with --json flag."""
        with patch('sys.stdin.isatty', return_value=True):
            with patch('sys.argv', ['script.py', '--json']):
                result = is_running_in_subprocess()
                
                # Has --json flag
                assert result is True


class TestEnvironmentInfo:
    """Test human-readable environment information."""
    
    def test_environment_info_local(self):
        """Test environment info for local environment."""
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {
                "type": "local",
                "platform": "darwin"
            }
            
            info = get_environment_info()
            
            assert "local" in info.lower()
            assert "darwin" in info.lower()
    
    def test_environment_info_codespace(self):
        """Test environment info for Codespace."""
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {
                "type": "codespace",
                "platform": "linux"
            }
            
            info = get_environment_info()
            
            assert "codespace" in info.lower() or "github" in info.lower()
    
    def test_environment_info_docker(self):
        """Test environment info for Docker."""
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {
                "type": "docker",
                "platform": "linux"
            }
            
            info = get_environment_info()
            
            assert "docker" in info.lower()
    
    def test_environment_info_ci(self):
        """Test environment info for CI/CD."""
        with patch('lightweight_rag.environment.detect_environment') as mock_detect:
            mock_detect.return_value = {
                "type": "ci",
                "platform": "linux"
            }
            
            info = get_environment_info()
            
            assert "ci" in info.lower() or "cd" in info.lower()
