#!/usr/bin/env python3
"""
Environment detection and adaptation utilities for lightweight_rag.

This module helps the system adapt to different environments like 
GitHub Codespaces, local development, and deployed environments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def detect_environment() -> Dict[str, Any]:
    """
    Detect the current runtime environment and return environment info.
    
    Returns:
        Dict containing environment information:
        - type: 'codespace', 'local', 'docker', 'unknown'
        - platform: 'linux', 'darwin', 'windows'
        - working_dir: Current working directory
        - home_dir: User home directory
        - temp_dir: Temp directory path
        - is_interactive: Whether running interactively
    """
    env_info = {
        'type': 'unknown',
        'platform': sys.platform,
        'working_dir': str(Path.cwd()),
        'home_dir': str(Path.home()),
        'temp_dir': '/tmp' if os.name == 'posix' else os.getenv('TEMP', '/tmp'),
        'is_interactive': sys.stdin.isatty(),
    }
    
    # Detect GitHub Codespaces
    if os.getenv('CODESPACES') == 'true':
        env_info['type'] = 'codespace'
        env_info['workspace_dir'] = os.getenv('GITHUB_WORKSPACE', '/workspaces')
        
    # Detect Docker container
    elif os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
        env_info['type'] = 'docker'
        
    # Detect various CI environments
    elif any(os.getenv(var) for var in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL']):
        env_info['type'] = 'ci'
        
    # Otherwise assume local development
    else:
        env_info['type'] = 'local'
    
    return env_info


def get_default_paths(env_info: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Get default paths that work well in the current environment.
    
    Args:
        env_info: Environment info from detect_environment(), auto-detected if None
        
    Returns:
        Dict with 'pdf_dir' and 'cache_dir' paths
    """
    if env_info is None:
        env_info = detect_environment()
    
    paths = {}
    
    if env_info['type'] == 'codespace':
        # In Codespaces, use workspace-relative paths
        workspace = Path(env_info.get('workspace_dir', '/workspaces'))
        if workspace.exists():
            paths['pdf_dir'] = str(workspace / 'pdfs')
            paths['cache_dir'] = str(workspace / '.rag_cache')
        else:
            # Fallback to current directory
            paths['pdf_dir'] = 'pdfs'
            paths['cache_dir'] = '.rag_cache'
            
    elif env_info['type'] == 'docker':
        # In Docker, prefer /data if it exists, otherwise use /tmp for cache
        if Path('/data').exists():
            paths['pdf_dir'] = '/data/pdfs'
            paths['cache_dir'] = '/data/.rag_cache'
        else:
            paths['pdf_dir'] = 'pdfs'
            paths['cache_dir'] = '/tmp/.rag_cache'
            
    elif env_info['type'] == 'ci':
        # In CI, use temp directory for cache to avoid permission issues
        temp_dir = Path(env_info['temp_dir'])
        paths['pdf_dir'] = 'pdfs'
        paths['cache_dir'] = str(temp_dir / '.rag_cache')
        
    else:
        # Local development - use current directory
        paths['pdf_dir'] = 'pdfs'
        paths['cache_dir'] = '.rag_cache'
    
    return paths


def adapt_config_for_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt configuration for the current environment.
    
    This function modifies paths and settings to work better in different environments.
    """
    env_info = detect_environment()
    
    # If paths aren't set or are defaults, use environment-appropriate paths
    if not config.get('paths'):
        config['paths'] = {}
    
    paths = config['paths']
    default_paths = get_default_paths(env_info)
    
    # Use environment defaults if current paths don't exist or are generic defaults
    current_pdf_dir = Path(paths.get('pdf_dir', ''))
    current_cache_dir = Path(paths.get('cache_dir', ''))
    
    if not current_pdf_dir.exists() and paths.get('pdf_dir') in [None, 'pdfs']:
        paths['pdf_dir'] = default_paths['pdf_dir']
        
    if not current_cache_dir.exists() and paths.get('cache_dir') in [None, '.rag_cache', '.raq_cache']:
        paths['cache_dir'] = default_paths['cache_dir']
    
    # Adjust performance settings for environment
    if not config.get('performance'):
        config['performance'] = {}
    
    performance = config['performance']
    
    # In CI environments, be more conservative with resources
    if env_info['type'] == 'ci':
        performance.setdefault('api_semaphore_size', 2)  # Lower concurrency
        performance.setdefault('pdf_thread_workers', 1)  # Single threaded
        
    # In Codespaces, moderate resource usage
    elif env_info['type'] == 'codespace':
        performance.setdefault('api_semaphore_size', 3)
        performance.setdefault('pdf_thread_workers', 2)
        
    # Docker and local can use more resources
    else:
        performance.setdefault('api_semaphore_size', 5)
        # pdf_thread_workers will auto-detect if None
    
    return config


def is_running_in_subprocess() -> bool:
    """Check if we're running as a subprocess (vs. interactive shell)."""
    return not sys.stdin.isatty() or '--json' in sys.argv


def get_environment_info() -> str:
    """Get a human-readable description of the current environment."""
    env_info = detect_environment()
    
    desc = f"Environment: {env_info['type']} on {env_info['platform']}"
    if env_info['type'] == 'codespace':
        desc += " (GitHub Codespaces)"
    elif env_info['type'] == 'docker':
        desc += " (Docker container)"
    elif env_info['type'] == 'ci':
        desc += " (CI/CD environment)"
    
    return desc