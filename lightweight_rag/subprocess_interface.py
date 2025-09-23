#!/usr/bin/env python3
"""
JSON subprocess interface for lightweight_rag.

This provides a simple JSON-based interface for external programs (like Node.js) 
to use the lightweight_rag module via subprocess calls.
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from .config import get_default_config, load_config, merge_configs
from .main import run_rag_pipeline
from .environment import adapt_config_for_environment


def create_error_response(error_message: str, query: str = None) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "success": False,
        "query": query,
        "results": [],
        "error": error_message,
        "count": 0
    }


def create_success_response(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Create a standardized success response."""
    return {
        "success": True,
        "query": query,
        "results": results,
        "error": None,
        "count": len(results)
    }


def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate input JSON structure."""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    if "query" not in data:
        return False, "Missing required field 'query'"
    
    if not isinstance(data["query"], str) or not data["query"].strip():
        return False, "Field 'query' must be a non-empty string"
    
    if "config" in data and not isinstance(data["config"], dict):
        return False, "Field 'config' must be an object if provided"
    
    # Validate config structure if provided
    if "config" in data:
        config = data["config"]
        
        # Validate paths if provided
        if "paths" in config:
            if not isinstance(config["paths"], dict):
                return False, "config.paths must be an object"
            
            paths = config["paths"]
            for path_key in ["pdf_dir", "cache_dir"]:
                if path_key in paths and not isinstance(paths[path_key], str):
                    return False, f"config.paths.{path_key} must be a string"
        
        # Validate rerank settings if provided
        if "rerank" in config:
            if not isinstance(config["rerank"], dict):
                return False, "config.rerank must be an object"
            
            rerank = config["rerank"]
            if "final_top_k" in rerank:
                if not isinstance(rerank["final_top_k"], int) or rerank["final_top_k"] < 1:
                    return False, "config.rerank.final_top_k must be a positive integer"
    
    return True, ""


def process_config(config_data: Dict[str, Any] = None, config_file: str = None) -> Dict[str, Any]:
    """Process configuration with proper defaults and merging."""
    # Start with defaults
    cfg = get_default_config()
    
    # Load from file if specified
    if config_file:
        try:
            file_cfg = load_config(config_file)
            cfg = merge_configs(cfg, file_cfg)
        except Exception as e:
            # Don't fail on config file errors, just use defaults
            pass
    
    # Merge provided config
    if config_data:
        cfg = merge_configs(cfg, config_data)
    
    # Adapt configuration for current environment
    cfg = adapt_config_for_environment(cfg)
    
    # Set quiet mode for subprocess usage
    cfg["_quiet_mode"] = True
    
    # Normalize and ensure paths exist
    try:
        pdf_dir = Path(cfg["paths"]["pdf_dir"]).expanduser().resolve()
        pdf_dir.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["pdf_dir"] = str(pdf_dir)
    except (OSError, PermissionError) as e:
        # If we can't create the PDF directory, that might be okay - it could be read-only
        pdf_dir = Path(cfg["paths"]["pdf_dir"]).expanduser().resolve()
        cfg["paths"]["pdf_dir"] = str(pdf_dir)
    
    try:
        cache_dir = Path(cfg["paths"]["cache_dir"]).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg["paths"]["cache_dir"] = str(cache_dir)
    except (OSError, PermissionError) as e:
        # Cache directory creation failure is more serious, but let's handle it gracefully
        import tempfile
        cache_dir = Path(tempfile.mkdtemp(prefix="rag_cache_"))
        cfg["paths"]["cache_dir"] = str(cache_dir)
    
    return cfg


async def process_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single query and return formatted response."""
    try:
        # In subprocess mode, suppress all output except the final JSON
        import sys
        import os
        
        # Redirect both stdout and stderr to devnull to suppress all output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            with open(os.devnull, 'w') as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                results = await run_rag_pipeline(config, query)
            return create_success_response(results, query)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
                
    except Exception as e:
        return create_error_response(str(e), query)


def main():
    """Main entry point for subprocess interface."""
    try:
        # Read JSON from stdin
        input_data = json.load(sys.stdin)
        
        # Validate input
        is_valid, error_msg = validate_input(input_data)
        if not is_valid:
            response = create_error_response(error_msg)
            print(json.dumps(response))
            sys.exit(1)
        
        query = input_data["query"]
        config_data = input_data.get("config", {})
        config_file = input_data.get("config_file")
        
        # Process configuration
        config = process_config(config_data, config_file)
        
        # Process query
        response = asyncio.run(process_query(query, config))
        
        # Output response
        print(json.dumps(response, ensure_ascii=False))
        
        # Exit with appropriate code
        sys.exit(0 if response["success"] else 1)
        
    except json.JSONDecodeError as e:
        response = create_error_response(f"Invalid JSON input: {str(e)}")
        print(json.dumps(response))
        sys.exit(1)
        
    except KeyboardInterrupt:
        response = create_error_response("Process interrupted")
        print(json.dumps(response))
        sys.exit(1)
        
    except Exception as e:
        response = create_error_response(f"Unexpected error: {str(e)}")
        print(json.dumps(response))
        sys.exit(1)


if __name__ == "__main__":
    main()