#!/usr/bin/env python3
"""Tests for configuration system in lightweight-rag."""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    get_default_config, load_config, apply_env_vars,
    apply_cli_overrides, load_full_config, merge_configs
)


class TestDefaultConfig:
    """Test default configuration."""
    
    def test_default_config_structure(self):
        """Test that default config has expected structure."""
        config = get_default_config()
        
        # Check top-level keys
        expected_keys = ["paths", "indexing", "bm25", "prf", "bonuses", 
                        "diversity", "rerank", "citations", "output"]
        for key in expected_keys:
            assert key in config
        
        # Check some specific values
        assert config["paths"]["pdf_dir"] == "pdfs"
        assert config["bm25"]["k1"] == 1.5
        assert config["prf"]["enabled"] is False
        assert config["bonuses"]["proximity"]["enabled"] is True


class TestConfigFile:
    """Test configuration file loading."""
    
    def test_load_existing_config_file(self):
        """Test loading an existing config file."""
        # Create a temporary config file
        config_data = {
            "paths": {"pdf_dir": "test_pdfs"},
            "bm25": {"k1": 2.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            assert result["paths"]["pdf_dir"] == "test_pdfs"
            assert result["bm25"]["k1"] == 2.0
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_config_file(self):
        """Test loading a non-existent config file returns defaults."""
        result = load_config("nonexistent.yaml")
        # Should return defaults since file doesn't exist
        assert result["paths"]["pdf_dir"] == "pdfs"  # Default value
    
    def test_load_invalid_yaml_file(self):
        """Test loading an invalid YAML file returns defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            # Should return defaults since YAML is invalid
            assert result["paths"]["pdf_dir"] == "pdfs"  # Default value
        finally:
            os.unlink(temp_file)
    
    def test_load_empty_config_file(self):
        """Test loading an empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_file = f.name
        
        try:
            result = load_config(temp_file)
            # Should return defaults
            assert result["paths"]["pdf_dir"] == "pdfs"  # Default value
        finally:
            os.unlink(temp_file)


class TestEnvOverrides:
    """Test environment variable overrides."""
    
    @patch.dict(os.environ, {"RAG_PATHS_PDF_DIR": "env_pdfs"})
    def test_env_override_simple(self):
        """Test simple environment variable override."""
        config = {"paths": {"pdf_dir": "default"}}
        result = apply_env_vars(config)
        assert result["paths"]["pdf_dir"] == "env_pdfs"
    
    @patch.dict(os.environ, {
        "RAG_BM25_K1": "2.5",
        "RAG_PRF_ENABLED": "true"
    })
    def test_env_override_multiple(self):
        """Test multiple environment variable overrides."""
        config = {
            "bm25": {"k1": 1.5},
            "prf": {"enabled": False}
        }
        result = apply_env_vars(config)
        assert result["bm25"]["k1"] == 2.5
        assert result["prf"]["enabled"] is True
    
    @patch.dict(os.environ, {"RAG_PATHS_CACHE_DIR": "null"})
    def test_env_override_null_value(self):
        """Test environment variable with null value."""
        config = {"paths": {"cache_dir": ".cache"}}
        result = apply_env_vars(config)
        # Since it's a string type, should set to "null" string, not None
        assert result["paths"]["cache_dir"] == "null"
    
    @patch.dict(os.environ, {"RAG_INVALID_PATH": "test"})
    def test_env_override_invalid_path(self):
        """Test environment variable with invalid path is ignored."""
        config = {"paths": {"pdf_dir": "default"}}
        result = apply_env_vars(config)
        assert result == config  # Should be unchanged
    
    @patch.dict(os.environ, {"OTHER_VAR": "test"})
    def test_env_override_non_rag_var_ignored(self):
        """Test non-RAG environment variables are ignored."""
        config = {"paths": {"pdf_dir": "default"}}
        result = apply_env_vars(config)
        assert result == config  # Should be unchanged


class TestCLIOverrides:
    """Test CLI argument overrides."""
    
    def test_cli_override_simple(self):
        """Test simple CLI override."""
        config = {"rerank": {"final_top_k": 8}}
        cli_args = Namespace(k=15)
        result = apply_cli_overrides(config, cli_args)
        assert result["rerank"]["final_top_k"] == 15
    
    def test_cli_override_pdf_dir(self):
        """Test PDF directory CLI override."""
        config = {"paths": {"pdf_dir": "default"}}
        cli_args = Namespace(pdf_dir="cli_pdfs")
        result = apply_cli_overrides(config, cli_args)
        assert result["paths"]["pdf_dir"] == "cli_pdfs"
    
    def test_cli_override_boolean_flags(self):
        """Test boolean flag overrides."""
        config = {
            "prf": {"enabled": False},
            "bonuses": {"proximity": {"enabled": True}},
            "diversity": {"enabled": True},
            "rerank": {"semantic": {"enabled": False}}
        }
        cli_args = Namespace(
            rm3=True,
            no_prox=True,
            no_diversity=True,
            semantic_rerank=True
        )
        result = apply_cli_overrides(config, cli_args)
        
        assert result["prf"]["enabled"] is True
        assert result["bonuses"]["proximity"]["enabled"] is False
        assert result["diversity"]["enabled"] is False
        assert result["rerank"]["semantic"]["enabled"] is True
    
    def test_cli_override_none_values_ignored(self):
        """Test that None CLI values are ignored."""
        config = {"bm25": {"k1": 1.5}}
        cli_args = Namespace(k=None)
        result = apply_cli_overrides(config, cli_args)
        assert result["bm25"]["k1"] == 1.5  # Should remain unchanged
    
    def test_cli_override_multiple_values(self):
        """Test multiple CLI overrides."""
        config = {
            "bonuses": {
                "proximity": {"window": 30, "weight": 0.2},
                "ngram": {"weight": 0.1}
            },
            "diversity": {"per_doc_penalty": 0.3, "max_per_doc": 2}
        }
        cli_args = Namespace(
            prox_window=50,
            prox_lambda=0.3,
            ngram_lambda=0.2,
            div_lambda=0.4,
            max_per_doc=3
        )
        result = apply_cli_overrides(config, cli_args)
        
        assert result["bonuses"]["proximity"]["window"] == 50
        assert result["bonuses"]["proximity"]["weight"] == 0.3
        assert result["bonuses"]["ngram"]["weight"] == 0.2
        assert result["diversity"]["per_doc_penalty"] == 0.4
        assert result["diversity"]["max_per_doc"] == 3


class TestFullConfigLoad:
    """Test complete configuration loading with precedence."""
    
    def test_full_config_with_defaults_only(self):
        """Test loading with defaults only (no file, no env, no CLI)."""
        with patch("config.load_config", return_value=get_default_config()):
            with patch.dict(os.environ, {}, clear=True):
                config = load_full_config("nonexistent.yaml", Namespace())
                
                # Should have all default values
                assert config["paths"]["pdf_dir"] == "pdfs"
                assert config["bm25"]["k1"] == 1.5
    
    def test_full_config_precedence(self):
        """Test configuration precedence: defaults < file < env < CLI."""
        # File config (already includes defaults)
        file_config = get_default_config()
        file_config["paths"]["pdf_dir"] = "file_pdfs"
        file_config["bm25"]["k1"] = 2.0
        
        # CLI args
        cli_args = Namespace(pdf_dir="cli_pdfs", k=None)  # k=None should not override
        
        with patch("config.load_config", return_value=file_config):
            with patch.dict(os.environ, {"RAG_BM25_K1": "3.0"}):
                config = load_full_config("test.yaml", cli_args)
                
                # CLI should win for pdf_dir
                assert config["paths"]["pdf_dir"] == "cli_pdfs"
                # Env should win for k1 (CLI is None)
                assert config["bm25"]["k1"] == 3.0
    
    def test_deep_merge_behavior(self):
        """Test that nested dictionaries are properly merged."""
        file_config = get_default_config()
        file_config["bonuses"]["proximity"]["window"] = 50
        file_config["bonuses"]["ngram"]["weight"] = 0.2
        
        with patch("config.load_config", return_value=file_config):
            with patch.dict(os.environ, {}, clear=True):
                config = load_full_config("test.yaml", Namespace())
                
                # Should have both file overrides and defaults
                assert config["bonuses"]["proximity"]["window"] == 50  # From file
                assert config["bonuses"]["proximity"]["enabled"] is True  # From defaults
                assert config["bonuses"]["proximity"]["weight"] == 0.2  # From defaults
                assert config["bonuses"]["ngram"]["weight"] == 0.2  # From file


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    def test_config_with_invalid_types(self):
        """Test that config handles invalid type conversions gracefully."""
        with patch.dict(os.environ, {
            "RAG_BM25_K1": "not_a_number",
            "RAG_PRF_ENABLED": "not_a_boolean"
        }):
            config = {"bm25": {"k1": 1.5}, "prf": {"enabled": False}}
            # This should raise an exception with invalid number conversion
            with pytest.raises(ValueError):
                apply_env_vars(config)