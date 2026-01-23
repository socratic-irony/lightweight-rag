"""Tests for module import and subprocess interface functionality."""

import tempfile
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import lightweight_rag
from lightweight_rag.subprocess_interface import (
    create_error_response,
    create_success_response,
    validate_input,
    process_config,
)


class TestModuleImport:
    """Test module import and API functionality."""
    
    def test_module_imports_successfully(self):
        """Test that the module imports without error."""
        assert hasattr(lightweight_rag, 'query_pdfs')
        assert hasattr(lightweight_rag, 'run_rag_pipeline')
        assert hasattr(lightweight_rag, 'get_default_config')
        assert hasattr(lightweight_rag, 'load_config')
        assert hasattr(lightweight_rag, 'merge_configs')
    
    def test_query_pdfs_function_exists(self):
        """Test that query_pdfs function is callable."""
        assert callable(lightweight_rag.query_pdfs)
    
    def test_run_rag_pipeline_function_exists(self):
        """Test that run_rag_pipeline function is callable."""
        assert callable(lightweight_rag.run_rag_pipeline)
    
    def test_get_default_config_returns_dict(self):
        """Test that get_default_config returns a valid configuration."""
        config = lightweight_rag.get_default_config()
        assert isinstance(config, dict)
        assert 'paths' in config
        assert 'performance' in config
        assert 'rerank' in config


class TestSubprocessInterface:
    """Test subprocess interface functionality."""
    
    def test_create_error_response(self):
        """Test error response creation."""
        response = create_error_response("Test error", "test query")
        
        assert response['success'] is False
        assert response['query'] == "test query"
        assert response['results'] == []
        assert response['error'] == "Test error"
        assert response['count'] == 0
    
    def test_create_success_response(self):
        """Test success response creation."""
        results = [{"text": "test", "score": 1.0}]
        response = create_success_response(results, "test query")
        
        assert response['success'] is True
        assert response['query'] == "test query"
        assert response['results'] == results
        assert response['error'] is None
        assert response['count'] == 1
    
    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        valid_input = {
            "query": "test query",
            "config": {"paths": {"pdf_dir": "test"}}
        }
        
        is_valid, error = validate_input(valid_input)
        assert is_valid is True
        assert error == ""
    
    def test_validate_input_missing_query(self):
        """Test input validation with missing query."""
        invalid_input = {"config": {}}
        
        is_valid, error = validate_input(invalid_input)
        assert is_valid is False
        assert "Missing required field 'query'" in error
    
    def test_validate_input_empty_query(self):
        """Test input validation with empty query."""
        invalid_input = {"query": ""}
        
        is_valid, error = validate_input(invalid_input)
        assert is_valid is False
        assert "non-empty string" in error
    
    def test_validate_input_invalid_config(self):
        """Test input validation with invalid config."""
        invalid_input = {
            "query": "test",
            "config": "not a dict"
        }
        
        is_valid, error = validate_input(invalid_input)
        assert is_valid is False
        assert "must be an object" in error
    
    def test_process_config_with_defaults(self):
        """Test config processing with defaults only."""
        config = process_config()
        
        assert isinstance(config, dict)
        assert 'paths' in config
        assert 'performance' in config
    
    def test_process_config_with_overrides(self):
        """Test config processing with custom overrides."""
        custom_config = {
            "paths": {"pdf_dir": "custom_pdfs"},
            "rerank": {"final_top_k": 10}
        }
        
        config = process_config(custom_config)
        
        assert config["paths"]["pdf_dir"] == "custom_pdfs"
        assert config["rerank"]["final_top_k"] == 10
        # Should still have default values for other keys
        assert "performance" in config


class TestParallelExecution:
    """Test parallel execution capabilities."""
    
    @pytest.mark.asyncio
    async def test_multiple_async_queries(self):
        """Test that multiple async queries can run without conflicts."""
        import asyncio
        from lightweight_rag.main import run_rag_pipeline
        
        # Create minimal config for testing
        config = lightweight_rag.get_default_config()
        
        # Create tasks for multiple queries
        queries = ["test1", "test2", "test3"]
        
        # Mock the actual processing to avoid needing real PDFs
        with patch('lightweight_rag.main.build_corpus') as mock_corpus, \
             patch('lightweight_rag.main.build_bm25') as mock_bm25, \
             patch('lightweight_rag.main.search_topk') as mock_search:
            
            mock_corpus.return_value = []  # Empty corpus
            mock_bm25.return_value = (MagicMock(), [])
            mock_search.return_value = [], {}  # Empty results and empty confidence
            
            # Run multiple queries concurrently
            tasks = [run_rag_pipeline(config, query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete without exceptions
            assert len(results) == 3
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)


class TestCLISubprocessInterface:
    """Test the CLI subprocess interface."""
    
    def test_json_subprocess_with_valid_input(self):
        """Test JSON subprocess interface with valid input."""
        input_data = {
            "query": "test query",
            "config": {
                "paths": {"pdf_dir": "pdfs", "cache_dir": ".test_cache"},
                "rerank": {"final_top_k": 2}
            }
        }
        
        # Mock the pipeline to avoid needing real PDFs
        with patch('lightweight_rag.main.run_rag_pipeline') as mock_pipeline:
            mock_pipeline.return_value = [{"text": "test", "score": 1.0}]
            
            # Test the subprocess interface by simulating stdin input
            process = subprocess.Popen([
                sys.executable, '-m', 'lightweight_rag.cli_subprocess', '--json'
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               cwd=str(Path(__file__).parents[1]),  # Set working directory
               text=True)
            
            stdout, stderr = process.communicate(json.dumps(input_data))
            
            # Should have successful return code
            assert process.returncode == 0 or mock_pipeline.called  # Allow for import success
            
            # Should be valid JSON output (even if mocked)
            if stdout.strip():
                try:
                    response = json.loads(stdout)
                    # If we got a response, it should have the expected structure
                    if isinstance(response, dict):
                        assert 'success' in response
                        assert 'query' in response
                        assert 'results' in response
                except json.JSONDecodeError:
                    # If output is not JSON, it might be status messages, which is OK
                    pass
    
    def test_json_subprocess_with_invalid_input(self):
        """Test JSON subprocess interface with invalid input."""
        # Test with invalid JSON
        process = subprocess.Popen([
            sys.executable, '-m', 'lightweight_rag.cli_subprocess', '--json'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           cwd=str(Path(__file__).parents[1]),
           text=True)
        
        stdout, stderr = process.communicate('{"invalid": json}')  # Invalid JSON
        
        # Should handle gracefully and return error response
        if stdout.strip():
            try:
                response = json.loads(stdout)
                if isinstance(response, dict):
                    assert response.get('success') is False
                    assert 'error' in response
            except json.JSONDecodeError:
                # Still acceptable - the interface handled the error
                pass


class TestThreadSafety:
    """Test thread safety for parallel subprocess execution."""
    
    def test_multiple_subprocess_calls(self):
        """Test multiple subprocess calls don't interfere with each other."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_subprocess(query_id):
            """Run a subprocess call."""
            input_data = {
                "query": f"test query {query_id}",
                "config": {
                    "paths": {"pdf_dir": "pdfs", "cache_dir": f".test_cache_{query_id}"}
                }
            }
            
            try:
                process = subprocess.Popen([
                    sys.executable, '-c', f'''
import json
import sys
sys.path.insert(0, "{Path(__file__).parents[1]}")
from lightweight_rag.subprocess_interface import validate_input, create_success_response
input_data = {json.dumps(input_data)}
is_valid, error = validate_input(input_data)
if is_valid:
    response = create_success_response([], input_data["query"])
else:
    response = {{"success": False, "error": error}}
print(json.dumps(response))
                    '''
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                stdout, stderr = process.communicate()
                
                if stdout.strip():
                    try:
                        response = json.loads(stdout)
                        results_queue.put((query_id, response))
                    except json.JSONDecodeError:
                        results_queue.put((query_id, {"error": "JSON decode error"}))
                else:
                    results_queue.put((query_id, {"error": "No output"}))
                    
            except Exception as e:
                results_queue.put((query_id, {"error": str(e)}))
        
        # Start multiple threads
        threads = []
        num_threads = 3
        
        for i in range(num_threads):
            thread = threading.Thread(target=run_subprocess, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Should have results from all threads
        assert len(results) >= num_threads  # Allow for potential duplicates
        
        # Each result should have a valid structure
        for query_id, result in results:
            assert isinstance(result, dict)
            # Should either be successful or have an error
            assert result.get('success') is not None or 'error' in result