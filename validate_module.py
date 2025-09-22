#!/usr/bin/env python3
"""
Simple validation test for module readiness.
Tests the key scenarios needed for Node.js integration.
"""

import json
import subprocess
import sys
from pathlib import Path

# Add the repo to Python path
repo_path = Path(__file__).parent
sys.path.insert(0, str(repo_path))

def test_module_import():
    """Test basic module import."""
    print("Testing module import...")
    try:
        import lightweight_rag
        assert hasattr(lightweight_rag, 'query_pdfs')
        assert hasattr(lightweight_rag, 'get_default_config')
        print("✓ Module import successful")
        return True
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False

def test_subprocess_json():
    """Test JSON subprocess interface."""
    print("Testing JSON subprocess interface...")
    
    input_data = {
        "query": "test",
        "config": {
            "paths": {"pdf_dir": "pdfs", "cache_dir": ".test_cache"},
            "rerank": {"final_top_k": 3}
        }
    }
    
    try:
        process = subprocess.Popen([
            sys.executable, '-m', 'lightweight_rag.cli_subprocess', '--json'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           cwd=str(repo_path), text=True)
        
        stdout, stderr = process.communicate(json.dumps(input_data), timeout=120)
        
        if stdout:
            # Try to find JSON in the output (may be mixed with progress messages)
            lines = stdout.strip().split('\n')
            for line in reversed(lines):  # Start from the end to find the JSON response
                if line.strip().startswith('{'):
                    try:
                        response = json.loads(line)
                        if 'success' in response and 'query' in response:
                            print(f"✓ JSON subprocess interface works: success={response['success']}")
                            return True
                    except json.JSONDecodeError:
                        continue
            
            print(f"✗ Could not find valid JSON response in output: {stdout[:200]}...")
            return False
        else:
            print(f"✗ No output from subprocess. Stderr: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Subprocess timed out")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"✗ Subprocess test failed: {e}")
        return False

def test_cli_interface():
    """Test CLI interface."""
    print("Testing CLI interface...")
    
    try:
        process = subprocess.Popen([
            sys.executable, '-m', 'lightweight_rag.cli_subprocess',
            '--query', 'test query', '--pdf_dir', 'pdfs', '--top_k', '2'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           cwd=str(repo_path), text=True)
        
        stdout, stderr = process.communicate(timeout=60)
        
        if stdout:
            # Try to find JSON in the output  
            lines = stdout.strip().split('\n')
            for line in reversed(lines):
                if line.strip().startswith('{'):
                    try:
                        response = json.loads(line)
                        if 'success' in response:
                            print(f"✓ CLI interface works: success={response['success']}")
                            return True
                    except json.JSONDecodeError:
                        continue
            
            print(f"✗ Could not find valid JSON in CLI output: {stdout[:200]}...")
            return False
        else:
            print(f"✗ No output from CLI. Stderr: {stderr}")
            return False
            
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run validation tests."""
    print("Lightweight RAG Module Readiness Validation")
    print("=" * 50)
    
    tests = [
        test_module_import,
        test_subprocess_json,
        test_cli_interface
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ Module is ready for Node.js integration!")
        print("\nUsage examples:")
        print("1. JSON subprocess:")
        print("   echo '{\"query\":\"test\"}' | python -m lightweight_rag.cli_subprocess")
        print("2. CLI interface:")
        print("   python -m lightweight_rag.cli_subprocess --query 'test' --pdf_dir pdfs")
        print("3. Direct import:")
        print("   import lightweight_rag; results = lightweight_rag.query_pdfs('test', config)")
        return True
    else:
        print("❌ Module has issues that need to be resolved.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)