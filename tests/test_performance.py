"""Tests for performance improvements and robustness features."""

import os
import pytest
from unittest.mock import patch, MagicMock
from lightweight_rag.performance import (
    seed_numpy, get_optimal_worker_count, create_api_semaphore,
    deterministic_sort_key, sort_results_deterministically,
    process_with_thread_pool
)


class TestThreadPool:
    """Test the thread pool utility with progress callbacks."""

    def test_process_with_thread_pool_progress(self):
        """Test that the on_progress callback is called correctly."""
        items = list(range(10))
        processed = []
        progress_calls = []

        def worker(item):
            return item * 2

        def on_progress(completed, total):
            progress_calls.append((completed, total))

        results = process_with_thread_pool(worker, items, max_workers=2, on_progress=on_progress)

        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        assert len(progress_calls) == 10
        # Final call should be (10, 10)
        assert progress_calls[-1] == (10, 10)
        # Ensure they are incremental
        completed_counts = [c for c, t in progress_calls]
        assert sorted(completed_counts) == completed_counts


class TestNumpySeeding:
    """Test numpy seeding functionality."""
    
    @patch('lightweight_rag.performance.HAS_NUMPY', True)
    @patch('lightweight_rag.performance.np')
    def test_seed_numpy_with_numpy_available(self, mock_np):
        """Test that numpy is seeded when available."""
        seed_numpy(42)
        mock_np.random.seed.assert_called_once_with(42)
    
    @patch('lightweight_rag.performance.HAS_NUMPY', False)
    def test_seed_numpy_without_numpy(self):
        """Test that seeding works without numpy installed."""
        # Should not raise an exception
        seed_numpy(42)
    
    def test_seed_numpy_with_none(self):
        """Test that None seed is ignored."""
        with patch('lightweight_rag.performance.np') as mock_np:
            seed_numpy(None)
            mock_np.random.seed.assert_not_called()


class TestWorkerCount:
    """Test optimal worker count calculation."""
    
    @patch('os.cpu_count', return_value=8)
    def test_get_optimal_worker_count_normal(self, mock_cpu_count):
        """Test normal CPU count detection."""
        assert get_optimal_worker_count() == 8
    
    @patch('os.cpu_count', return_value=None)
    def test_get_optimal_worker_count_none(self, mock_cpu_count):
        """Test fallback when cpu_count returns None."""
        assert get_optimal_worker_count() == 4
    
    @patch('os.cpu_count', side_effect=Exception("CPU detection failed"))
    def test_get_optimal_worker_count_exception(self, mock_cpu_count):
        """Test fallback when cpu_count raises exception."""
        assert get_optimal_worker_count() == 4


class TestApiSemaphore:
    """Test API semaphore creation."""
    
    def test_create_api_semaphore_default(self):
        """Test semaphore creation with default limit."""
        semaphore = create_api_semaphore()
        assert semaphore._value == 5
    
    def test_create_api_semaphore_custom(self):
        """Test semaphore creation with custom limit."""
        semaphore = create_api_semaphore(10)
        assert semaphore._value == 10


class TestDeterministicSorting:
    """Test deterministic sorting functionality."""
    
    def test_deterministic_sort_key_complete_item(self):
        """Test sort key generation with complete item data."""
        item = {
            'score': 0.85,
            'page': 5,
            'source': 'paper.pdf',
            'doc_id': 'doc_1'
        }
        key = deterministic_sort_key(item)
        expected = (-0.85, 5, 'paper.pdf', 'doc_1')
        assert key == expected
    
    def test_deterministic_sort_key_missing_fields(self):
        """Test sort key generation with missing fields."""
        item = {'score': 0.75}
        key = deterministic_sort_key(item)
        expected = (-0.75, 0, '', '')
        assert key == expected
    
    def test_sort_results_deterministically(self):
        """Test complete deterministic sorting."""
        results = [
            {'score': 0.8, 'page': 2, 'source': 'b.pdf', 'doc_id': '2'},  # idx 0
            {'score': 0.9, 'page': 1, 'source': 'a.pdf', 'doc_id': '1'},  # idx 1
            {'score': 0.8, 'page': 1, 'source': 'b.pdf', 'doc_id': '2'},  # idx 2
            {'score': 0.8, 'page': 2, 'source': 'a.pdf', 'doc_id': '1'},  # idx 3
        ]
        
        sorted_results = sort_results_deterministically(results)
        
        # Expected order: highest score first, then by page (asc), then source (asc), then doc_id (asc)
        # idx 1: score=0.9, page=1, source='a.pdf', doc_id='1'
        # idx 2: score=0.8, page=1, source='b.pdf', doc_id='2'
        # idx 3: score=0.8, page=2, source='a.pdf', doc_id='1' 
        # idx 0: score=0.8, page=2, source='b.pdf', doc_id='2'
        expected_order = [1, 2, 3, 0]  # indices of original results
        for i, expected_idx in enumerate(expected_order):
            assert sorted_results[i] == results[expected_idx]
    
    def test_sort_results_score_priority(self):
        """Test that higher scores come first."""
        results = [
            {'score': 0.7, 'page': 1, 'source': 'a.pdf', 'doc_id': '1'},
            {'score': 0.9, 'page': 1, 'source': 'a.pdf', 'doc_id': '1'},
        ]
        
        sorted_results = sort_results_deterministically(results)
        assert sorted_results[0]['score'] == 0.9
        assert sorted_results[1]['score'] == 0.7
    
    def test_sort_results_page_tiebreaker(self):
        """Test that earlier pages break ties."""
        results = [
            {'score': 0.8, 'page': 3, 'source': 'a.pdf', 'doc_id': '1'},
            {'score': 0.8, 'page': 1, 'source': 'a.pdf', 'doc_id': '1'},
        ]
        
        sorted_results = sort_results_deterministically(results)
        assert sorted_results[0]['page'] == 1
        assert sorted_results[1]['page'] == 3
    
    def test_sort_results_source_tiebreaker(self):
        """Test that lexicographic source order breaks ties."""
        results = [
            {'score': 0.8, 'page': 1, 'source': 'z.pdf', 'doc_id': '1'},
            {'score': 0.8, 'page': 1, 'source': 'a.pdf', 'doc_id': '1'},
        ]
        
        sorted_results = sort_results_deterministically(results)
        assert sorted_results[0]['source'] == 'a.pdf'
        assert sorted_results[1]['source'] == 'z.pdf'