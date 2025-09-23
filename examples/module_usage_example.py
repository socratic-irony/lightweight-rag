#!/usr/bin/env python3
"""
Example script demonstrating lightweight_rag module usage.

This example shows how to use lightweight_rag as a Python module in other projects.
"""

import asyncio
import lightweight_rag


async def main():
    """Demonstrate basic lightweight_rag usage."""
    
    # Get default configuration
    config = lightweight_rag.get_default_config()
    
    # Customize configuration for your needs
    config["paths"]["pdf_dir"] = "./pdfs"  # Your PDF directory
    config["paths"]["cache_dir"] = "./.rag_cache"  # Cache directory
    config["rerank"]["final_top_k"] = 5  # Return top 5 results
    
    # Enable RRF (Reciprocal Rank Fusion) for better results
    config["fusion"]["rrf"]["enabled"] = True
    config["fusion"]["robust_query"]["enabled"] = True
    
    # Example query
    query = "machine learning algorithms"
    
    try:
        # Run the RAG pipeline
        print(f"Searching for: '{query}'")
        print("=" * 50)
        
        results = await lightweight_rag.run_rag_pipeline(config, query)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('citation', 'Unknown citation')}")
                print(f"   Score: {result.get('score', 0):.3f}")
                text = result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
                print(f"   Text: {text}")
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have PDFs in the ./pdfs directory")


if __name__ == "__main__":
    print("Lightweight RAG Module Example")
    print("==============================")
    asyncio.run(main())