/**
 * Node.js example of using lightweight_rag as a subprocess
 * 
 * This demonstrates how to integrate the Python RAG system into a Node.js application
 * with support for parallel searches and proper error handling.
 */

const { spawn } = require('child_process');
const path = require('path');

class LightweightRAG {
    constructor(options = {}) {
        this.pythonPath = options.pythonPath || 'python';
        this.ragModulePath = options.ragModulePath || path.join(__dirname, '..', 'lightweight_rag');
        this.defaultConfig = {
            paths: {
                pdf_dir: options.pdfDir || './pdfs',
                cache_dir: options.cacheDir || './.rag_cache'
            },
            rerank: {
                final_top_k: options.topK || 8
            },
            performance: {
                deterministic: true,
                numpy_seed: 42
            },
            ...options.config
        };
        this.timeout = options.timeout || 300000; // 5 minutes default
    }

    /**
     * Execute a single query
     * @param {string} query - The search query
     * @param {Object} config - Optional config overrides
     * @returns {Promise<Object>} - Search results
     */
    async query(query, config = {}) {
        return new Promise((resolve, reject) => {
            const mergedConfig = { ...this.defaultConfig, ...config };
            const input = {
                query: query,
                config: mergedConfig
            };

            const process = spawn(this.pythonPath, ['-m', 'lightweight_rag.cli_subprocess', '--json'], {
                cwd: path.dirname(this.ragModulePath),
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            const timer = setTimeout(() => {
                process.kill();
                reject(new Error(`Query timeout after ${this.timeout}ms`));
            }, this.timeout);

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                clearTimeout(timer);
                
                try {
                    const result = JSON.parse(stdout);
                    if (result.success) {
                        resolve({
                            query: result.query,
                            results: result.results,
                            count: result.count
                        });
                    } else {
                        reject(new Error(`RAG query failed: ${result.error}`));
                    }
                } catch (e) {
                    reject(new Error(`Failed to parse RAG response: ${e.message}. Stdout: ${stdout}, Stderr: ${stderr}`));
                }
            });

            process.on('error', (error) => {
                clearTimeout(timer);
                reject(new Error(`Process error: ${error.message}`));
            });

            // Send input
            process.stdin.write(JSON.stringify(input));
            process.stdin.end();
        });
    }

    /**
     * Execute multiple queries in parallel
     * @param {Array<string|Object>} queries - Array of queries or query objects
     * @param {Object} defaultConfig - Default config for all queries
     * @returns {Promise<Array>} - Array of results
     */
    async queryParallel(queries, defaultConfig = {}) {
        const promises = queries.map(queryItem => {
            if (typeof queryItem === 'string') {
                return this.query(queryItem, defaultConfig);
            } else if (typeof queryItem === 'object' && queryItem.query) {
                const config = { ...defaultConfig, ...queryItem.config };
                return this.query(queryItem.query, config);
            } else {
                return Promise.reject(new Error('Invalid query format'));
            }
        });

        return Promise.allSettled(promises);
    }

    /**
     * Execute queries in batches to limit concurrency
     * @param {Array} queries - Array of queries
     * @param {number} batchSize - Number of concurrent queries
     * @param {Object} config - Default config
     * @returns {Promise<Array>} - All results
     */
    async queryBatched(queries, batchSize = 3, config = {}) {
        const results = [];
        
        for (let i = 0; i < queries.length; i += batchSize) {
            const batch = queries.slice(i, i + batchSize);
            const batchResults = await this.queryParallel(batch, config);
            results.push(...batchResults);
        }
        
        return results;
    }
}

// Example usage
async function example() {
    try {
        const rag = new LightweightRAG({
            pdfDir: './pdfs',
            cacheDir: './.rag_cache',
            topK: 5,
            timeout: 120000 // 2 minutes
        });

        console.log('=== Single Query Example ===');
        const singleResult = await rag.query('machine learning applications');
        console.log(`Found ${singleResult.count} results for: "${singleResult.query}"`);
        singleResult.results.forEach((result, idx) => {
            console.log(`${idx + 1}. ${result.text.substring(0, 100)}... (Score: ${result.score})`);
        });

        console.log('\n=== Parallel Queries Example ===');
        const queries = [
            'cancel culture',
            'social media impact',
            'academic freedom'
        ];
        
        const parallelResults = await rag.queryParallel(queries);
        parallelResults.forEach((result, idx) => {
            if (result.status === 'fulfilled') {
                console.log(`Query "${queries[idx]}": ${result.value.count} results`);
            } else {
                console.error(`Query "${queries[idx]}" failed:`, result.reason.message);
            }
        });

        console.log('\n=== Batched Queries Example ===');
        const manyQueries = [
            'artificial intelligence',
            'machine learning ethics',
            'digital transformation',
            'social justice',
            'academic research',
            'data science methods'
        ];
        
        const batchedResults = await rag.queryBatched(manyQueries, 2);
        batchedResults.forEach((result, idx) => {
            if (result.status === 'fulfilled') {
                console.log(`Batched query ${idx + 1}: ${result.value.count} results`);
            } else {
                console.error(`Batched query ${idx + 1} failed:`, result.reason.message);
            }
        });

    } catch (error) {
        console.error('Example failed:', error);
    }
}

// Export the class for use in other modules
module.exports = LightweightRAG;

// Run example if this file is executed directly
if (require.main === module) {
    example();
}