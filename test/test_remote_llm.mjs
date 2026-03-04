#!/usr/bin/env node
/**
 * QMD Remote LLM Backend Tests
 * 
 * Tests the RemoteLlamaCpp integration that offloads embedding, reranking,
 * and query expansion to remote llama-server instances.
 * 
 * Prerequisites:
 *   - 3 llama-server instances running:
 *     - :8080 embedding (embeddinggemma-300M, --embedding)
 *     - :8081 reranker (qwen3-reranker-0.6b, --reranking --batch-size 4096)
 *     - :8082 query expansion (qmd-query-expansion-1.7B)
 *   - QMD indexed with memory files
 *   - QMD_REMOTE_MODE=1 env var set
 * 
 * Usage:
 *   QMD_REMOTE_MODE=1 \
 *   XDG_CONFIG_HOME=~/.openclaw/agents/main/qmd/xdg-config \
 *   XDG_CACHE_HOME=~/.openclaw/agents/main/qmd/xdg-cache \
 *   node test_remote_llm.mjs
 */

const EMBED_URL = process.env.QMD_REMOTE_EMBED_URL || 'http://localhost:8080';
const RERANK_URL = process.env.QMD_REMOTE_RERANK_URL || 'http://localhost:8081';
const GENERATE_URL = process.env.QMD_REMOTE_GENERATE_URL || 'http://localhost:8082';

let passed = 0;
let failed = 0;
const failures = [];

function assert(condition, name, detail = '') {
    if (condition) {
        console.log(`  ✅ ${name}`);
        passed++;
    } else {
        console.log(`  ❌ ${name}${detail ? ': ' + detail : ''}`);
        failed++;
        failures.push(name);
    }
}

async function httpPost(url, body, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(`HTTP ${res.status}: ${text}`);
        }
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

// ============================================================================
// Test 1: Embedding API
// ============================================================================
async function testEmbedding() {
    console.log('\n🔬 Test 1: Embedding API');
    
    // 1a: Single text embedding
    const t0 = Date.now();
    const res = await httpPost(`${EMBED_URL}/v1/embeddings`, {
        input: 'gsplat backward kernel optimization',
        model: 'embed',
    });
    const elapsed = Date.now() - t0;
    
    assert(res.data && res.data.length === 1, 'Single embedding returns 1 result');
    assert(res.data[0].embedding.length === 768, `Embedding dimension is 768 (got ${res.data[0].embedding?.length})`);
    assert(elapsed < 5000, `Latency < 5s (got ${elapsed}ms)`);
    
    // 1b: Batch embedding
    const t1 = Date.now();
    const batchRes = await httpPost(`${EMBED_URL}/v1/embeddings`, {
        input: [
            'gsplat backward kernel',
            '视频渲染抖动修复',
            'Hugo博客发布流程',
            'podcast TTS Cherry voice',
            'ICSE巴西旅行',
        ],
        model: 'embed',
    });
    const batchElapsed = Date.now() - t1;
    
    assert(batchRes.data.length === 5, `Batch returns 5 results (got ${batchRes.data.length})`);
    assert(batchRes.data.every(d => d.embedding.length === 768), 'All batch embeddings are 768-dim');
    assert(batchElapsed < 10000, `Batch latency < 10s (got ${batchElapsed}ms)`);
    
    // 1c: Cosine similarity sanity check
    const simRes = await httpPost(`${EMBED_URL}/v1/embeddings`, {
        input: [
            'video rendering jitter fix',     // semantically similar to #1
            'gsplat gradient bug fix',         // semantically similar to #0
            'cooking recipe for pasta',        // unrelated
        ],
        model: 'embed',
    });
    
    function cosine(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
    
    const emb_gsplat = batchRes.data[0].embedding;
    const emb_video = batchRes.data[1].embedding;
    const emb_gsplat2 = simRes.data[1].embedding;
    const emb_video2 = simRes.data[0].embedding;
    const emb_unrelated = simRes.data[2].embedding;
    
    const sim_gsplat = cosine(emb_gsplat, emb_gsplat2);
    const sim_video = cosine(emb_video, emb_video2);
    const sim_unrelated = cosine(emb_gsplat, emb_unrelated);
    
    assert(sim_gsplat > sim_unrelated, `gsplat↔gsplat2 (${sim_gsplat.toFixed(3)}) > gsplat↔pasta (${sim_unrelated.toFixed(3)})`);
    assert(sim_video > sim_unrelated, `video↔video2 (${sim_video.toFixed(3)}) > gsplat↔pasta (${sim_unrelated.toFixed(3)})`);
}

// ============================================================================
// Test 2: Reranking API
// ============================================================================
async function testReranking() {
    console.log('\n🔬 Test 2: Reranking API');
    
    const query = '视频抖动怎么修复';
    const docs = [
        'Hugo博客发布流程 git push GitHub Actions 自动构建',
        '视频抖动 V1-V3 Video依赖Chrome seeking headless不精确 → V4改用OffthreadVideo',
        'gsplat backward kernel per-gaussian 并行化',
        '字幕切分 splitSentences 先按句号切 超30字按逗号二次切',
        'BGM音量 volume=0.09 舒缓钢琴曲',
    ];
    
    const t0 = Date.now();
    const res = await httpPost(`${RERANK_URL}/v1/rerank`, {
        model: 'rerank', query, documents: docs, top_n: docs.length,
    });
    const elapsed = Date.now() - t0;
    
    assert(res.results && res.results.length === docs.length, `Returns ${docs.length} results`);
    assert(elapsed < 10000, `Latency < 10s (got ${elapsed}ms)`);
    
    // The video jitter fix doc (index 1) should rank first
    const sorted = res.results.sort((a, b) => b.relevance_score - a.relevance_score);
    assert(sorted[0].index === 1, `Top result is video fix doc (index=${sorted[0].index}, expected 1)`);
    assert(sorted[0].relevance_score > 0.5, `Top score > 0.5 (got ${sorted[0].relevance_score.toFixed(4)})`);
    
    // Unrelated docs should score low
    const hugoScore = res.results.find(r => r.index === 0).relevance_score;
    const videoScore = res.results.find(r => r.index === 1).relevance_score;
    assert(videoScore > hugoScore * 10, `Video score (${videoScore.toFixed(4)}) >> Hugo score (${hugoScore.toFixed(4)})`);
}

// ============================================================================
// Test 3: Query Expansion API
// ============================================================================
async function testQueryExpansion() {
    console.log('\n🔬 Test 3: Query Expansion API');
    
    const t0 = Date.now();
    const res = await httpPost(`${GENERATE_URL}/v1/chat/completions`, {
        model: 'qexpand',
        messages: [{ role: 'user', content: '/no_think Expand this search query: gsplat gradient bug' }],
        max_tokens: 600,
        temperature: 0.7,
    });
    const elapsed = Date.now() - t0;
    
    assert(res.choices && res.choices.length > 0, 'Returns at least 1 choice');
    const content = res.choices[0].message.content;
    assert(content.length > 10, `Response length > 10 chars (got ${content.length})`);
    assert(elapsed < 10000, `Latency < 10s (got ${elapsed}ms)`);
    
    // Check format: should have lex:/vec:/hyde: lines
    const lines = content.trim().split('\n').filter(l => l.includes(':'));
    const hasTypedLines = lines.some(l => /^(lex|vec|hyde):/.test(l.trim()));
    assert(hasTypedLines, `Output contains typed lines (lex/vec/hyde): ${lines.slice(0,2).join('; ')}`);
    
    console.log(`  📝 Expansion output (first 200 chars): ${content.slice(0, 200)}`);
}

// ============================================================================
// Test 4: QMD CLI end-to-end (query mode)
// ============================================================================
async function testQmdCli() {
    console.log('\n🔬 Test 4: QMD CLI end-to-end');
    
    const { execSync } = await import('child_process');
    const env = {
        ...process.env,
        QMD_REMOTE_MODE: '1',
        XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,
        XDG_CACHE_HOME: process.env.XDG_CACHE_HOME,
    };
    
    // Test cases: query → expected file pattern in results
    const tests = [
        { query: 'gsplat梯度bug shared memory', expect: /memory\.md|2025-02-25|2026-02/ },
        { query: '视频渲染抖动 OffthreadVideo', expect: /memory\.md|2026-02-27|2026-02-28/ },
        { query: 'podcast TTS Cherry voice', expect: /memory\.md|2026-03/ },
        { query: 'Hugo博客发布 GitHub Actions', expect: /memory\.md|blog|2026-02/ },
        { query: 'ICSE巴西旅行签证', expect: /china-rio-flights|memory\.md/ },
    ];
    
    for (const { query, expect } of tests) {
        const t0 = Date.now();
        try {
            const out = execSync(
                `qmd query "${query}" --json -n 3 2>/dev/null`,
                { env, timeout: 30000, encoding: 'utf-8', shell: true }
            );
            const elapsed = Date.now() - t0;
            
            // Parse JSON - find the array in output
            const jsonMatch = out.match(/\[[\s\S]*\]/);
            if (!jsonMatch) {
                assert(false, `"${query}" returns valid JSON`, `output: ${out.slice(0, 100)}`);
                continue;
            }
            
            const results = JSON.parse(jsonMatch[0]);
            assert(results.length > 0, `"${query}" returns results (got ${results.length}, ${elapsed}ms)`);
            
            const files = results.map(r => r.file).join(', ');
            const matched = expect.test(files);
            assert(matched, `"${query}" matches expected files`, `got: ${files}`);
        } catch (e) {
            assert(false, `"${query}" completes without error`, e.message.slice(0, 100));
        }
    }
}

// ============================================================================
// Test 5: Performance benchmarks
// ============================================================================
async function testPerformance() {
    console.log('\n🔬 Test 5: Performance benchmarks');
    
    // Run 3 sequential full pipeline queries and measure
    const queries = ['gradient bug fix', '视频渲染问题', 'Hugo blog deployment'];
    const times = [];
    
    const { execSync } = await import('child_process');
    const env = {
        ...process.env,
        QMD_REMOTE_MODE: '1',
    };
    
    for (const q of queries) {
        const t0 = Date.now();
        try {
            execSync(`qmd query "${q}" --json -n 3 2>/dev/null`, {
                env, timeout: 30000, encoding: 'utf-8', shell: true
            });
        } catch {}
        times.push(Date.now() - t0);
    }
    
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const max = Math.max(...times);
    
    console.log(`  📊 Times: ${times.map(t => t + 'ms').join(', ')}`);
    console.log(`  📊 Average: ${avg.toFixed(0)}ms, Max: ${max}ms`);
    
    assert(avg < 15000, `Average pipeline time < 15s (got ${avg.toFixed(0)}ms)`);
    assert(max < 30000, `Max pipeline time < 30s (got ${max}ms)`);
}

// ============================================================================
// Run all tests
// ============================================================================
async function main() {
    console.log('═══════════════════════════════════════════');
    console.log('  QMD Remote LLM Backend Test Suite');
    console.log(`  Embed:    ${EMBED_URL}`);
    console.log(`  Rerank:   ${RERANK_URL}`);
    console.log(`  Generate: ${GENERATE_URL}`);
    console.log('═══════════════════════════════════════════');
    
    try { await testEmbedding(); } catch (e) { console.error('  💥 Test 1 crashed:', e.message); failed++; }
    try { await testReranking(); } catch (e) { console.error('  💥 Test 2 crashed:', e.message); failed++; }
    try { await testQueryExpansion(); } catch (e) { console.error('  💥 Test 3 crashed:', e.message); failed++; }
    try { await testQmdCli(); } catch (e) { console.error('  💥 Test 4 crashed:', e.message); failed++; }
    try { await testPerformance(); } catch (e) { console.error('  💥 Test 5 crashed:', e.message); failed++; }
    
    console.log('\n═══════════════════════════════════════════');
    console.log(`  Results: ${passed} passed, ${failed} failed`);
    if (failures.length) console.log(`  Failures: ${failures.join(', ')}`);
    console.log('═══════════════════════════════════════════');
    
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
