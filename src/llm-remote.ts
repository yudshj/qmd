/**
 * llm-remote.ts - Remote LLM backend for QMD
 *
 * Drop-in replacement for LlamaCpp that calls remote llama-server HTTP APIs
 * instead of loading models locally via node-llama-cpp.
 *
 * This enables running QMD on a low-resource machine (e.g. VPS with limited RAM)
 * while offloading model inference to a GPU server on the same network.
 *
 * Architecture:
 *   QMD (host machine)  ──HTTP──→  llama-server (GPU machine)
 *     BM25 index                     embedding   (port 8080)
 *     SQLite DB                      reranker    (port 8081)
 *     query dispatch                 qexpand     (port 8082)
 *
 * Environment variables:
 *   QMD_REMOTE_MODE=1              Enable remote backend
 *   QMD_REMOTE_EMBED_URL           Embedding server URL (default: http://localhost:8080)
 *   QMD_REMOTE_RERANK_URL          Reranker server URL  (default: http://localhost:8081)
 *   QMD_REMOTE_GENERATE_URL        Generation server URL (default: http://localhost:8082)
 *
 * llama-server startup:
 *   # Embedding (embeddinggemma-300M-Q8_0.gguf)
 *   llama-server --model embed.gguf --host 0.0.0.0 --port 8080 --embedding --n-gpu-layers 99
 *
 *   # Reranker (qwen3-reranker-0.6b-q8_0.gguf)
 *   llama-server --model rerank.gguf --host 0.0.0.0 --port 8081 --reranking --n-gpu-layers 99 --ctx-size 4096 --batch-size 4096
 *
 *   # Query Expansion (qmd-query-expansion-1.7B-q4_k_m.gguf)
 *   llama-server --model qexpand.gguf --host 0.0.0.0 --port 8082 --n-gpu-layers 99
 */

import type {
  LLM,
  EmbeddingResult,
  EmbedOptions,
  GenerateResult,
  GenerateOptions,
  Queryable,
  QueryType,
  RerankDocument,
  RerankResult,
  RerankOptions,
  ModelInfo,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

const EMBED_URL = process.env.QMD_REMOTE_EMBED_URL || "http://localhost:8080";
const RERANK_URL = process.env.QMD_REMOTE_RERANK_URL || "http://localhost:8081";
const GENERATE_URL = process.env.QMD_REMOTE_GENERATE_URL || "http://localhost:8082";

// =============================================================================
// HTTP Client
// =============================================================================

async function httpPost<T = unknown>(url: string, body: unknown, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

// =============================================================================
// Response Types (llama-server OpenAI-compatible API)
// =============================================================================

interface EmbeddingResponse {
  data: Array<{ embedding: number[]; index: number }>;
  model: string;
  usage: { prompt_tokens: number; total_tokens: number };
}

interface ChatCompletionResponse {
  choices: Array<{
    message: { content: string };
    finish_reason: string;
  }>;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

interface RerankResponse {
  results: Array<{
    index: number;
    relevance_score: number;
  }>;
}

// =============================================================================
// Remote LLM Implementation
// =============================================================================

/**
 * Remote LLM backend that delegates to llama-server HTTP APIs.
 * Implements the same LLM interface as LlamaCpp for seamless drop-in replacement.
 */
export class RemoteLlamaCpp implements LLM {
  private disposed = false;

  // Model URIs (for tracking/logging only)
  readonly embedModelUri = "remote:embeddinggemma-300M";
  readonly generateModelUri = "remote:qmd-query-expansion-1.7B";
  readonly rerankModelUri = "remote:qwen3-reranker-0.6b";

  constructor() {
    console.error(
      `[QMD] Using remote LLM backend: ` +
        `embed=${EMBED_URL}, rerank=${RERANK_URL}, generate=${GENERATE_URL}`
    );
  }

  // ---------------------------------------------------------------------------
  // Embedding
  // ---------------------------------------------------------------------------

  async embed(text: string, _options?: EmbedOptions): Promise<EmbeddingResult | null> {
    try {
      const res = await httpPost<EmbeddingResponse>(`${EMBED_URL}/v1/embeddings`, {
        input: text,
        model: "embed",
      });
      return { embedding: res.data[0].embedding, model: this.embedModelUri };
    } catch (e: unknown) {
      console.error("[QMD Remote] embed error:", (e as Error).message);
      return null;
    }
  }

  /**
   * Batch embedding - llama-server supports array input natively.
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (!texts.length) return [];
    try {
      const res = await httpPost<EmbeddingResponse>(
        `${EMBED_URL}/v1/embeddings`,
        { input: texts, model: "embed" },
        60000
      );
      return res.data.map((d) => ({
        embedding: d.embedding,
        model: this.embedModelUri,
      }));
    } catch (e: unknown) {
      console.error("[QMD Remote] embedBatch error:", (e as Error).message);
      return texts.map(() => null);
    }
  }

  // ---------------------------------------------------------------------------
  // Text Generation
  // ---------------------------------------------------------------------------

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    try {
      const res = await httpPost<ChatCompletionResponse>(`${GENERATE_URL}/v1/chat/completions`, {
        model: "generate",
        messages: [{ role: "user", content: prompt }],
        max_tokens: options?.maxTokens ?? 600,
        temperature: options?.temperature ?? 0.7,
      });
      return {
        text: res.choices[0].message.content,
        model: this.generateModelUri,
        done: true,
      };
    } catch (e: unknown) {
      console.error("[QMD Remote] generate error:", (e as Error).message);
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Query Expansion
  // ---------------------------------------------------------------------------

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;

    try {
      const prompt = `/no_think Expand this search query: ${query}`;
      const res = await httpPost<ChatCompletionResponse>(
        `${GENERATE_URL}/v1/chat/completions`,
        {
          model: "qexpand",
          messages: [{ role: "user", content: prompt }],
          max_tokens: 600,
          temperature: 0.7,
          top_k: 20,
          top_p: 0.8,
          repeat_penalty: 1.5,
        },
        30000
      );

      const result = res.choices[0].message.content;
      const lines = result.trim().split("\n");

      // Parse typed lines (lex:, vec:, hyde:)
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower
        .replace(/[^a-z0-9\s\u4e00-\u9fff]/g, " ")
        .split(/\s+/)
        .filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        return queryTerms.length === 0 || queryTerms.some((t) => lower.includes(t));
      };

      const validTypes = new Set(["lex", "vec", "hyde"]);
      const queryables: Queryable[] = lines
        .map((line) => {
          const ci = line.indexOf(":");
          if (ci === -1) return null;
          const type = line.slice(0, ci).trim();
          if (!validTypes.has(type)) return null;
          const text = line.slice(ci + 1).trim();
          if (!text || !hasQueryTerm(text)) return null;
          return { type: type as QueryType, text };
        })
        .filter((q): q is Queryable => q !== null);

      const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
      if (filtered.length > 0) return filtered;

      // Fallback
      return this.fallbackExpansion(query, includeLexical);
    } catch (e: unknown) {
      console.error("[QMD Remote] expandQuery error:", (e as Error).message);
      return this.fallbackExpansion(query, options?.includeLexical ?? true);
    }
  }

  private fallbackExpansion(query: string, includeLexical: boolean): Queryable[] {
    const fb: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "vec", text: query },
    ];
    if (includeLexical) fb.unshift({ type: "lex", text: query });
    return fb;
  }

  // ---------------------------------------------------------------------------
  // Reranking
  // ---------------------------------------------------------------------------

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options?: RerankOptions
  ): Promise<RerankResult> {
    const texts = documents.map((d) => d.text);

    try {
      // Use native /v1/rerank endpoint (requires llama-server --reranking flag)
      const res = await httpPost<RerankResponse>(
        `${RERANK_URL}/v1/rerank`,
        { model: "rerank", query, documents: texts, top_n: texts.length },
        60000
      );

      if (res.results) {
        const sorted = res.results.sort((a, b) => b.relevance_score - a.relevance_score);
        return {
          results: sorted.map((r) => ({
            file: documents[r.index].file,
            score: r.relevance_score,
            index: r.index,
          })),
          model: this.rerankModelUri,
        };
      }

      console.error("[QMD Remote] rerank response missing results");
    } catch (e: unknown) {
      console.error("[QMD Remote] rerank error:", (e as Error).message);
    }

    // Fallback: return documents in original order
    console.error("[QMD Remote] rerank FALLBACK used for", texts.length, "docs");
    return {
      results: documents.map((d, i) => ({ file: d.file, score: 1 - i * 0.01, index: i })),
      model: this.rerankModelUri,
    };
  }

  // ---------------------------------------------------------------------------
  // Model Info & Lifecycle
  // ---------------------------------------------------------------------------

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true, path: "remote" };
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}
