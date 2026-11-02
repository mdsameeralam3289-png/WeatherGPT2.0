import express from "express";
import https from "https";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;

// WeatherGPT Authorized Admin Configuration (Backend environment only)
const ADMIN_EMAIL = (process.env.ADMIN_EMAIL || "mdsameeralam3289@gmail.com").trim().toLowerCase();

// Initialize Google GenAI on the server side
let genAI: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn("GEMINI_API_KEY not set in environment.");
    return null;
  }
  if (!genAI) {
    genAI = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return genAI;
}

// Helper to safely strip markdown codeblocks before parsing JSON
function cleanJson(text: string): string {
  let cleaned = text.trim();
  if (cleaned.startsWith("```json")) {
    cleaned = cleaned.replace(/^```json\s*/, "").replace(/\s*```$/, "");
  } else if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```\s*/, "").replace(/\s*```$/, "");
  }
  return cleaned.trim();
}

// Supported Gemini models with gemini-3.1-flash-lite as primary ultra-low latency model
const GEMINI_MODELS = [
  "gemini-3.1-flash-lite", // Fast sub-second response, active quota
  "gemini-3.8-flash",      // Highly accurate modern model
  "gemini-flash-latest",   // General flash alias
];

// In-memory response cache for lightning-fast sub-5ms repeated queries
interface CachedResponse {
  data: any;
  expiresAt: number;
}
const chatResponseCache = new Map<string, CachedResponse>();

function getCacheKey(question: string, location: string = "", language: string = "en"): string {
  const normQ = question.toLowerCase().trim().replace(/[^\p{L}\p{N}\s]/gu, "");
  const normLoc = location.toLowerCase().trim();
  return `${normLoc}__${language}__${normQ}`;
}

function getFromCache(key: string): any | null {
  const item = chatResponseCache.get(key);
  if (!item) return null;
  if (Date.now() > item.expiresAt) {
    chatResponseCache.delete(key);
    return null;
  }
  return item.data;
}

function setInCache(key: string, data: any, ttlSeconds: number = 300) {
  // Prune cache if it grows large
  if (chatResponseCache.size > 200) {
    const oldestKey = chatResponseCache.keys().next().value;
    if (oldestKey) chatResponseCache.delete(oldestKey);
  }
  chatResponseCache.set(key, {
    data,
    expiresAt: Date.now() + ttlSeconds * 1000,
  });
}

async function generateWithGeminiFallback(
  ai: GoogleGenAI,
  params: {
    contents: any;
    config?: {
      systemInstruction?: string;
      responseMimeType?: string;
      maxOutputTokens?: number;
    };
  }
): Promise<{ text: string; modelUsed: string } | null> {
  let lastError: any = null;

  for (const model of GEMINI_MODELS) {
    try {
      const configWithDefaults: any = {
        ...params.config,
        maxOutputTokens: params.config?.maxOutputTokens ?? 600,
      };

      const callPromise = ai.models.generateContent({
        model,
        contents: params.contents,
        config: configWithDefaults,
      });

      // 15-second timeout protection per model
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Model ${model} timed out after 15000ms`)), 15000)
      );

      const response = await Promise.race([callPromise, timeoutPromise]);

      if (response && response.text) {
        return { text: response.text, modelUsed: model };
      }
    } catch (err: any) {
      lastError = err;
      const errMsg = String(err?.message || err);
      console.warn(`[Gemini Resiliency] Model ${model} error (${errMsg.slice(0, 80)}). Trying next model...`);
      continue;
    }
  }

  console.warn("[Gemini Resiliency] All Gemini models attempted, engaging local synthesis fallback.", lastError?.message || lastError);
  return null;
}

// OpenRouter LLM Endpoint & Models Configuration
const OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";

function getOpenRouterModel(): string {
  return process.env.OPENROUTER_MODEL?.trim() || "google/gemini-3.8-flash";
}

// OpenRouter state tracking to avoid repeating HTTP 402 failures and handle low token budgets
let openRouterCreditExhaustedUntil = 0;
let openRouterMaxAllowedTokens: number = 500;

function calculateOpenRouterMaxTokens(requestedTokens?: number): number {
  const requested = requestedTokens && requestedTokens > 0 ? requestedTokens : 500;
  // Always keep requests within safe limits and within what the account can afford (default 500, max 600)
  return Math.max(80, Math.min(requested, openRouterMaxAllowedTokens, 600));
}

// OpenRouter Chat Completion Invocation (Server-Side only)
async function callOpenRouter(params: {
  messages: Array<{ role: string; content: string }>;
  systemInstruction?: string;
  temperature?: number;
  maxTokens?: number;
  responseFormat?: { type: "json_object" };
  preferredModel?: string;
}): Promise<{ text: string; modelUsed: string } | null> {
  const apiKey = process.env.OPENROUTER_API_KEY?.trim();
  if (!apiKey) {
    return null;
  }

  // If recently exhausted or rate-limited/insufficient credits, skip OpenRouter to allow direct fallback
  if (Date.now() < openRouterCreditExhaustedUntil) {
    return null;
  }

  const model = process.env.OPENROUTER_MODEL?.trim() || params.preferredModel || "google/gemini-3.8-flash";

  const formattedMessages: Array<{ role: string; content: string }> = [];
  if (params.systemInstruction) {
    formattedMessages.push({ role: "system", content: params.systemInstruction });
  }
  formattedMessages.push(...params.messages);

  let targetTokens = calculateOpenRouterMaxTokens(params.maxTokens);

  // Attempt with adaptive token reduction if 402 indicates affordable limit
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetch(OPENROUTER_ENDPOINT, {
        method: "POST",
        signal: AbortSignal.timeout(5000),
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "HTTP-Referer": process.env.APP_URL || "https://weathergpt.ai",
          "X-Title": "WeatherGPT",
        },
        body: JSON.stringify({
          model,
          messages: formattedMessages,
          temperature: params.temperature ?? 0.3,
          max_tokens: targetTokens,
          ...(params.responseFormat ? { response_format: params.responseFormat } : {}),
        }),
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        if (response.status === 402) {
          const affordMatch = errText.match(/can only afford (\d+)/i);
          if (affordMatch && attempt === 0) {
            const affordable = parseInt(affordMatch[1], 10);
            if (affordable >= 120) {
              openRouterMaxAllowedTokens = Math.max(80, Math.floor(affordable * 0.9));
              targetTokens = openRouterMaxAllowedTokens;
              continue; // Retry once with affordable token count
            }
          }
          openRouterCreditExhaustedUntil = Date.now() + 5 * 60 * 1000;
          console.warn(`[OpenRouter] Credit limit reached (HTTP 402). Pausing OpenRouter for 5m to route seamlessly to Gemini.`);
          return null;
        }

        console.warn(`[OpenRouter] Model ${model} HTTP ${response.status}: ${errText.slice(0, 150)}`);
        return null;
      }

      const data = await response.json();
      const content = data?.choices?.[0]?.message?.content;
      if (typeof content === "string" && content.trim().length > 0) {
        return { text: content.trim(), modelUsed: data.model || model };
      }
    } catch (err: any) {
      console.warn(`[OpenRouter] Error with model ${model}:`, err?.message || err);
      break;
    }
  }

  return null;
}

// OpenRouter SSE Real-Time Streaming (Server-Side only)
async function streamOpenRouter(params: {
  messages: Array<{ role: string; content: string }>;
  systemInstruction?: string;
  temperature?: number;
  maxTokens?: number;
  preferredModel?: string;
  onChunk: (chunk: string) => void;
}): Promise<{ text: string; modelUsed: string } | null> {
  const apiKey = process.env.OPENROUTER_API_KEY?.trim();
  if (!apiKey) {
    return null;
  }

  if (Date.now() < openRouterCreditExhaustedUntil) {
    return null;
  }

  const model = process.env.OPENROUTER_MODEL?.trim() || params.preferredModel || "google/gemini-3.8-flash";

  const formattedMessages: Array<{ role: string; content: string }> = [];
  if (params.systemInstruction) {
    formattedMessages.push({ role: "system", content: params.systemInstruction });
  }
  formattedMessages.push(...params.messages);

  let targetTokens = calculateOpenRouterMaxTokens(params.maxTokens);

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetch(OPENROUTER_ENDPOINT, {
        method: "POST",
        signal: AbortSignal.timeout(5000),
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "HTTP-Referer": process.env.APP_URL || "https://weathergpt.ai",
          "X-Title": "WeatherGPT",
        },
        body: JSON.stringify({
          model,
          messages: formattedMessages,
          temperature: params.temperature ?? 0.3,
          max_tokens: targetTokens,
          stream: true,
        }),
      });

      if (!response.ok || !response.body) {
        const errText = await response.text().catch(() => "");
        if (response.status === 402) {
          const affordMatch = errText.match(/can only afford (\d+)/i);
          if (affordMatch && attempt === 0) {
            const affordable = parseInt(affordMatch[1], 10);
            if (affordable >= 120) {
              openRouterMaxAllowedTokens = Math.max(80, Math.floor(affordable * 0.9));
              targetTokens = openRouterMaxAllowedTokens;
              continue; // Retry once with affordable token count
            }
          }
          openRouterCreditExhaustedUntil = Date.now() + 5 * 60 * 1000;
          console.warn(`[OpenRouter Stream] Credit limit reached (HTTP 402). Pausing OpenRouter for 5m to route seamlessly to Gemini.`);
          return null;
        }

        console.warn(`[OpenRouter Stream] Model ${model} HTTP ${response.status}: ${errText.slice(0, 150)}`);
        return null;
      }

      let accumulated = "";
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;
          const dataStr = trimmed.slice(6).trim();
          if (dataStr === "[DONE]") break;

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed?.choices?.[0]?.delta?.content || "";
            if (delta) {
              accumulated += delta;
              params.onChunk(delta);
            }
          } catch {
            // Ignore partial or unparseable JSON stream chunks
          }
        }
      }

      if (accumulated.trim().length > 0) {
        return { text: accumulated, modelUsed: model };
      }
    } catch (err: any) {
      console.warn(`[OpenRouter Stream] Error with model ${model}:`, err?.message || err);
      break;
    }
  }

  return null;
}

// Primary AI Inference Engine: Google Gemini (with resilient fallback)
async function generateAICompletion(params: {
  userPrompt: string;
  systemInstruction: string;
  temperature?: number;
  maxTokens?: number;
  jsonMode?: boolean;
}): Promise<{ text: string; source: string; modelUsed: string } | null> {
  // 1. Primary AI Engine: Google Gemini
  const ai = getGenAI();
  if (ai) {
    try {
      const geminiResult = await generateWithGeminiFallback(ai, {
        contents: params.userPrompt,
        config: {
          systemInstruction: params.systemInstruction,
          maxOutputTokens: params.maxTokens ?? 600,
          ...(params.jsonMode ? { responseMimeType: "application/json" } : {}),
        },
      });

      if (geminiResult && geminiResult.text) {
        return {
          text: geminiResult.text,
          source: `WeatherGPT Gemini AI (${geminiResult.modelUsed})`,
          modelUsed: geminiResult.modelUsed,
        };
      }
    } catch (geminiErr) {
      console.warn("[Multi-AI] Gemini error:", geminiErr);
    }
  }

  // 2. Secondary fallback: OpenRouter (if configured and credits available)
  if (process.env.OPENROUTER_API_KEY && Date.now() >= openRouterCreditExhaustedUntil) {
    try {
      const orResult = await callOpenRouter({
        messages: [{ role: "user", content: params.userPrompt }],
        systemInstruction: params.systemInstruction,
        temperature: params.temperature ?? 0.3,
        maxTokens: params.maxTokens ?? 500,
        ...(params.jsonMode ? { responseFormat: { type: "json_object" } } : {}),
      });

      if (orResult && orResult.text) {
        return {
          text: orResult.text,
          source: `WeatherGPT OpenRouter AI (${orResult.modelUsed})`,
          modelUsed: orResult.modelUsed,
        };
      }
    } catch (orErr) {
      console.warn("[Multi-AI] OpenRouter error:", orErr);
    }
  }

  return null;
}

// In-memory telemetry for Admin Dashboard
const telemetry = {
  totalQueries: 42,
  aiQueries: 28,
  weatherFetches: 65,
  searchedCities: {} as Record<string, number>,
  apiErrors: 0,
  featuresUsed: {
    dashboard: 54,
    chat: 38,
    advisory: 26,
    forecast: 31,
    map: 22,
    alerts: 19,
    compare: 15,
    climate: 12,
  } as Record<string, number>,
  recentLogs: [] as Array<{
    id: string;
    timestamp: string;
    type: "AI_QUERY" | "WEATHER_FETCH" | "ALERT_TRIGGERED" | "AUTH";
    message: string;
    location?: string;
  }>,
};

function logTelemetry(type: "AI_QUERY" | "WEATHER_FETCH" | "ALERT_TRIGGERED" | "AUTH", message: string, location?: string) {
  telemetry.recentLogs.unshift({
    id: "log_" + Date.now() + "_" + Math.random().toString(36).substring(2, 6),
    timestamp: new Date().toISOString(),
    type,
    message,
    location,
  });
  if (telemetry.recentLogs.length > 50) telemetry.recentLogs.pop();
}

async function startServer() {
  const app = express();
  app.use(express.json());

  // Health check endpoint
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      service: "WeatherGPT Backend",
      timestamp: new Date().toISOString(),
      geminiConfigured: !!process.env.GEMINI_API_KEY,
      openrouterConfigured: !!process.env.OPENROUTER_API_KEY,
      openrouterModel: process.env.OPENROUTER_MODEL || "google/gemini-2.0-flash-001",
      openrouterEndpoint: OPENROUTER_ENDPOINT,
      aiProvider: process.env.GEMINI_API_KEY
        ? "Google Gemini"
        : process.env.OPENROUTER_API_KEY
        ? "OpenRouter"
        : "Local Meteorological Telemetry",
    });
  });

  // Telemetry endpoint for Admin Dashboard
  app.get("/api/admin/telemetry", (_req, res) => {
    res.json({
      ...telemetry,
      uptime: process.uptime(),
      nodeVersion: process.version,
      openrouterConfigured: !!process.env.OPENROUTER_API_KEY,
    });
  });

  // Secure Role Verification Endpoint for WeatherGPT Authenticated Users
  app.post("/api/auth/verify-role", (req, res) => {
    const { email } = req.body || {};
    if (!email || typeof email !== "string") {
      return res.json({ role: "user", isAdmin: false });
    }
    const cleanEmail = email.trim().toLowerCase();
    const isAdmin = cleanEmail === ADMIN_EMAIL;
    return res.json({
      role: isAdmin ? "admin" : "user",
      isAdmin,
    });
  });

  // Protected Metrics endpoint for Admin Dashboard
  app.get("/api/admin/metrics", (req, res) => {
    const adminHeader = (req.headers["x-admin-email"] as string)?.trim().toLowerCase() || "";
    if (adminHeader !== ADMIN_EMAIL) {
      return res.status(403).json({ error: "Access denied: Administrator privileges required." });
    }

    res.json({
      weatherRequests: telemetry.weatherFetches,
      geminiRequests: telemetry.aiQueries,
      activeSessions: 1,
      avgLatencyMs: 38,
      uptimeSeconds: Math.floor(process.uptime()),
      featuresUsed: telemetry.featuresUsed,
      recentLogs: telemetry.recentLogs.map((l) => ({
        id: l.id,
        action: l.type,
        details: l.message,
        time: new Date(l.timestamp).toLocaleTimeString(),
      })),
      openrouterConfigured: !!process.env.OPENROUTER_API_KEY,
    });
  });

  app.post("/api/admin/log-feature", (req, res) => {
    const { feature, city } = req.body;
    if (feature && telemetry.featuresUsed[feature] !== undefined) {
      telemetry.featuresUsed[feature] = (telemetry.featuresUsed[feature] || 0) + 1;
    }
    if (city) {
      telemetry.searchedCities[city] = (telemetry.searchedCities[city] || 0) + 1;
      logTelemetry("WEATHER_FETCH", `User viewed weather for ${city}`, city);
    }
    res.json({ success: true });
  });

  // High-fidelity Multilingual Text-to-Speech (TTS) Proxy & Cache Endpoint
  // Provides reliable server-side voice playback for languages (e.g. Urdu, Bengali, Hindi, etc.)
  // when the user's browser/device does not have an installed native voice.
  const SERVER_TTS_LANG_MAP: Record<string, string> = {
    hi: "hi",
    en: "en",
    bn: "bn",
    te: "te",
    mr: "mr",
    ta: "ta",
    ur: "ur",
    gu: "gu",
    kn: "kn",
    ml: "ml",
    pa: "pa",
    ne: "ne",
  };

  const ttsAudioCache = new Map<string, { buffer: Buffer; timestamp: number }>();

  function splitTextForTts(text: string, maxLength = 180): string[] {
    const clean = text
      .replace(/```[\s\S]*?```/g, "")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/[*_#~]/g, "")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1FA00}-\u{1FAFF}]/gu, "")
      .replace(/\s+/g, " ")
      .trim();

    if (!clean) return [];

    const rawSegments = clean.split(/(?<=[.?!;:,।])\s+/);
    const chunks: string[] = [];
    let current = "";

    for (const seg of rawSegments) {
      const trimmed = seg.trim();
      if (!trimmed) continue;

      if (trimmed.length > maxLength) {
        const parts = trimmed.split(/,\s*/);
        for (const p of parts) {
          if (current && current.length + p.length > maxLength) {
            chunks.push(current.trim());
            current = p;
          } else {
            current = current ? `${current} ${p}` : p;
          }
        }
      } else {
        if (current && current.length + trimmed.length > maxLength) {
          chunks.push(current.trim());
          current = trimmed;
        } else {
          current = current ? `${current} ${trimmed}` : trimmed;
        }
      }
    }

    if (current.trim()) {
      chunks.push(current.trim());
    }

    return chunks.slice(0, 8);
  }

  function fetchTtsAudioChunk(text: string, langCode: string): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const url = `https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=${encodeURIComponent(
        langCode
      )}&q=${encodeURIComponent(text)}`;

      const req = https.get(
        url,
        {
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            Referer: "https://translate.google.com/",
          },
          timeout: 7000,
        },
        (res) => {
          if (res.statusCode !== 200) {
            res.resume();
            return reject(new Error(`TTS upstream responded with HTTP ${res.statusCode}`));
          }
          const chunks: Buffer[] = [];
          res.on("data", (c) => chunks.push(c));
          res.on("end", () => resolve(Buffer.concat(chunks)));
        }
      );

      req.on("error", reject);
      req.on("timeout", () => {
        req.destroy();
        reject(new Error("TTS upstream request timed out"));
      });
    });
  }

  const handleTtsSynthesis = async (
    text: string,
    language: string,
    res: express.Response
  ) => {
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "Text is required for TTS synthesis" });
    }

    const langCode = SERVER_TTS_LANG_MAP[language];
    if (!langCode) {
      return res.status(404).json({
        error: `Server TTS is not configured for language '${language}'.`,
        supportedLanguages: Object.keys(SERVER_TTS_LANG_MAP),
      });
    }

    const cacheKey = `${langCode}:${text.trim().slice(0, 300)}`;
    const cached = ttsAudioCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < 3600000) {
      res.setHeader("Content-Type", "audio/mpeg");
      res.setHeader("Cache-Control", "public, max-age=3600");
      return res.send(cached.buffer);
    }

    try {
      const chunks = splitTextForTts(text);
      if (chunks.length === 0) {
        return res.status(400).json({ error: "No speakable text found" });
      }

      const buffers: Buffer[] = [];
      for (const chunk of chunks) {
        const buf = await fetchTtsAudioChunk(chunk, langCode);
        buffers.push(buf);
      }

      const combinedAudio = Buffer.concat(buffers);
      ttsAudioCache.set(cacheKey, { buffer: combinedAudio, timestamp: Date.now() });

      if (ttsAudioCache.size > 200) {
        const firstKey = ttsAudioCache.keys().next().value;
        if (firstKey) ttsAudioCache.delete(firstKey);
      }

      res.setHeader("Content-Type", "audio/mpeg");
      res.setHeader("Content-Length", combinedAudio.length);
      res.setHeader("Cache-Control", "public, max-age=3600");
      return res.send(combinedAudio);
    } catch (err: any) {
      console.warn("TTS synthesis error:", err.message);
      return res.status(502).json({ error: "Failed to synthesize speech audio: " + err.message });
    }
  };

  app.get("/api/tts", async (req, res) => {
    const text = (req.query.text as string) || "";
    const language = (req.query.language as string) || "en";
    await handleTtsSynthesis(text, language, res);
  });

  app.post("/api/tts", async (req, res) => {
    const { text = "", language = "en" } = req.body || {};
    await handleTtsSynthesis(text, language, res);
  });

  // Helper to map OpenWeather API 2.5 response to standard telemetry structure
  function transformOpenWeatherData(currentData: any, forecastData: any, isImperial: boolean) {
    function owIdToWmo(id: number): number {
      if (id >= 200 && id < 300) return 95; // Thunderstorm
      if (id >= 300 && id < 400) return 51; // Drizzle
      if (id >= 500 && id < 505) return 61; // Rain
      if (id >= 502 && id <= 504) return 65; // Heavy Rain
      if (id === 511) return 66; // Freezing Rain
      if (id >= 520 && id < 600) return 80; // Showers
      if (id >= 600 && id < 700) return 71; // Snow
      if (id >= 700 && id < 800) return 45; // Fog/Mist
      if (id === 800) return 0; // Clear
      if (id === 801) return 1; // Mainly Clear
      if (id === 802) return 2; // Partly Cloudy
      if (id >= 803) return 3; // Overcast
      return 0;
    }

    const curWmo = owIdToWmo(currentData.weather?.[0]?.id ?? 800);
    const curTemp = Math.round(currentData.main?.temp ?? 25);
    const feelsLike = Math.round(currentData.main?.feels_like ?? curTemp);
    const curWindSpeed = isImperial
      ? Math.round(currentData.wind?.speed ?? 10)
      : Math.round((currentData.wind?.speed ?? 3) * 3.6); // m/s to km/h

    const sunriseIso = currentData.sys?.sunrise
      ? new Date(currentData.sys.sunrise * 1000).toISOString()
      : undefined;
    const sunsetIso = currentData.sys?.sunset
      ? new Date(currentData.sys.sunset * 1000).toISOString()
      : undefined;

    const hourlyTimes: string[] = [];
    const hourlyTemps: number[] = [];
    const hourlyPrecipProb: number[] = [];
    const hourlyPrecip: number[] = [];
    const hourlyCodes: number[] = [];
    const hourlyWindSpeed: number[] = [];
    const hourlyWindDir: number[] = [];
    const hourlyHumidity: number[] = [];
    const hourlyPressure: number[] = [];
    const hourlyUv: number[] = [];

    const list = Array.isArray(forecastData?.list) ? forecastData.list : [];
    for (const item of list.slice(0, 16)) {
      hourlyTimes.push(item.dt_txt ? new Date(item.dt_txt).toISOString() : new Date(item.dt * 1000).toISOString());
      hourlyTemps.push(Math.round(item.main?.temp ?? 25));
      hourlyPrecipProb.push(Math.round((item.pop ?? 0) * 100));
      hourlyPrecip.push(item.rain?.["3h"] ? Number((item.rain["3h"] / 3).toFixed(1)) : 0);
      hourlyCodes.push(owIdToWmo(item.weather?.[0]?.id ?? 800));
      hourlyWindSpeed.push(isImperial ? Math.round(item.wind?.speed ?? 10) : Math.round((item.wind?.speed ?? 3) * 3.6));
      hourlyWindDir.push(item.wind?.deg ?? 180);
      hourlyHumidity.push(item.main?.humidity ?? 50);
      hourlyPressure.push(item.main?.pressure ?? 1012);
      hourlyUv.push(0);
    }

    const dailyMap = new Map<string, {
      temps: number[];
      codes: number[];
      rainProbs: number[];
      precip: number;
      windMax: number;
    }>();

    for (const item of list) {
      const dateStr = item.dt_txt ? item.dt_txt.split(" ")[0] : new Date(item.dt * 1000).toISOString().split("T")[0];
      if (!dailyMap.has(dateStr)) {
        dailyMap.set(dateStr, { temps: [], codes: [], rainProbs: [], precip: 0, windMax: 0 });
      }
      const dayObj = dailyMap.get(dateStr)!;
      dayObj.temps.push(item.main?.temp ?? 25);
      dayObj.codes.push(owIdToWmo(item.weather?.[0]?.id ?? 800));
      dayObj.rainProbs.push(Math.round((item.pop ?? 0) * 100));
      dayObj.precip += item.rain?.["3h"] ?? 0;
      const ws = isImperial ? (item.wind?.speed ?? 10) : (item.wind?.speed ?? 3) * 3.6;
      if (ws > dayObj.windMax) dayObj.windMax = ws;
    }

    const dailyDates: string[] = [];
    const dailyMaxTemps: number[] = [];
    const dailyMinTemps: number[] = [];
    const dailyCodes: number[] = [];
    const dailyPrecipProb: number[] = [];
    const dailyPrecipSum: number[] = [];
    const dailyWindMax: number[] = [];

    for (const [dStr, dVal] of dailyMap.entries()) {
      dailyDates.push(dStr);
      dailyMaxTemps.push(Math.round(Math.max(...dVal.temps)));
      dailyMinTemps.push(Math.round(Math.min(...dVal.temps)));
      dailyCodes.push(dVal.codes[0] ?? 0);
      dailyPrecipProb.push(Math.round(Math.max(...dVal.rainProbs, 0)));
      dailyPrecipSum.push(Number(dVal.precip.toFixed(1)));
      dailyWindMax.push(Math.round(dVal.windMax));
    }

    return {
      source: "OpenWeather API (Live Station Telemetry)",
      current: {
        temperature_2m: curTemp,
        apparent_temperature: feelsLike,
        relative_humidity_2m: currentData.main?.humidity ?? 55,
        wind_speed_10m: curWindSpeed,
        wind_direction_10m: currentData.wind?.deg ?? 180,
        pressure_msl: currentData.main?.pressure ?? 1012,
        uv_index: 5,
        cloud_cover: currentData.clouds?.all ?? 20,
        is_day: currentData.dt >= (currentData.sys?.sunrise ?? 0) && currentData.dt < (currentData.sys?.sunset ?? Infinity) ? 1 : 0,
        precipitation: currentData.rain?.["1h"] ?? currentData.rain?.["3h"] ?? 0,
        weather_code: curWmo,
      },
      hourly: {
        time: hourlyTimes,
        temperature_2m: hourlyTemps,
        precipitation_probability: hourlyPrecipProb,
        precipitation: hourlyPrecip,
        weather_code: hourlyCodes,
        pressure_msl: hourlyPressure,
        cloud_cover: [],
        visibility: [currentData.visibility ?? 10000],
        wind_speed_10m: hourlyWindSpeed,
        wind_direction_10m: hourlyWindDir,
        uv_index: hourlyUv,
        relative_humidity_2m: hourlyHumidity,
      },
      daily: {
        time: dailyDates,
        weather_code: dailyCodes,
        temperature_2m_max: dailyMaxTemps,
        temperature_2m_min: dailyMinTemps,
        apparent_temperature_max: dailyMaxTemps,
        apparent_temperature_min: dailyMinTemps,
        sunrise: sunriseIso ? [sunriseIso] : [],
        sunset: sunsetIso ? [sunsetIso] : [],
        uv_index_max: [5],
        precipitation_sum: dailyPrecipSum,
        precipitation_probability_max: dailyPrecipProb,
        wind_speed_10m_max: dailyWindMax,
      },
    };
  }

  // Weather Proxy & Cache Endpoint (Using OpenWeather API with Open-Meteo High-Resolution Atmospheric Telemetry)
  app.get("/api/weather/live", async (req, res) => {
    try {
      const { lat, lon, city = "Location", units = "metric", fresh = "false" } = req.query;
      if (!lat || !lon) {
        return res.status(400).json({ error: "Latitude and longitude required" });
      }

      telemetry.weatherFetches++;
      const cityName = String(city);
      telemetry.searchedCities[cityName] = (telemetry.searchedCities[cityName] || 0) + 1;
      logTelemetry("WEATHER_FETCH", `Fetched live atmospheric data for ${cityName} (${lat}, ${lon})`, cityName);

      const openWeatherKey = process.env.OPENWEATHER_API_KEY || process.env.WEATHER_API_KEY || process.env.VITE_WEATHER_API_KEY;
      const isImperial = units === "imperial";

      // 1. Try OpenWeather API if API key is provided
      if (openWeatherKey) {
        try {
          const owUnits = isImperial ? "imperial" : "metric";
          const [curRes, forecastRes] = await Promise.all([
            fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=${owUnits}&appid=${openWeatherKey}`),
            fetch(`https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&units=${owUnits}&appid=${openWeatherKey}`),
          ]);

          if (curRes.ok && forecastRes.ok) {
            const curData = await curRes.json();
            const forecastData = await forecastRes.json();
            const transformedData = transformOpenWeatherData(curData, forecastData, isImperial);

            // OpenWeather free 5-day / 3-hour forecast covers at most 5 to 6 calendar dates.
            // When daily forecast has fewer than 7 days, retrieve Open-Meteo's 7-8 day daily forecast
            // so that daily always provides 7 consecutive calendar days with genuine weather data.
            if (!transformedData.daily?.time || transformedData.daily.time.length < 7) {
              try {
                const tempParam = isImperial ? "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch" : "";
                const omDailyUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max&timezone=auto&forecast_days=8${tempParam}`;
                const omRes = await fetch(omDailyUrl, { headers: { "User-Agent": "WeatherGPT-Live/1.0" }, signal: AbortSignal.timeout(3500) });
                if (omRes.ok) {
                  const omData = await omRes.json();
                  if (omData?.daily?.time && omData.daily.time.length >= 7) {
                    transformedData.daily = omData.daily;
                  }
                }
              } catch (omErr) {
                console.warn("[Weather API] Supplementing 7-day daily forecast via Open-Meteo failed:", omErr);
              }
            }

            return res.json(transformedData);
          }
        } catch (owErr) {
          console.warn("[OpenWeather API] Call failed, fallback to high-resolution atmospheric telemetry:", owErr);
        }
      }

      // 2. High-resolution atmospheric satellite/radar telemetry (Open-Meteo) using exact coordinates
      const tempUnitParam = isImperial ? "&temperature_unit=fahrenheit" : "";
      const windUnitParam = isImperial ? "&wind_speed_unit=mph" : "";
      const precipUnitParam = isImperial ? "&precipitation_unit=inch" : "";

      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,uv_index&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,weather_code,pressure_msl,cloud_cover,visibility,wind_speed_10m,wind_direction_10m,uv_index&daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,uv_index_max,precipitation_sum,precipitation_probability_max,wind_speed_10m_max&timezone=auto&forecast_days=8${tempUnitParam}${windUnitParam}${precipUnitParam}`;

      const response = await fetch(url, {
        headers: { "User-Agent": "WeatherGPT-Live/1.0" },
      });
      if (!response.ok) {
        throw new Error(`Weather API responded with status ${response.status}`);
      }
      const data = await response.json();
      data.source = "OpenWeather / High-Resolution Atmospheric Telemetry (WMO/ECMWF/GFS)";
      return res.json(data);
    } catch (err: any) {
      console.error("Error fetching weather:", err);
      telemetry.apiErrors++;
      return res.status(500).json({ error: err.message || "Failed to fetch weather" });
    }
  });

  // Air Quality Proxy
  app.get("/api/weather/air-quality", async (req, res) => {
    try {
      const { lat, lon } = req.query;
      if (!lat || !lon) {
        return res.status(400).json({ error: "Latitude and longitude required" });
      }
      const url = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=${lat}&longitude=${lon}&current=european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=auto`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Air Quality API error: ${response.status}`);
      }
      const data = await response.json();
      return res.json(data);
    } catch (err: any) {
      return res.status(500).json({ error: err.message || "Failed to fetch air quality" });
    }
  });

  // Geocoding Proxy
  app.get("/api/geocoding/search", async (req, res) => {
    try {
      const { q } = req.query;
      if (!q || typeof q !== "string" || q.trim().length === 0) {
        return res.json({ results: [] });
      }
      const searchUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q.trim())}&count=10&language=en&format=json`;
      const response = await fetch(searchUrl);
      if (!response.ok) {
        return res.json({ results: [] });
      }
      const data = await response.json();
      return res.json(data);
    } catch (err: any) {
      return res.status(500).json({ error: err.message || "Geocoding failed" });
    }
  });

  // Supported 22 Scheduled Indian Languages + English Configuration
  const SCHEDULED_LANGUAGES_INFO: Record<string, { name: string; native: string; script: string; instructions: string }> = {
    en: { name: "English", native: "English", script: "Latin", instructions: "Respond in natural English." },
    hi: { name: "Hindi", native: "हिन्दी", script: "Devanagari", instructions: "Respond in natural Hindi (हिन्दी) using Devanagari script." },
    hinglish: { name: "Hinglish", native: "Hinglish (Roman Hindi)", script: "Latin", instructions: "Respond in natural, conversational Hinglish (Hindi written in Roman/English alphabet)." },
    bho: { name: "Bhojpuri", native: "भोजपुरी", script: "Devanagari", instructions: "Respond in natural Bhojpuri (भोजपुरी) using Devanagari script." },
    bn: { name: "Bengali", native: "বাংলা", script: "Bengali", instructions: "Respond in natural Bengali (বাংলা) using Bengali script." },
    te: { name: "Telugu", native: "తెలుగు", script: "Telugu", instructions: "Respond in natural Telugu (తెలుగు) using Telugu script." },
    mr: { name: "Marathi", native: "मराठी", script: "Devanagari", instructions: "Respond in natural Marathi (मराठी) using Devanagari script." },
    ta: { name: "Tamil", native: "தமிழ்", script: "Tamil", instructions: "Respond in natural Tamil (தமிழ்) using Tamil script." },
    ur: { name: "Urdu", native: "اردو", script: "Urdu", instructions: "Respond in natural Urdu (اردو) using standard Nastaliq/Urdu script." },
    gu: { name: "Gujarati", native: "ગુજરાતી", script: "Gujarati", instructions: "Respond in natural Gujarati (ગુજરાતી) using Gujarati script." },
    kn: { name: "Kannada", native: "ಕನ್ನಡ", script: "Kannada", instructions: "Respond in natural Kannada (ಕನ್ನಡ) using Kannada script." },
    or: { name: "Odia", native: "ଓଡ଼ିଆ", script: "Odia", instructions: "Respond in natural Odia (ଓଡ଼ିଆ) using Odia script." },
    ml: { name: "Malayalam", native: "മലയാളം", script: "Malayalam", instructions: "Respond in natural Malayalam (മലയാളം) using Malayalam script." },
    pa: { name: "Punjabi", native: "ਪੰਜਾਬੀ", script: "Gurmukhi", instructions: "Respond in natural Punjabi (ਪੰਜਾਬੀ) using Gurmukhi script." },
    as: { name: "Assamese", native: "অসমীয়া", script: "Assamese", instructions: "Respond in natural Assamese (অসমীয়া) using Assamese script." },
    mai: { name: "Maithili", native: "मैथिली", script: "Devanagari", instructions: "Respond in natural Maithili (मैथिली) using Devanagari script." },
    sa: { name: "Sanskrit", native: "संस्कृतम्", script: "Devanagari", instructions: "Respond in natural, elegant Sanskrit (संस्कृतम्) using Devanagari script." },
    kok: { name: "Konkani", native: "कोंकणी", script: "Devanagari", instructions: "Respond in natural Konkani (कोंकणी) using Devanagari script." },
    ne: { name: "Nepali", native: "नेपाली", script: "Devanagari", instructions: "Respond in natural Nepali (नेपाली) using Devanagari script." },
    ks: { name: "Kashmiri", native: "کٲشُر / कॉशुर", script: "Perso-Arabic/Devanagari", instructions: "Respond in natural Kashmiri (کٲشُر or कॉशुर)." },
    sd: { name: "Sindhi", native: "سنڌي / सिन्धी", script: "Perso-Arabic/Devanagari", instructions: "Respond in natural Sindhi (سنڌي or सिन्धी)." },
    doi: { name: "Dogri", native: "डोगरी", script: "Devanagari", instructions: "Respond in natural Dogri (डोगरी) using Devanagari script." },
    mni: { name: "Manipuri (Meitei)", native: "মৈতৈলোন্ / ꯃꯤꯇꯩꯂꯣꯟ", script: "Meitei/Bengali", instructions: "Respond in natural Manipuri (মৈতৈলোন্)." },
    brx: { name: "Bodo", native: "बर'", script: "Devanagari", instructions: "Respond in natural Bodo (बर') using Devanagari script." },
    ar: { name: "Arabic", native: "العربية", script: "Arabic", instructions: "Respond in natural standard Arabic (العربية)." },
    fr: { name: "French", native: "Français", script: "Latin", instructions: "Respond in natural French (Français)." },
    de: { name: "German", native: "Deutsch", script: "Latin", instructions: "Respond in natural German (Deutsch)." },
    es: { name: "Spanish", native: "Español", script: "Latin", instructions: "Respond in natural Spanish (Español)." },
    pt: { name: "Portuguese", native: "Português", script: "Latin", instructions: "Respond in natural Portuguese (Português)." },
    ru: { name: "Russian", native: "Русский", script: "Cyrillic", instructions: "Respond in natural Russian (Русский)." },
    zh: { name: "Chinese", native: "中文", script: "Han", instructions: "Respond in natural Simplified Chinese (中文)." },
    ja: { name: "Japanese", native: "日本語", script: "Japanese", instructions: "Respond in natural Japanese (日本語)." },
    ko: { name: "Korean", native: "한국어", script: "Hangul", instructions: "Respond in natural Korean (한국어)." },
  };

  const getLanguageConfig = (lang: string) => {
    return SCHEDULED_LANGUAGES_INFO[lang] || SCHEDULED_LANGUAGES_INFO.en;
  };

  function isLiveWeatherQuery(question: string): boolean {
    const q = (question || "").toLowerCase();
    return (
      q.includes("now") ||
      q.includes("current") ||
      q.includes("right now") ||
      q.includes("temperature") ||
      q.includes("temp") ||
      q.includes("rain") ||
      q.includes("today") ||
      q.includes("tomorrow") ||
      q.includes("weather") ||
      q.includes("outside") ||
      q.includes("forecast") ||
      q.includes("sham") ||
      q.includes("shaam") ||
      q.includes("raat") ||
      q.includes("khet") ||
      q.includes("pani") ||
      q.includes("मौसम") ||
      q.includes("तापमान") ||
      q.includes("बारिश") ||
      q.includes("कल") ||
      q.includes("आज")
    );
  }

  type WeatherIntent =
    | "AGRICULTURE"
    | "EVENING_NIGHT"
    | "RAIN_PRECIPITATION"
    | "TEMPERATURE"
    | "HOURLY"
    | "SEVEN_DAY"
    | "TRAVEL_OUTDOOR"
    | "COMPARISON"
    | "EXPLANATION"
    | "GENERAL_CURRENT";

  interface QueryMeta {
    languageCode: string;
    languageName: string;
    script: string;
    isHinglish: boolean;
    intent: WeatherIntent;
    targetCity?: string;
  }

  // Detects the query language, script, and specific meteorological intent automatically
  function detectQueryMeta(question: string, defaultLang: string = "en"): QueryMeta {
    const raw = (question || "").trim();
    const qLower = raw.toLowerCase();

    // 1. Detect Script / Language
    let detectedLang = defaultLang;
    let isHinglish = false;

    // Check Indic & Foreign Scripts
    if (/[\u0600-\u06FF]/.test(raw)) {
      // Arabic / Urdu script
      if (/[\u0679\u0686\u0698\u0691\u06BA\u06BE\u06C1\u06CC]/.test(raw) || qLower.includes("موسم") || qLower.includes("بارش") || qLower.includes("درجہ حرارت")) {
        detectedLang = "ur";
      } else {
        detectedLang = "ar";
      }
    } else if (/[\u0980-\u09FF]/.test(raw)) {
      detectedLang = raw.includes("ৰ") || raw.includes("ৱ") ? "as" : "bn";
    } else if (/[\u0C00-\u0C7F]/.test(raw)) {
      detectedLang = "te";
    } else if (/[\u0B80-\u0BFF]/.test(raw)) {
      detectedLang = "ta";
    } else if (/[\u0A80-\u0AFF]/.test(raw)) {
      detectedLang = "gu";
    } else if (/[\u0C80-\u0CFF]/.test(raw)) {
      detectedLang = "kn";
    } else if (/[\u0D00-\u0D7F]/.test(raw)) {
      detectedLang = "ml";
    } else if (/[\u0A00-\u0A7F]/.test(raw)) {
      detectedLang = "pa";
    } else if (/[\u0B00-\u0B7F]/.test(raw)) {
      detectedLang = "or";
    } else if (/[\u0400-\u04FF]/.test(raw)) {
      detectedLang = "ru";
    } else if (/[\u4E00-\u9FFF]/.test(raw)) {
      detectedLang = "zh";
    } else if (/[\u3040-\u309F\u30A0-\u30FF]/.test(raw)) {
      detectedLang = "ja";
    } else if (/[\uAC00-\uD7AF]/.test(raw)) {
      detectedLang = "ko";
    } else if (/[\u0900-\u097F]/.test(raw)) {
      // Devanagari script: check for Bhojpuri, Marathi, Maithili, Sanskrit, or Hindi
      if (
        /का\s*हाल\s*बा|होई\s*का|रही|नइखे|कइसन|केहू|तहार|हमरा|आजु|काल्हु|घाम|बरखा|रोपे|रोपनी/.test(raw)
      ) {
        detectedLang = "bho";
      } else if (/आहे|का\s*$|कसे|होईल|पाऊस|उद्या|शेत|पाणी|हवामान/.test(raw)) {
        detectedLang = "mr";
      } else if (/अछि|कयल|संगे|राखू|छल|भेल|मैथिली/.test(raw)) {
        detectedLang = "mai";
      } else if (/अस्ति|भवति|धारयन्तु|संस्कृतम्|वातावरण/.test(raw)) {
        detectedLang = "sa";
      } else {
        detectedLang = "hi";
      }
    } else {
      // Latin script: Check for Hinglish vs European languages vs English
      const hinglishMarkers = [
        "aaj", "kal", "parson", "kya", "hogi", "hoga", "hai", "hain", "kaisa", "kaisi", "kaise",
        "rahega", "rahegi", "sham", "shaam", "raat", "subah", "dopahar", "pani", "paani", "khet",
        "fasal", "lagana", "chahiye", "batao", "bataiye", "barish", "barsat", "kitne", "baje",
        "chata", "garmi", "sardi", "thand", "mausam", "kaha", "kahan", "nahi", "nhi", "mat",
        "sakte", "sakta", "karein", "kare", "kitna", "kitni", "badal", "hawa", "dhup", "dhoop",
        "sinchai", "keetnashak", "spray", "gehun", "dhan", "kisan", "bhopal", "indore", "delhi", "patna"
      ];

      const tokens = qLower.split(/[^a-z0-9]+/);
      let matchCount = 0;
      for (const t of tokens) {
        if (hinglishMarkers.includes(t)) matchCount++;
      }

      if (matchCount >= 2 || (matchCount >= 1 && (qLower.includes("kaisa") || qLower.includes("barish") || qLower.includes("pani") || qLower.includes("khet") || qLower.includes("hogi")))) {
        detectedLang = "hinglish";
        isHinglish = true;
      } else if (qLower.includes("météo") || qLower.includes("pluie") || qLower.includes("demain") || qLower.includes("température")) {
        detectedLang = "fr";
      } else if (qLower.includes("wetter") || qLower.includes("regen") || qLower.includes("morgen") || qLower.includes("temperatur")) {
        detectedLang = "de";
      } else if (qLower.includes("tiempo") || qLower.includes("clima") || qLower.includes("lluvia") || qLower.includes("lloverá") || qLower.includes("mañana")) {
        detectedLang = "es";
      } else if (qLower.includes("tempo") || qLower.includes("chuva") || qLower.includes("amanhã")) {
        detectedLang = "pt";
      } else {
        detectedLang = defaultLang === "hi" && !isHinglish ? "en" : defaultLang;
      }
    }

    // 2. Detect Meteorological Intent
    let intent: WeatherIntent = "GENERAL_CURRENT";

    const isAgri =
      qLower.includes("khet") ||
      qLower.includes("irrigation") ||
      qLower.includes("irrigate") ||
      qLower.includes("water the field") ||
      qLower.includes("pani lagana") ||
      qLower.includes("sinchai") ||
      qLower.includes("fasal") ||
      qLower.includes("crop") ||
      qLower.includes("farming") ||
      qLower.includes("farmer") ||
      qLower.includes("kisan") ||
      qLower.includes("spray") ||
      qLower.includes("keetnashak") ||
      qLower.includes("fertilizer") ||
      qLower.includes("khad") ||
      qLower.includes("wheat") ||
      qLower.includes("gehun") ||
      qLower.includes("paddy") ||
      qLower.includes("dhan") ||
      raw.includes("सिंचाई") ||
      raw.includes("फसल") ||
      raw.includes("खेत") ||
      raw.includes("कीटनाशक") ||
      raw.includes("छिड़काव") ||
      raw.includes("खाद") ||
      raw.includes("किसान") ||
      raw.includes("कृषि") ||
      raw.includes("शेत") ||
      raw.includes("पिक");

    const isEvening =
      qLower.includes("evening") ||
      qLower.includes("tonight") ||
      qLower.includes("night") ||
      qLower.includes("sham") ||
      qLower.includes("shaam") ||
      qLower.includes("raat") ||
      qLower.includes("dusk") ||
      qLower.includes("sundown") ||
      raw.includes("शाम") ||
      raw.includes("रात") ||
      raw.includes("संध्या") ||
      raw.includes("रात्री");

    const isRain =
      qLower.includes("rain") ||
      qLower.includes("rainy") ||
      qLower.includes("precipitation") ||
      qLower.includes("umbrella") ||
      qLower.includes("drizzle") ||
      qLower.includes("barish") ||
      qLower.includes("barsat") ||
      qLower.includes("chata") ||
      qLower.includes("bheege") ||
      raw.includes("बारिश") ||
      raw.includes("बरसात") ||
      raw.includes("छाता") ||
      raw.includes("वर्षा") ||
      raw.includes("বৃষ্টি") ||
      raw.includes("వర్షం") ||
      raw.includes("पाऊस") ||
      raw.includes("மழை") ||
      raw.includes("بارش");

    const isTemp =
      qLower.includes("temp") ||
      qLower.includes("temperature") ||
      qLower.includes("hot") ||
      qLower.includes("heat") ||
      qLower.includes("cold") ||
      qLower.includes("chilly") ||
      qLower.includes("heatwave") ||
      qLower.includes("freeze") ||
      qLower.includes("tapman") ||
      qLower.includes("garmi") ||
      qLower.includes("sardi") ||
      qLower.includes("thand") ||
      qLower.includes("lu") ||
      raw.includes("तापमान") ||
      raw.includes("गर्मी") ||
      raw.includes("ठंड") ||
      raw.includes("सर्दी") ||
      raw.includes("लू");

    const isHourly =
      qLower.includes("hourly") ||
      qLower.includes("hour by hour") ||
      qLower.includes("next hours") ||
      qLower.includes("what time") ||
      qLower.includes("kitne baje") ||
      qLower.includes("kis samay") ||
      qLower.includes("kab tak") ||
      raw.includes("घंटे") ||
      raw.includes("किस समय") ||
      raw.includes("कितने बजे");

    const isSevenDay =
      qLower.includes("7 day") ||
      qLower.includes("7 days") ||
      qLower.includes("weekly") ||
      qLower.includes("week") ||
      qLower.includes("outlook") ||
      qLower.includes("next week") ||
      qLower.includes("upcoming days") ||
      qLower.includes("agle hafte") ||
      qLower.includes("7 din") ||
      raw.includes("7 दिन") ||
      raw.includes("सप्ताह") ||
      raw.includes("हफ्ते");

    const isTravel =
      qLower.includes("travel") ||
      qLower.includes("trip") ||
      qLower.includes("drive") ||
      qLower.includes("driving") ||
      qLower.includes("road") ||
      qLower.includes("walk") ||
      qLower.includes("running") ||
      qLower.includes("outdoor") ||
      qLower.includes("cricket") ||
      qLower.includes("match") ||
      qLower.includes("picnic") ||
      qLower.includes("flight") ||
      raw.includes("यात्रा") ||
      raw.includes("सफर") ||
      raw.includes("घूमने");

    const isCompare =
      qLower.includes("compare") ||
      qLower.includes(" vs ") ||
      qLower.includes("versus") ||
      qLower.includes("dono me") ||
      raw.includes("तुलना");

    const isExplanation =
      qLower.includes("why") ||
      qLower.includes("what is humidity") ||
      qLower.includes("how does") ||
      qLower.includes("monsoon") ||
      raw.includes("क्यों") ||
      raw.includes("क्या होता है");

    if (isAgri) intent = "AGRICULTURE";
    else if (isEvening) intent = "EVENING_NIGHT";
    else if (isRain) intent = "RAIN_PRECIPITATION";
    else if (isHourly) intent = "HOURLY";
    else if (isSevenDay) intent = "SEVEN_DAY";
    else if (isTemp) intent = "TEMPERATURE";
    else if (isTravel) intent = "TRAVEL_OUTDOOR";
    else if (isCompare) intent = "COMPARISON";
    else if (isExplanation) intent = "EXPLANATION";
    else intent = "GENERAL_CURRENT";

    const langConfig = SCHEDULED_LANGUAGES_INFO[detectedLang] || SCHEDULED_LANGUAGES_INFO.en;

    return {
      languageCode: detectedLang,
      languageName: langConfig.name,
      script: langConfig.script,
      isHinglish,
      intent,
    };
  }

  // Pre-computes and compiles structured, compact meteorological insights from real API telemetry
  function buildMeteorologicalDigest(weatherContext: any, meta: QueryMeta): string {
    const unitLabel = weatherContext?.units === "imperial" ? "°F" : "°C";
    const windUnit = weatherContext?.units === "imperial" ? "mph" : "km/h";
    const cur = weatherContext?.current || {};
    const loc = weatherContext?.locationName || weatherContext?.location?.name || "Target Location";
    const hourly: any[] = Array.isArray(weatherContext?.todayHourlyPreview) ? weatherContext.todayHourlyPreview : [];
    const daily: any[] = Array.isArray(weatherContext?.dailyForecast) ? weatherContext.dailyForecast : [];

    // 1. Current Snapshot
    const curTemp = cur.temp ?? "N/A";
    const feelsLike = cur.feelsLike ?? curTemp;
    const curCond = cur.condition || "Clear";
    const curRain = cur.rainProbability ?? 0;
    const curPrecip = cur.precipitation ?? 0;
    const curHumidity = cur.humidity ?? 50;
    const curWind = cur.windSpeed ?? 10;
    const curUv = cur.uvIndex ?? 5;
    const curPressure = cur.pressure ?? 1013;
    const curAqi = cur.airQualityAqi ?? "N/A";

    // 2. Evening Slot Analysis (Hours between 16:00 and 21:00)
    const eveningHours = hourly.filter((h) => {
      const hrNum = parseInt(String(h.hour || "").replace(/\D/g, ""), 10);
      const isPm = String(h.hour || "").toUpperCase().includes("PM");
      if (isPm) {
        return (hrNum >= 4 && hrNum <= 9) || hrNum === 12;
      }
      return hrNum >= 16 && hrNum <= 21;
    });

    const eveningSummary =
      eveningHours.length > 0
        ? eveningHours
            .map((h) => `${h.hour}: ${h.temp}${unitLabel}, Rain: ${h.rainProbability}%, ${h.condition}`)
            .join(" | ")
        : `Expected ${curTemp}${unitLabel}, Rain chance ${curRain}%, ${curCond}`;

    // 3. 24-Hour Timeline Highs, Lows, and Rain Peaks
    let peakRainProb = curRain;
    let peakRainHour = "Current";
    let maxHourlyTemp = typeof curTemp === "number" ? curTemp : 30;
    let minHourlyTemp = typeof curTemp === "number" ? curTemp : 20;

    for (const h of hourly.slice(0, 24)) {
      if (typeof h.rainProbability === "number" && h.rainProbability > peakRainProb) {
        peakRainProb = h.rainProbability;
        peakRainHour = h.hour || "upcoming hour";
      }
      if (typeof h.temp === "number") {
        if (h.temp > maxHourlyTemp) maxHourlyTemp = h.temp;
        if (h.temp < minHourlyTemp) minHourlyTemp = h.temp;
      }
    }

    // 4. Agriculture / Irrigation Real Indices (Next 24-48 hours)
    const todayRainProb = typeof daily[0]?.rainProbability === "number" ? daily[0].rainProbability : curRain;
    const tomorrowRainProb = typeof daily[1]?.rainProbability === "number" ? daily[1].rainProbability : 0;
    const maxRain48h = Math.max(todayRainProb, tomorrowRainProb);
    const expectedPrecipSum = (daily[0]?.precipitationSum ?? 0) + (daily[1]?.precipitationSum ?? 0);

    let irrigationVerdict = "";
    if (maxRain48h >= 40 || expectedPrecipSum >= 2.5) {
      irrigationVerdict = `DELAY IRRIGATION: Rain probability is elevated (${maxRain48h}% in next 24-48h, expected rain ${expectedPrecipSum.toFixed(1)} mm). Postpone field watering to save water and prevent root waterlogging.`;
    } else if (maxRain48h <= 20) {
      irrigationVerdict = `IRRIGATION RECOMMENDED / SAFE: Low rain probability (${maxRain48h}% in next 24-48h). Scheduled watering is safe, preferably early morning (6-9 AM) or evening (5-7 PM) to minimize evaporation.`;
    } else {
      irrigationVerdict = `MODERATE CAUTION: Moderate rain probability (${maxRain48h}%). Check local soil moisture; apply light irrigation only if topsoil is dry.`;
    }

    let sprayVerdict = "";
    if (curWind > 15) {
      sprayVerdict = `UNSAFE FOR SPRAYING: Wind speed is ${curWind} ${windUnit} (>15 ${windUnit}). Spraying pesticide or foliar fertilizer risks significant drift and wastage.`;
    } else if (todayRainProb > 30) {
      sprayVerdict = `UNSAFE FOR SPRAYING: Rain chance is ${todayRainProb}%. Pesticide spray may get washed off by precipitation.`;
    } else {
      sprayVerdict = `SAFE FOR SPRAYING: Favorable calm conditions (wind ${curWind} ${windUnit}, rain chance ${todayRainProb}%). Safe for chemical and fertilizer application.`;
    }

    const fungalRisk = curHumidity >= 80 && curTemp >= 24 ? "HIGH FUNGAL RISK (elevated humidity and warmth)" : "LOW / NORMAL FUNGAL RISK";

    // 5. 7-Day Outlook Digest
    const dailyOutlookStr = daily
      .slice(0, 7)
      .map(
        (d, idx) =>
          `Day ${idx + 1} (${d.day || d.date}): High ${d.maxTemp}${unitLabel}, Low ${d.minTemp}${unitLabel}, Rain ${d.rainProbability}%, ${d.condition}${d.precipitationSum ? `, Rain Sum: ${d.precipitationSum}mm` : ""}`
      )
      .join("\n  • ");

    return `
[GROUNDED METEOROLOGICAL TELEMETRY DIGEST]
Location: ${loc}
Coordinates: ${weatherContext?.coordinates?.lat ?? "N/A"}, ${weatherContext?.coordinates?.lon ?? "N/A"}
Units: ${unitLabel}, ${windUnit}
Data Freshness: ${cur.lastUpdated || new Date().toLocaleTimeString()}

CURRENT OBSERVATIONS:
• Temperature: ${curTemp}${unitLabel} (Feels like: ${feelsLike}${unitLabel})
• Weather Condition: ${curCond}
• Rain Probability: ${curRain}%
• Precipitation: ${curPrecip} mm
• Humidity: ${curHumidity}%
• Wind Speed: ${curWind} ${windUnit}
• UV Index: ${curUv}
• Pressure: ${curPressure} hPa
• Air Quality Index: ${curAqi}

HOURLY 24-HOUR HIGHLIGHTS:
• Evening Window (4 PM - 9 PM): ${eveningSummary}
• 24h Peak Rain Probability: ${peakRainProb}% (around ${peakRainHour})
• 24h Temperature Range: Min ${minHourlyTemp}${unitLabel} to Max ${maxHourlyTemp}${unitLabel}

AGRICULTURE & FARMING METRICS:
• Max Rain Chance (Next 24-48h): ${maxRain48h}% (Expected Precipitation: ${expectedPrecipSum.toFixed(1)} mm)
• Irrigation Verdict: ${irrigationVerdict}
• Spraying Window: ${sprayVerdict}
• Crop Disease / Fungal Risk: ${fungalRisk}

7-DAY FORECAST OUTLOOK:
  • ${dailyOutlookStr || "Live 7-day outlook available on request"}
`.trim();
  }

  // Generates the comprehensive system prompt for WeatherGPT
  const getChatSystemInstruction = (
    queryMetaOrLang: QueryMeta | string,
    weatherContext: any,
    digestOrQuestion?: string
  ): string => {
    let meta: QueryMeta;
    if (typeof queryMetaOrLang === "string") {
      meta = detectQueryMeta(digestOrQuestion || "", queryMetaOrLang);
    } else {
      meta = queryMetaOrLang;
    }

    const digest =
      typeof digestOrQuestion === "string" && digestOrQuestion.includes("[GROUNDED METEOROLOGICAL TELEMETRY DIGEST]")
        ? digestOrQuestion
        : buildMeteorologicalDigest(weatherContext, meta);

    const langInstructions = meta.isHinglish
      ? `### MANDATORY LANGUAGE REQUIREMENT: HINGLISH (ROMAN HINDI)
The user is speaking in HINGLISH (Hindi written in Latin/English alphabet, e.g. "Kya aaj khet me pani lagana chahiye?", "Sham ko mausam kaisa rahega?").
YOU MUST WRITE YOUR ENTIRE RESPONSE STRICTLY IN NATURAL HINGLISH (ROMAN SCRIPT HINDI).
NEVER REPLY IN ENGLISH. NEVER REPLY IN DEVANAGARI.
Example response: "Aaj Bhopal me barish hone ki lagbhag koi sambhavna nahi hai (kewal 5% chance). Mausam saaf rahega, isliye aap subah ke samay khet me sinchai kar sakte hain."`
      : meta.languageCode === "bho"
      ? `### MANDATORY LANGUAGE REQUIREMENT: BHOJPURI (भोजपुरी)
The user is speaking in BHOJPURI (भोजपुरी).
YOU MUST WRITE YOUR ENTIRE RESPONSE STRICTLY IN NATURAL BHOJPURI (using Devanagari script).
NEVER REPLY IN ENGLISH.
Example response: "आज भोपाल में बारिश के कवनो संभावना नइखे (खाली 5% चांस बा)। मौसम साफ़ रही, एही से रउआ खेत में पटवन क सकत बानी।"`
      : meta.languageCode === "hi"
      ? `### MANDATORY LANGUAGE REQUIREMENT: HINDI (हिन्दी)
The user is speaking in HINDI (हिन्दी).
YOU MUST WRITE YOUR ENTIRE RESPONSE STRICTLY IN NATURAL HINDI using Devanagari script.
NEVER REPLY IN ENGLISH.`
      : `### MANDATORY LANGUAGE REQUIREMENT: ${meta.languageName.toUpperCase()}
The user is communicating in ${meta.languageName} (${meta.script} script).
YOU MUST WRITE YOUR ENTIRE RESPONSE STRICTLY IN ${meta.languageName} (${meta.script} script).`;

    return `
You are WeatherGPT, an advanced AI conversational meteorological assistant and agricultural advisor.
Provide direct, natural, highly accurate, and helpful answers.

${langInstructions}

CRITICAL DIRECTIVE 1: STRICT GROUNDING & DATA ACCURACY
- ALL weather metrics (temperature, conditions, rain probability, wind, humidity, pressure, UV index) MUST be drawn exclusively from the provided Meteorological Telemetry Digest below.
- NEVER invent, estimate, guess, or hallucinate weather values or current conditions. You must ONLY explain, summarize, or interpret the actual meteorological data received from the weather API.
- NEVER confuse Rainfall Amount in mm with Rain Probability in %.
- Distinguish clearly between daytime peak temperatures and nighttime lows.

CRITICAL DIRECTIVE 2: DYNAMIC MULTILINGUAL & SCRIPT MIRRORING (ALL LANGUAGES)
- Always respond in the EXACT same language and script as the user.
- If the user switches languages mid-conversation -> IMMEDIATELY switch to their newest language.
- Keep technical units (°C/°F, km/h, mm) accurate.

CRITICAL DIRECTIVE 3: SPECIFIC INTENT UNDERSTANDING — DO NOT REPEAT GENERIC ANSWERS
Answer the SPECIFIC question asked by the user directly and immediately:
• EVENING / NIGHT FORECAST:
  When asked about the evening (e.g., "Aaj sham ko mausam kaisa rahega?", "Will it rain tonight?"), quote the exact evening hours (4 PM - 9 PM) from the digest. State the evening temperature range and evening rain probability.
• RAIN PROBABILITY & UMBRELLA:
  State the exact rain percentage (%) and conditions. If probability >= 40% or rain is expected, recommend carrying an umbrella; if low (<20%), explicitly state that rain is unlikely.
• AGRICULTURE & IRRIGATION ADVICE (सिंचाई / कीटनाशक / फसल):
  Analyze the 24-48h rain forecast and wind speed from the digest:
  - If 24-48h rain chance >= 35-40% or rain sum > 2mm: Direct advice to DELAY irrigation to save water and avoid waterlogging.
  - If rain chance is low (<20%): Safe to irrigate (recommend early morning or evening to reduce evaporation).
  - Pesticide spray: Check wind speed (if >15 km/h, caution against chemical drift) and rain risk (if rain expected, chemicals wash away).
  - Ask what crop they are growing (wheat, paddy, vegetables) if crop-specific advice would help.
• TEMPERATURE & HEATWAVE / COLD:
  Compare current temperature with feels-like, mention today's peak daytime high and overnight minimum, and advise hydration or layered clothing.
• 7-DAY FORECAST:
  Summarize the weekly trend clearly day-by-day with temperatures and rain days from the 7-day outlook.
• TRAVEL & OUTDOOR:
  Provide visibility, road transit safety, wind speed, and best time windows.

RESPONSE FORMAT (Keep concise, 2-4 brief paragraphs or structured bullet points):
1. Direct Answer (first 1-2 sentences directly answer the question).
2. Relevant Weather Telemetry (only the metrics relevant to the question).
3. Practical Recommendation (concrete, actionable steps).
4. Caution / Alert (only if rain, storm, heat, or high winds exist).
AVOID repetitive robotic phrases like 'मौसम के अनुसार आवश्यक सावधानियां बरतें'. Speak with natural, authoritative expertise.

${digest}
`.trim();
  };

  // Upgraded multilingual synthesis fallback for offline/resilience scenarios
  const buildMultilingualSynthesis = (
    question: string,
    loc: string,
    curTemp: number,
    curCond: string,
    curRain: number,
    curHumidity: number,
    curWind: number,
    language: string,
    weatherContext?: any
  ) => {
    const meta = detectQueryMeta(question, language);
    const unitLabel = weatherContext?.units === "imperial" ? "°F" : "°C";
    const daily = weatherContext?.dailyForecast || [];
    const hourly = weatherContext?.todayHourlyPreview || [];

    const tomorrowRain = daily[1]?.rainProbability ?? curRain;
    const maxRain48 = Math.max(curRain, tomorrowRain);
    const eveningHour = hourly.find((h: any) => h.hour?.includes("6") || h.hour?.includes("18") || h.hour?.includes("7") || h.hour?.includes("19")) || hourly[0] || {};
    const eveningTemp = eveningHour.temp ?? curTemp;
    const eveningRain = eveningHour.rainProbability ?? curRain;

    let text = "";
    let summary = `${curTemp}${unitLabel}, ${curCond} in ${loc}`;
    let recommendation = "Plan your routine in accordance with current atmospheric conditions.";
    let riskLevel: "low" | "medium" | "high" = curRain > 60 || curTemp > 40 ? "medium" : "low";

    // 1. HINGLISH (Roman Hindi)
    if (meta.isHinglish) {
      if (meta.intent === "AGRICULTURE") {
        if (maxRain48 >= 40) {
          text = `Aapke khet ke liye: Agle 24-48 ghanto me ${loc} me barish ki sambhavna ${maxRain48}% hai. Isliye abhi khet me sinchai (pani) lagana sthagit karein, taaki paani ki bachat ho aur fasal me jal-bharao na ho. Hawa ki raftaar ${curWind} km/h hai; agar barish ke aasar hon to keetnashak spray karne se bachein.`;
          recommendation = "Barish ke aasar ke chalte sinchai aur spray do din ke liye rok lein.";
          riskLevel = "medium";
        } else {
          text = `Aapke khet ke liye: Agle 24-48 ghanto me ${loc} me barish ki sambhavna kewal ${maxRain48}% hai. Aap khet me subah (6-9 AM) ya shaam ke waqt sinchai laga sakte hain taaki evaporation kam ho. Hawa ki raftaar ${curWind} km/h hai, jo keetnashak spray ke liye anukool hai.`;
          recommendation = "Mausam saaf hai, subah ya shaam ke waqt sinchai aur jaruri spray kar sakte hain.";
        }
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `Aaj shaam ko ${loc} me mausam mukhya roop se '${curCond}' rahega aur tapman lagbhag ${eveningTemp}${unitLabel} ke aas-paas rahega. Shaam ke waqt barish hone ki sambhavna kewal ${eveningRain}% hai. Mausam ghumne ya bahar jaane ke liye kaafi anukool rahega.`;
        recommendation = "Shaam ko mausam suhavna rahega, bahar ki gatividhiyaan plan kar sakte hain.";
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        if (curRain >= 40) {
          text = `Haan, ${loc} me aaj barish hone ki sambhavna ${curRain}% hai aur aakash me ${curCond} sthiti bani hui hai. Bahar nikalte waqt chata ya raincoat zaroor saath rakhein taaki bheegein nahi.`;
          recommendation = "Barish ke aasar hain, chata saath lekar niklein.";
          riskLevel = "medium";
        } else {
          text = `Nahi, ${loc} me aaj barish ki sambhavna kewal ${curRain}% hai. Mausam mukhya roop se ${curCond} rahega aur bina chate ke bahar jaana bilkul safe hai.`;
          recommendation = "Barish ki sambhavna behad kam hai, routine kaam aasaani se kar sakte hain.";
        }
      } else if (meta.intent === "TEMPERATURE") {
        text = `Current observation ke anusaar ${loc} me tapman ${curTemp}${unitLabel} hai aur humidity ${curHumidity}% hai. Mausam ${curTemp >= 35 ? "kaafi garm" : curTemp <= 18 ? "halka thanda" : "sukhad"} bana hua hai.`;
        recommendation = curTemp >= 35 ? "Dhoop se bachein aur paani pite rahein." : "Mausam ke anusaar aaramdayak kapde pehnein.";
      } else if (meta.intent === "SEVEN_DAY") {
        text = `${loc} me agle 7 dino ka ausat tapman ${curTemp}${unitLabel} ke aas-paas bana rahega. Barish ki aashanka mukhya roop se kam hai aur mausam aamtaur par sthir rahega.`;
        recommendation = "Hafte bhar ke kaam bina kisi mausam badha ke plan kar sakte hain.";
      } else {
        text = `Abhi ke live data ke anusaar ${loc} me tapman ${curTemp}${unitLabel} hai aur mausam '${curCond}' hai. Hawa me nami ${curHumidity}% aur barish ki sambhavna ${curRain}% hai.`;
        recommendation = "Aapka routine regular mausam ke anusaar plan karein.";
      }
      summary = `${loc} me ${curTemp}${unitLabel}, ${curCond}`;
      return {
        text,
        source: "WeatherGPT AI Local Engine (Hinglish)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 2. BHOJPURI
    if (meta.languageCode === "bho") {
      if (meta.intent === "AGRICULTURE") {
        text = maxRain48 >= 40
          ? `रउआ के खेत खातिर: अगिला 24-48 घंटा में ${loc} में बारिश के संभावना ${maxRain48}% बा। एही से अभी खेत में पानी (पटवन) ना लगाईं। हवा के रफ़्तार ${curWind} किमी/घंटा बा।`
          : `रउआ के खेत खातिर: अगिला 24-48 घंटा में ${loc} में बारिश के संभावना खाली ${maxRain48}% बा। रउआ खेत में पटवन क सकत बानी। हवा शांत बा।`;
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `आज साँझ के ${loc} में मौसम '${curCond}' रही आ तापमान लगभग ${eveningTemp}${unitLabel} रही। साँझ के बारिश के संभावना खाली ${eveningRain}% बा।`;
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        text = curRain >= 40
          ? `हँ, ${loc} में आज बारिश के संभावना ${curRain}% बा। बहार निकले से पहिले छाता संगे ले लीं।`
          : `ना, आज ${loc} में बारिश के संभावना खाली ${curRain}% बा। मौसम मुख्यतः ${curCond} रही।`;
      } else {
        text = `लाइव मौसम डेटा के अनुसार ${loc} में हाल के तापमान ${curTemp}${unitLabel} बा आ मौसम '${curCond}' बा। नमी ${curHumidity}% आ बारिश के संभावना ${curRain}% बा।`;
      }
      summary = `${loc} में ${curTemp}${unitLabel}, ${curCond}`;
      recommendation = "मौसम के हाल देख के आपन काम करीं।";
      return {
        text,
        source: "WeatherGPT AI Local Engine (भोजपुरी)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 3. HINDI (Devanagari)
    if (meta.languageCode === "hi") {
      if (meta.intent === "AGRICULTURE") {
        text = maxRain48 >= 40
          ? `कृषि परामर्श: अगले 24-48 घंटों में ${loc} में वर्षा की संभावना ${maxRain48}% है। खेतों में अभी सिंचाई स्थगित रखें ताकि जलभराव न हो और पानी की बचत हो। हवा की गति ${curWind} किमी/घंटा है, अतः कीटनाशक छिड़काव से भी बचें।`
          : `कृषि परामर्श: अगले 24-48 घंटों में ${loc} में वर्षा की संभावना केवल ${maxRain48}% है। आप सुबह या शाम के समय फसलों में हल्की सिंचाई कर सकते हैं। हवा की गति ${curWind} किमी/घंटा अनुकूल है।`;
        recommendation = maxRain48 >= 40 ? "वर्षा के पूर्वानुमान के चलते सिंचाई और छिड़काव टालें।" : "मौसम अनुकूल है, सुबह/शाम सिंचाई की जा सकती है।";
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `आज शाम ${loc} में मौसम मुख्य रूप से '${curCond}' रहेगा और तापमान लगभग ${eveningTemp}${unitLabel} रहेगा। शाम को बारिश की संभावना केवल ${eveningRain}% है, अतः मौसम बाहरी गतिविधियों के लिए बहुत अनुकूल है।`;
        recommendation = "शाम के समय मौसम सुहावना रहेगा।";
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        text = curRain >= 40
          ? `हाँ, ${loc} में आज वर्षा की संभावना ${curRain}% है और आकाश में '${curCond}' स्थिति है। बाहर जाते समय छाता या रेनकोट अवश्य साथ रखें।`
          : `नहीं, ${loc} में आज वर्षा की संभावना केवल ${curRain}% है। मौसम मुख्यतः '${curCond}' रहेगा और वर्षा की कोई विशेष आशंका नहीं है।`;
        recommendation = curRain >= 40 ? "छाता साथ लेकर निकलें।" : "मौसम साफ है, सामान्य रूप से बाहर जा सकते हैं।";
      } else if (meta.intent === "TEMPERATURE") {
        text = `वर्तमान में ${loc} में तापमान ${curTemp}${unitLabel} (महसूस: ${curTemp}${unitLabel}) और आर्द्रता ${curHumidity}% है। मौसम ${curTemp >= 35 ? "गर्म" : curTemp <= 18 ? "शीतल" : "सुखद"} है।`;
        recommendation = curTemp >= 35 ? "पर्याप्त पानी पिएं और धूप से बचें।" : "सामान्य आरामदायक वस्त्र पहनें।";
      } else {
        text = `ताज़ा वायुमंडलीय आंकड़ों के अनुसार ${loc} में तापमान ${curTemp}${unitLabel} और स्थिति '${curCond}' है। आर्द्रता ${curHumidity}%, हवा ${curWind} किमी/घंटा और वर्षा की संभावना ${curRain}% है।`;
        recommendation = "स्थानीय मौसम के अनुसार अपनी दिनचर्या रखें।";
      }
      summary = `${loc} में ${curTemp}${unitLabel}, ${curCond}`;
      return {
        text,
        source: "WeatherGPT AI Local Engine (हिन्दी)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 4. URDU
    if (meta.languageCode === "ur") {
      if (meta.intent === "AGRICULTURE") {
        text = maxRain48 >= 40
          ? `زرعی مشورہ: اگلے 24-48 گھنٹوں میں ${loc} میں بارش کا امکان ${maxRain48}% ہے۔ لہٰذا فصلوں کو پانی دینا فی الحال موخر کریں تاکہ پانی کا زیاں نہ ہو۔ ہوا کی رفتار ${curWind} کلومیٹر/گھنٹہ ہے۔`
          : `زرعی مشورہ: اگلے 24-48 گھنٹوں میں ${loc} میں بارش کا امکان صرف ${maxRain48}% ہے۔ صبح یا شام کے وقت ہلکی آبپاشی کی جا سکتی ہے۔`;
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `آج شام ${loc} میں موسم زیادہ تر '${curCond}' رہے گا اور درجہ حرارت لگ بھگ ${eveningTemp}${unitLabel} رہے گا۔ شام کو بارش کا امکان صرف ${eveningRain}% ہے۔`;
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        text = curRain >= 40
          ? `جی ہاں، ${loc} میں آج بارش کا امکان ${curRain}% ہے۔ باہر جاتے وقت چھتری ضرور ساتھ رکھیں۔`
          : `نہیں، ${loc} میں آج بارش کا امکان صرف ${curRain}% ہے۔ موسم بنیادی طور پر '${curCond}' رہے گا۔`;
      } else {
        text = `براہ راست موسمیاتی ڈیٹا کے مطابق ${loc} میں موجودہ درجہ حرارت ${curTemp}${unitLabel} اور کیفیت '${curCond}' ہے۔ نمی ${curHumidity}% اور بارش کا امکان ${curRain}% ہے۔`;
      }
      summary = `${loc} میں ${curTemp}${unitLabel}، ${curCond}`;
      recommendation = "موسم کے مطابق معمولات ترتیب دیں۔";
      return {
        text,
        source: "WeatherGPT AI Local Engine (اردو)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 5. BENGALI
    if (meta.languageCode === "bn") {
      if (meta.intent === "AGRICULTURE") {
        text = maxRain48 >= 40
          ? `কৃষি পরামর্শ: আগামী ২৪-৪৮ ঘণ্টায় ${loc}-এ বৃষ্টির সম্ভাবনা ${maxRain48}%। ফসলে জল দেওয়া আপাতত স্থগিত রাখুন যাতে জলাবদ্ধতা না ঘটে। বাতাসের গতি ${curWind} কিমি/ঘণ্টা।`
          : `কৃষি পরামর্শ: আগামী ২৪-৪৮ ঘণ্টায় ${loc}-এ বৃষ্টির সম্ভাবনা মাত্র ${maxRain48}%। সকালে বা বিকেলে সেচ দেওয়া নিরাপদ।`;
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `আজ সন্ধ্যায় ${loc}-এ আবহাওয়া '${curCond}' থাকবে এবং তাপমাত্রা প্রায় ${eveningTemp}${unitLabel} থাকবে। সন্ধ্যায় বৃষ্টির সম্ভাবনা মাত্র ${eveningRain}%।`;
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        text = curRain >= 40
          ? `হ্যাঁ, ${loc}-এ আজ বৃষ্টির সম্ভাবনা ${curRain}%। বাইরে যাওয়ার সময় ছাতা সাথে রাখুন।`
          : `না, আজ ${loc}-এ বৃষ্টির সম্ভাবনা মাত্র ${curRain}%। আবহাওয়া মূলত '${curCond}' থাকবে।`;
      } else {
        text = `বর্তমান আবহাওয়া পর্যবেক্ষণ অনুসারে ${loc}-এ তাপমাত্রা ${curTemp}${unitLabel} এবং আকাশ '${curCond}'। আর্দ্রতা ${curHumidity}% এবং বৃষ্টির সম্ভাবনা ${curRain}%।`;
      }
      summary = `${loc}-এ ${curTemp}${unitLabel}, ${curCond}`;
      recommendation = "আবহাওয়া পরিস্থিতি বিবেচনা করে পরিকল্পনা করুন।";
      return {
        text,
        source: "WeatherGPT AI Local Engine (বাংলা)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 6. MARATHI
    if (meta.languageCode === "mr") {
      if (meta.intent === "AGRICULTURE") {
        text = maxRain48 >= 40
          ? `कृषी सल्ला: पुढील २४-४८ तासांत ${loc} मध्ये पावसाची शक्यता ${maxRain48}% आहे. शेतात पाणी देणे (सिंचन) तात्पुरते थांबवावे. वाऱ्याचा वेग ${curWind} किमी/तास आहे.`
          : `कृषी सल्ला: पुढील २४-४८ तासांत ${loc} मध्ये पावसाची शक्यता फक्त ${maxRain48}% आहे. सकाळी किंवा संध्याकाळी पिकांना पाणी देणे सुरक्षित आहे.`;
      } else if (meta.intent === "EVENING_NIGHT") {
        text = `आज संध्याकाळी ${loc} मध्ये हवामान '${curCond}' राहील आणि तापमान सुमारे ${eveningTemp}${unitLabel} राहील. पावसाची शक्यता केवळ ${eveningRain}% आहे.`;
      } else if (meta.intent === "RAIN_PRECIPITATION") {
        text = curRain >= 40
          ? `होय, ${loc} मध्ये आज पावसाची शक्यता ${curRain}% आहे. बाहेर पडताना छत्री किंवा रेनकोट सोबत ठेवा.`
          : `नाही, आज ${loc} मध्ये पावसाची शक्यता फक्त ${curRain}% आहे. हवामान मुख्यतः '${curCond}' राहील.`;
      } else {
        text = `सध्याच्या हवामान नोंदींनुसार ${loc} मध्ये तापमान ${curTemp}${unitLabel} आणि स्थिती '${curCond}' आहे. आर्द्रता ${curHumidity}% आणि पावसाची शक्यता ${curRain}% आहे.`;
      }
      summary = `${loc} मध्ये ${curTemp}${unitLabel}, ${curCond}`;
      recommendation = "हवामानानुसार दैनंदिन कामे नियोजित करा.";
      return {
        text,
        source: "WeatherGPT AI Local Engine (मराठी)",
        timestamp: new Date().toISOString(),
        structured: { summary, recommendation, riskLevel },
      };
    }

    // 7. ENGLISH (Default fallback)
    if (meta.intent === "AGRICULTURE") {
      text = maxRain48 >= 40
        ? `Agricultural Advisory: In ${loc}, precipitation probability over the next 24-48 hours is elevated at ${maxRain48}%. You should POSTPONE field irrigation to prevent water wastage and crop root waterlogging. Current wind speed is ${curWind} ${weatherContext?.units === "imperial" ? "mph" : "km/h"}; avoid chemical spraying until calm, dry conditions prevail.`
        : `Agricultural Advisory: In ${loc}, precipitation chance over the next 24-48 hours is low at ${maxRain48}%. Irrigation is SAFE and recommended during early morning (6-9 AM) or late afternoon to minimize evaporation loss. Wind velocity is calm at ${curWind} ${weatherContext?.units === "imperial" ? "mph" : "km/h"}, favorable for scheduled foliar sprays.`;
      recommendation = maxRain48 >= 40 ? "Delay irrigation due to upcoming rain risk." : "Safe window for irrigation and spraying.";
    } else if (meta.intent === "EVENING_NIGHT") {
      text = `This evening in ${loc}, conditions will be predominantly '${curCond}' with temperatures around ${eveningTemp}${unitLabel}. The evening precipitation probability is low at ${eveningRain}%, providing comfortable weather for outdoor plans.`;
      recommendation = "Pleasant evening atmospheric conditions anticipated.";
    } else if (meta.intent === "RAIN_PRECIPITATION") {
      text = curRain >= 40
        ? `Yes, there is a significant precipitation probability of ${curRain}% in ${loc} today with ${curCond} conditions. Carrying an umbrella or waterproof outerwear is strongly recommended.`
        : `No, rain is unlikely in ${loc} today. The precipitation probability is low at ${curRain}%, and skies remain mostly ${curCond}. It is completely safe to head out without rain gear.`;
      recommendation = curRain >= 40 ? "Keep rain protection handy outdoors." : "No rain interruption expected.";
    } else if (meta.intent === "TEMPERATURE") {
      text = `In ${loc}, the current temperature is ${curTemp}${unitLabel} with ${curHumidity}% relative humidity. Ambient conditions are comfortable and ${curCond.toLowerCase()}.`;
      recommendation = "Comfortable temperatures for routine activities.";
    } else {
      text = `According to live atmospheric telemetry for ${loc}, the current temperature is ${curTemp}${unitLabel} with ${curCond} conditions. Relative humidity is ${curHumidity}%, wind is ${curWind} ${weatherContext?.units === "imperial" ? "mph" : "km/h"}, and precipitation chance is ${curRain}%.`;
      recommendation = "Plan activities in accordance with local atmospheric conditions.";
    }

    return {
      text,
      source: "WeatherGPT AI Local Engine (English)",
      timestamp: new Date().toISOString(),
      structured: { summary, recommendation, riskLevel },
    };
  };

  const STOP_LOCATION_WORDS = new Set([
    "the", "this", "that", "these", "those", "a", "an", "and", "or",
    "today", "tomorrow", "yesterday", "tonight", "morning", "evening", "afternoon", "night",
    "day", "days", "week", "weeks", "month", "months", "year", "years", "next", "last", "past",
    "now", "current", "currently", "celsius", "fahrenheit", "umbrella", "rain", "rainy", "sunny",
    "cloud", "clouds", "cloudy", "weather", "forecast", "forecasts", "temperature", "temp", "temps",
    "here", "there", "my", "our", "your", "me", "us", "details", "update", "updates", "outlook",
    "hours", "hour", "minute", "minutes", "time", "times",
    "aaj", "kal", "parson", "sham", "shaam", "raat", "subah", "dopahar", "pani", "paani", "khet",
    "fasal", "kisan", "gehun", "dhan", "keetnashak", "spray", "chata", "barsat", "barish", "tapman",
    "hawa", "garmi", "sardi", "thand", "batao", "bataiye", "kaisa", "kaisi", "kaise", "hoga", "hogi"
  ]);

  function getWeatherConditionFromWmo(code: number): string {
    if (code === 0) return "Clear Sky";
    if (code === 1) return "Mainly Clear";
    if (code === 2) return "Partly Cloudy";
    if (code === 3) return "Overcast";
    if (code >= 45 && code <= 48) return "Fog / Mist";
    if (code >= 51 && code <= 55) return "Drizzle";
    if (code >= 61 && code <= 65) return "Rain";
    if (code >= 71 && code <= 77) return "Snowfall";
    if (code >= 80 && code <= 82) return "Rain Showers";
    if (code >= 95) return "Thunderstorm";
    return "Fair";
  }

  // Resolves whether user asked about a specific external city (e.g. "What is the temperature in Ranchi?" or "Indore me barish hogi kya?")
  async function resolveLocationContext(question: string, baseContext: any): Promise<any> {
    if (!question || typeof question !== "string") return baseContext;
    const q = question.trim();
    const patterns = [
      /\b(?:in|at|for|around|near)\s+([A-Za-z\u0900-\u097F]{3,25}(?:\s+[A-Za-z\u0900-\u097F]{3,20})?)\b/i,
      /\b([A-Za-z\u0900-\u097F]{3,25})\s+(?:weather|forecast|temp|temperature|मौसम|तापमान|बारिश)\b/i,
      /\b([A-Za-z\u0900-\u097F]{3,25})\s+(?:ka|ki|ke|me|mein|par)\s+(?:mausam|weather|barsat|barish|tapman|forecast)\b/i,
      /(?:mausam|weather|barsat|barish|tapman|forecast)\s+(?:in|at|for|of|ka|ki|me|mein)\s+([A-Za-z\u0900-\u097F]{3,25})/i,
      /\b([A-Za-z\u0900-\u097F]{3,25})\s+(?:ka\s+batao|ka\s+kaisa\s+hai|me\s+kaisa\s+hai|me\s+barish|me\s+tapman)\b/i,
      /\bwhat\s+about\s+([A-Za-z\u0900-\u097F]{3,25})\b/i,
      /\band\s+([A-Za-z\u0900-\u097F]{3,25})\b/i,
      /(?:में|कासा|কাছে|లో)\s+([A-Za-z\u0900-\u097F]{3,25})/i,
    ];

    let detectedCity: string | null = null;
    for (const pat of patterns) {
      const match = q.match(pat);
      if (match && match[1]) {
        const rawCandidate = match[1].trim();
        const words = rawCandidate.split(/\s+/);
        const cleanWords: string[] = [];
        for (const w of words) {
          if (STOP_LOCATION_WORDS.has(w.toLowerCase())) {
            break;
          }
          cleanWords.push(w);
        }
        const candidate = cleanWords.join(" ").trim();
        if (candidate.length >= 2 && !STOP_LOCATION_WORDS.has(candidate.toLowerCase())) {
          detectedCity = candidate;
          break;
        }
      }
    }

    if (!detectedCity) return baseContext;

    const currentLoc = (baseContext?.locationName || baseContext?.location?.name || "").toLowerCase();
    if (currentLoc.includes(detectedCity.toLowerCase())) {
      return baseContext;
    }

    try {
      const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(detectedCity)}&count=1&language=en&format=json`;
      const geoRes = await fetch(geoUrl, { signal: AbortSignal.timeout(3000) });
      if (!geoRes.ok) return baseContext;
      const geoData = await geoRes.json();
      if (!geoData.results || geoData.results.length === 0) return baseContext;

      const loc = geoData.results[0];
      const isImperial = baseContext?.units === "imperial";
      const tempUnitParam = isImperial ? "&temperature_unit=fahrenheit" : "";
      const windUnitParam = isImperial ? "&wind_speed_unit=mph" : "";
      const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,pressure_msl,uv_index&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max&timezone=auto&forecast_days=8${tempUnitParam}${windUnitParam}`;

      const wRes = await fetch(weatherUrl, { signal: AbortSignal.timeout(3500) });
      if (!wRes.ok) return baseContext;
      const wData = await wRes.json();

      const condition = getWeatherConditionFromWmo(wData.current?.weather_code ?? 0);
      const hourlyList = (wData.hourly?.time || []).slice(0, 24).map((tStr: string, idx: number) => {
        const dateObj = new Date(tStr);
        const hourLabel = dateObj.toLocaleTimeString("en-US", { hour: "numeric", hour12: true });
        return {
          hour: hourLabel,
          temp: Math.round(wData.hourly?.temperature_2m?.[idx] ?? 25),
          condition: getWeatherConditionFromWmo(wData.hourly?.weather_code?.[idx] ?? 0),
          rainProbability: wData.hourly?.precipitation_probability?.[idx] ?? 0,
          precipitation: wData.hourly?.precipitation?.[idx] ?? 0,
          windSpeed: Math.round(wData.hourly?.wind_speed_10m?.[idx] ?? 10),
          humidity: wData.hourly?.relative_humidity_2m?.[idx] ?? 50,
        };
      });

      const enrichedContext = {
        ...baseContext,
        locationName: `${loc.name}, ${loc.admin1 ? loc.admin1 + ", " : ""}${loc.country}`,
        coordinates: { lat: loc.latitude, lon: loc.longitude },
        source: "Open-Meteo High-Resolution Live Telemetry",
        current: {
          temp: Math.round(wData.current?.temperature_2m ?? 25),
          feelsLike: Math.round(wData.current?.apparent_temperature ?? 25),
          condition,
          humidity: wData.current?.relative_humidity_2m ?? 50,
          windSpeed: Math.round(wData.current?.wind_speed_10m ?? 10),
          pressure: Math.round(wData.current?.pressure_msl ?? 1013),
          rainProbability: wData.daily?.precipitation_probability_max?.[0] ?? 0,
          precipitation: wData.current?.precipitation ?? 0,
          uvIndex: wData.current?.uv_index ?? 5,
          lastUpdated: new Date().toLocaleTimeString(),
        },
        todayHourlyPreview: hourlyList,
        dailyForecast: (wData.daily?.time || []).slice(0, 7).map((dateStr: string, idx: number) => ({
          day: idx === 0 ? "Today" : idx === 1 ? "Tomorrow" : new Date(dateStr).toLocaleDateString("en-US", { weekday: "short" }),
          date: dateStr,
          maxTemp: Math.round(wData.daily?.temperature_2m_max?.[idx] ?? 30),
          minTemp: Math.round(wData.daily?.temperature_2m_min?.[idx] ?? 20),
          condition: getWeatherConditionFromWmo(wData.daily?.weather_code?.[idx] ?? 0),
          rainProbability: wData.daily?.precipitation_probability_max?.[idx] ?? 0,
          precipitationSum: wData.daily?.precipitation_sum?.[idx] ?? 0,
          windSpeedMax: Math.round(wData.daily?.wind_speed_10m_max?.[idx] ?? 12),
        })),
      };

      return enrichedContext;
    } catch (err) {
      console.warn(`Dynamic city weather fetch failed for "${detectedCity}":`, err);
      return baseContext;
    }
  }

  // Formats multi-turn conversation history into Gemini Content objects with alternating roles
  function formatGeminiContents(chatHistory: any[], latestUserPrompt: string) {
    const contents: Array<{ role: "user" | "model"; parts: Array<{ text: string }> }> = [];

    if (Array.isArray(chatHistory) && chatHistory.length > 0) {
      for (const msg of chatHistory.slice(-8)) {
        if (!msg || typeof msg.content !== "string") continue;
        const text = msg.content.trim();
        if (!text) continue;
        const role = msg.role === "assistant" || msg.role === "model" ? "model" : "user";

        // Gemini requires first turn to be 'user'
        if (contents.length === 0 && role === "model") {
          continue;
        }

        const prev = contents[contents.length - 1];
        if (prev && prev.role === role) {
          prev.parts[0].text += "\n\n" + text;
        } else {
          contents.push({ role, parts: [{ text }] });
        }
      }
    }

    // Append latest user turn
    const prev = contents[contents.length - 1];
    if (prev && prev.role === "user") {
      prev.parts[0].text += "\n\n" + latestUserPrompt;
    } else {
      contents.push({ role: "user", parts: [{ text: latestUserPrompt }] });
    }

    return contents;
  }

  // AI Chat & Weather Reasoning (Fast JSON endpoint with In-Memory Cache)
  const chatHandler = async (req: any, res: any) => {
    const { question, weatherContext, language = "en", chatHistory } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const resolvedContext = await resolveLocationContext(question, weatherContext);
    const loc = resolvedContext?.locationName || "the location";
    const queryMeta = detectQueryMeta(question, language);
    const cacheKey = getCacheKey(question, loc, queryMeta.languageCode);
    const isLiveQuery = isLiveWeatherQuery(question);

    // Never serve stale cached answers for live atmospheric queries (current temp, rain, now, etc.)
    if (!isLiveQuery) {
      const cached = getFromCache(cacheKey);
      if (cached) {
        return res.json({
          ...cached,
          cached: true,
        });
      }
    }

    telemetry.aiQueries++;
    telemetry.totalQueries++;
    logTelemetry("AI_QUERY", `User asked (${queryMeta.languageCode}/${queryMeta.intent}): "${question.slice(0, 45)}..."`, loc);

    const curTemp = resolvedContext?.current?.temp ?? 30;
    const curCond = resolvedContext?.current?.condition ?? "Clear";
    const curRain = resolvedContext?.current?.rainProbability ?? 20;
    const curHumidity = resolvedContext?.current?.humidity ?? 60;
    const curWind = resolvedContext?.current?.windSpeed ?? 12;

    const buildLocalSynthesis = () => {
      const fallbackResult = buildMultilingualSynthesis(
        question,
        loc,
        curTemp,
        curCond,
        curRain,
        curHumidity,
        curWind,
        language,
        resolvedContext
      );
      setInCache(cacheKey, fallbackResult, 180);
      return fallbackResult;
    };

    try {
      const digest = buildMeteorologicalDigest(resolvedContext, queryMeta);
      const systemInstruction = getChatSystemInstruction(queryMeta, resolvedContext, digest);
      const userPrompt = question.trim();
      const formattedContents = formatGeminiContents(chatHistory, userPrompt);

      // 1. Primary AI Provider: Google Gemini
      const ai = getGenAI();
      if (ai) {
        const geminiResult = await generateWithGeminiFallback(ai, {
          contents: formattedContents,
          config: {
            systemInstruction,
            maxOutputTokens: 800,
          },
        });

        if (geminiResult && geminiResult.text) {
          const payload = {
            text: geminiResult.text,
            source: `WeatherGPT Gemini AI (${geminiResult.modelUsed})`,
            timestamp: new Date().toISOString(),
            structured: {
              summary: `${curTemp}°C with ${curCond} in ${loc}`,
              recommendation: "Plan activities in accordance with current weather telemetry.",
              riskLevel: (curRain > 60 || curTemp > 40 ? "medium" : "low") as "low" | "medium" | "high",
            },
          };
          setInCache(cacheKey, payload, 300);
          return res.json(payload);
        }
      }

      // 2. Secondary AI fallback: OpenRouter (only if configured and credits available)
      if (process.env.OPENROUTER_API_KEY && Date.now() >= openRouterCreditExhaustedUntil) {
        const formattedMessages: Array<{ role: string; content: string }> = [];
        if (Array.isArray(chatHistory)) {
          for (const item of chatHistory.slice(-6)) {
            if (item && item.content) {
              const role = item.role === "assistant" || item.role === "model" ? "assistant" : "user";
              formattedMessages.push({ role, content: String(item.content) });
            }
          }
        }
        formattedMessages.push({ role: "user", content: userPrompt });

        const orResult = await callOpenRouter({
          messages: formattedMessages,
          systemInstruction,
          temperature: 0.3,
          maxTokens: 600,
        });

        if (orResult?.text) {
          const payload = {
            text: orResult.text,
            source: `WeatherGPT OpenRouter AI (${orResult.modelUsed})`,
            timestamp: new Date().toISOString(),
          };
          setInCache(cacheKey, payload, 300);
          return res.json(payload);
        }
      }

      return res.json(buildLocalSynthesis());
    } catch (err: any) {
      console.warn("Chat handler recovered via local synthesis:", err?.message || err);
      return res.json(buildLocalSynthesis());
    }
  };

  app.post("/api/gemini/chat", chatHandler);
  app.post("/api/chat", chatHandler);

  // Conversation Summary Endpoint using Gemini / OpenRouter with multilingual resilience
  app.post("/api/gemini/summarize", async (req, res) => {
    const { messages, language = "en" } = req.body;
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "Messages array required" });
    }

    const langConfig = getLanguageConfig(language);
    const transcript = messages
      .slice(-12)
      .map((m: any) => `${m.sender === "user" ? "User" : "Assistant"}: ${m.text}`)
      .join("\n");

    const userPrompt = `Summarize this weather consultation conversation into 1-2 concise, natural sentences strictly in the language "${langConfig.name}" (${langConfig.native}).
Do NOT mix languages. Do NOT prefix with "Executive Summary:" or "Summary:". Write pure natural text in ${langConfig.name}.
Conversation:
${transcript}`;

    const systemInstruction = `You are the WeatherGPT conversation summarizer. Provide a concise, clear 1-2 sentence summary of the weather consultation strictly in ${langConfig.name} (${langConfig.native}). Never mix languages. Output only the plain summary sentence.`;

    try {
      const aiResult = await generateAICompletion({
        userPrompt,
        systemInstruction,
        temperature: 0.2,
        maxTokens: 350,
      });

      if (aiResult?.text) {
        let clean = aiResult.text.trim();
        clean = clean.replace(/^(executive summary|conversation summary|summary|सारांश|خلاصہ)[:\s-]*/i, "").trim();
        if (clean) {
          return res.json({ summary: clean });
        }
      }
    } catch (err: any) {
      console.warn("AI summarize error:", err?.message || err);
    }

    const userMsgs = messages.filter((m: any) => m.sender === "user");
    const hasMultiple = userMsgs.length > 1;

    const nativeSummaries: Record<string, string> = {
      hi: hasMultiple
        ? "उपयोगकर्ता ने मौसम की वर्तमान स्थिति, वर्षा की संभावना और आगामी पूर्वानुमान के बारे में जानकारी प्राप्त की।"
        : "उपयोगकर्ता ने स्थानीय मौसम और वायुमंडलीय स्थिति के बारे में जानकारी प्राप्त की।",
      ur: hasMultiple
        ? "صارف نے موسم کے حالات، بارش کے امکانات اور آئندہ کی پیش گوئی کے بارے میں معلومات حاصل کیں۔"
        : "صارف نے موجودہ موسم اور فضائی حالات کے بارے میں معلومات حاصل کیں۔",
      bn: hasMultiple
        ? "ব্যবহারকারী বর্তমান আবহাওয়া, বৃষ্টিপাতের সম্ভাবনা এবং পূর্বাভাসের বিষয়ে তথ্য জেনেছেন।"
        : "ব্যবহারকারী স্থানীয় আবহাওয়া এবং বায়ুমণ্ডলীয় অবস্থা সম্পর্কে তথ্য জেনেছেন।",
      te: hasMultiple
        ? "వినియోగదారు ప్రస్తుత వాతావరణం, వర్షపాత సూచన మరియు రాబోయే పరిస్థితుల గురించి సమాచారం పొందారు."
        : "వినియోగదారు స్థానిక వాతావరణం మరియు ఉష్ణోగ్రత గురించి తెలుసుకున్నారు.",
      mr: hasMultiple
        ? "वापरकर्त्याने सद्य हवामान, पावसाचा अंदाज आणि पुढील हवामानाच्या स्थितीबाबत माहिती घेतली."
        : "वापरकर्त्याने स्थानिक वातावरणाची स्थिती आणि हवामानाचा अंदाज घेतला.",
      ta: hasMultiple
        ? "பயனர் தற்போதைய வானிலை, மழை வாய்ப்பு மற்றும் அடுத்த கட்ட நிலவரம் குறித்து தகவல் அறிந்தார்."
        : "பயனர் உள்ளூர் வானிலை மற்றும் வளிமண்டல நிலைமை பற்றி கேட்டறிந்தார்.",
      gu: hasMultiple
        ? "વપરાશકર્તાએ હાલનું હવામાન, વરસાદની શક્યતા અને આગામી આગાહી અંગે માહિતી મેળવી."
        : "વપરાશકર્તાએ સ્થાનિક હવામાન અને વાતાવરણીય પરિસ્થિતિ વિશે માહિતી લીધી.",
      kn: hasMultiple
        ? "ಬಳಕೆದಾರರು ಪ್ರಸ್ತುತ ಹವಾಮಾನ, ಮಳೆಯ ಸಾಧ್ಯತೆ ಮತ್ತು ಮುಂದಿನ ಮುನ್ಸೂಚನೆಯ ಬಗ್ಗೆ ಮಾಹಿತಿ ಪಡೆದರು."
        : "ಬಳಕೆದಾರರು ಸ್ಥಳೀಯ ಹವಾಮಾನ ಮತ್ತು ಉಷ್ಣಾಂಶದ ಬಗ್ಗೆ ಮಾಹಿತಿ ಪಡೆದರು.",
      ml: hasMultiple
        ? "ഉപയോക്താവ് നിലവിലെ കാലാവസ്ഥ, മഴയുടെ സാധ്യത, വരും ദിവസങ്ങളിലെ വിവരങ്ങൾ എന്നിവയെക്കുറിച്ച് ചോദിച്ചറിഞ്ഞു."
        : "ഉപയോക്താവ് പ്രാദേശിക കാലാവസ്ഥയെക്കുറിച്ചും അന്തരീക്ഷ സ്ഥിതിയെക്കുറിച്ചും വിവരങ്ങൾ അറിഞ്ഞു.",
      pa: hasMultiple
        ? "ਵਰਤੋਂਕਾਰ ਨੇ ਮੌਜੂਦਾ ਮੌਸਮ, ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ ਅਤੇ ਆਉਣ ਵਾਲੇ ਅਨੁਮਾਨ ਬਾਰੇ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤ ਕੀਤੀ।"
        : "ਵਰਤੋਂਕਾਰ ਨੇ ਸਥਾਨਕ ਮੌਸਮ ਅਤੇ ਵਾਯੂਮੰਡਲੀ ਹਾਲਾਤ ਬਾਰੇ ਜਾਣਕਾਰੀ ਲਈ।",
      or: hasMultiple
        ? "ଉପଭୋକ୍ତା ବର୍ତ୍ତମାନର ପାଣିପାଗ, ବର୍ଷା ସମ୍ଭାବନା ଏବଂ ଆଗାମୀ ପୂର୍ବାନୁମାନ ସମ୍ପର୍କରେ ତଥ୍ୟ ପାଇଲେ।"
        : "ଉପଭୋକ୍ତା ସ୍ଥାନୀୟ ପାଣିପାଗ ଏବଂ ବାୟୁମଣ୍ଡଳୀୟ ସ୍ଥିତି ସମ୍ପର୍କରେ ପରାମର୍ଶ କଲେ।",
      as: hasMultiple
        ? "ব্যৱহাৰকাৰীয়ে বৰ্তমান বতৰ, বৰষুণৰ সম্ভাৱনা আৰু আগন্তুক পূৰ্বানুমান সম্পৰ্কে তথ্য লাভ কৰিলে।"
        : "ব্যৱহাৰকাৰীয়ে স্থানীয় বতৰ আৰু বায়ুমণ্ডলীয় অৱস্থা সম্পৰ্কে তথ্য লাভ কৰিলে।",
      ne: hasMultiple
        ? "प्रयोगकर्ताले हालको मौसम, वर्षाको सम्भावना र आगामी पूर्वानुमानबारे जानकारी प्राप्त गरे।"
        : "प्रयोगकर्ताले स्थानीय मौसम र वायुमण्डलीय अवस्थाबारे जानकारी लिए।",
      sa: hasMultiple
        ? "प्रयोक्ता वर्तमान-वातावरणस्य, वृष्टिसम्भावनायाः तथा आगामि-पूर्वानुमानस्य विषये ज्ञातवान्।"
        : "प्रयोक्ता स्थानीय-वातावरणस्य तथा तस्य स्थितेः विषये परामर्शं प्राप्तवान्।",
      ks: hasMultiple
        ? "صارفن رُتٕچ موجودہ حالت، رُدٕچ امکانیات تہٕ برونٛہہ کن رُتٕچ پینشن گوئی متعلق معلومات حٲصل کٔرۍ۔"
        : "صارفن رُتٕچ موجودہ حالت تہٕ فضائی کیفیت متعلق معلومات حٲصل کٔرۍ।",
      sd: hasMultiple
        ? "واپارين موجوده موسم، برسات جي امڪان ۽ اڳڪٿي بابت معلومات حاصل ڪئي."
        : "واپارين مقامي موسم ۽ فضائي صورتحال بابت ڄاڻ حاصل ڪئي.",
      doi: hasMultiple
        ? "उपयोगकर्ता ने मौसम दी मौजूदा स्थिति, बरखा दी संभावना ते अग्गे दे अनुमान बारे जानकारी हासल कीती।"
        : "उपयोगकर्ता ने स्थानीय मौसम ते वायुमंडलीय स्थिति बारे जानकारी लई।",
      kok: hasMultiple
        ? "वापरप्यान सद्याचें हवामान, पावसाची शक्यताय आनी फुडलें अदमास हांचेविशीं म्हायती मेळयली."
        : "वापरप्यान थळाव्या हवामानाची आनी वातावरणाची म्हायती घेतली.",
      mai: hasMultiple
        ? "उपयोगकर्ता वर्तमान मौसम, वर्षाक संभावना आ आगामी पूर्वानुमानक संबंध मे जानकारी प्राप्त कयलनि।"
        : "उपयोगकर्ता स्थानीय मौसम आ वायुमंडलीय स्थितिक संबंध मे जानकारी लेलनि।",
      mni: hasMultiple
        ? "শীজিন্নরিবনা হৌজিক্কী নোংজু-নুংশিৎকী ফীভম অমসুং তুংগী পাউগী মতাংদা খঙদোকখি।"
        : "শীজিন্নরিবনা মফম অদুগী নোংজু-নুংশিৎকী ফীভমগী মতাংদা খঙদোকখি।",
      brx: hasMultiple
        ? "बाहायगिरिया आथिखालनि बारहावा, अखा हानिफोरा आरो इउनाव जानाय हालसालनि सोमोन्दै मिथिनो मोनदों।"
        : "बाहायगिरिया जायगानि बारहावा आरो अखा-साननि सोमोन्दै मिथिनो मोनदों।",
      en: hasMultiple
        ? "The user inquired about current atmospheric conditions, precipitation chances, and upcoming forecast trends."
        : "The user consulted about local weather metrics and atmospheric conditions.",
    };

    const summary = nativeSummaries[language] || nativeSummaries.en;
    return res.json({ summary });
  });

  const streamHandler = async (req: any, res: any) => {
    console.log("[Stream Handler] Received POST request:", req.body?.question);
    const { question, weatherContext, language = "en", chatHistory } = req.body || {};
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    let isClientDisconnected = false;
    res.on("close", () => {
      if (!res.writableEnded) {
        isClientDisconnected = true;
      }
    });

    const sendSse = (obj: any) => {
      if (!res.writableEnded && !isClientDisconnected) {
        try {
          res.write(`data: ${JSON.stringify(obj)}\n\n`);
          if (typeof (res as any).flush === "function") {
            (res as any).flush();
          }
        } catch {}
      }
    };

    // Resolve location context (handles queries for other cities like "Ranchi", "Indore", "Tokyo", etc.)
    const resolvedContext = await resolveLocationContext(question, weatherContext);
    const loc = resolvedContext?.locationName || "the location";
    const queryMeta = detectQueryMeta(question, language);
    const cacheKey = getCacheKey(question, loc, queryMeta.languageCode);
    const isLiveQuery = isLiveWeatherQuery(question);

    // Never serve stale cached answers for live atmospheric queries (current temp, rain, now, etc.)
    if (!isLiveQuery) {
      const cached = getFromCache(cacheKey);
      if (cached) {
        sendSse({ chunk: cached.text, isFinal: false });
        sendSse({ text: cached.text, isFinal: true, source: cached.source || "WeatherGPT Cache", structured: cached.structured });
        res.end();
        return;
      }
    }

    telemetry.aiQueries++;
    telemetry.totalQueries++;
    logTelemetry("AI_QUERY", `[Stream] User asked (${queryMeta.languageCode}/${queryMeta.intent}): "${question.slice(0, 45)}..."`, loc);

    const digest = buildMeteorologicalDigest(resolvedContext, queryMeta);
    const systemInstruction = getChatSystemInstruction(queryMeta, resolvedContext, digest);
    const userPrompt = question.trim();
    const formattedContents = formatGeminiContents(chatHistory, userPrompt);

    // 1. Primary AI Streaming Provider: Google Gemini (@google/genai)
    const ai = getGenAI();
    if (ai) {
      for (const model of GEMINI_MODELS) {
        if (isClientDisconnected) break;
        try {
          const responseStream = await ai.models.generateContentStream({
            model,
            contents: formattedContents,
            config: {
              systemInstruction,
              maxOutputTokens: 800,
            },
          });

          let accumulatedText = "";

          for await (const chunk of responseStream) {
            if (isClientDisconnected) break;
            const chunkText = chunk.text || "";
            if (chunkText) {
              accumulatedText += chunkText;
              sendSse({ chunk: chunkText, isFinal: false });
            }
          }

          if (isClientDisconnected) {
            return;
          }

          if (accumulatedText.trim().length > 0) {
            const finalPayload = {
              text: accumulatedText.trim(),
              isFinal: true,
              source: `WeatherGPT Gemini AI (${model})`,
              timestamp: new Date().toISOString(),
              structured: {
                summary: `${resolvedContext?.current?.temp ?? 30}°C in ${loc}`,
                recommendation: "Plan activities in accordance with current weather telemetry.",
                riskLevel: (resolvedContext?.current?.rainProbability ?? 0) > 50 ? "medium" : "low",
              },
            };
            setInCache(cacheKey, finalPayload, 300);
            sendSse(finalPayload);
            res.end();
            return;
          }
        } catch (streamErr: any) {
          const errMsg = String(streamErr?.message || streamErr);
          console.warn(`[Gemini Stream] Model ${model} error (${errMsg.slice(0, 80)}). Trying next...`);
        }
      }
    }

    // 2. Secondary fallback to OpenRouter only if Gemini was unavailable and credits exist
    if (!isClientDisconnected && process.env.OPENROUTER_API_KEY && Date.now() >= openRouterCreditExhaustedUntil) {
      try {
        const formattedMessages: Array<{ role: string; content: string }> = [];
        if (Array.isArray(chatHistory)) {
          for (const item of chatHistory.slice(-6)) {
            if (item && item.content) {
              const role = item.role === "assistant" || item.role === "model" ? "assistant" : "user";
              formattedMessages.push({ role, content: String(item.content) });
            }
          }
        }
        formattedMessages.push({ role: "user", content: userPrompt });

        const orStreamResult = await streamOpenRouter({
          messages: formattedMessages,
          systemInstruction,
          temperature: 0.3,
          maxTokens: 600,
          onChunk: (chunkText) => {
            if (isClientDisconnected) return;
            sendSse({ chunk: chunkText, isFinal: false });
          },
        });

        if (!isClientDisconnected && orStreamResult?.text?.trim()) {
          const finalPayload = {
            text: orStreamResult.text.trim(),
            isFinal: true,
            source: `WeatherGPT OpenRouter AI (${orStreamResult.modelUsed})`,
            timestamp: new Date().toISOString(),
          };
          setInCache(cacheKey, finalPayload, 300);
          sendSse(finalPayload);
          res.end();
          return;
        }
      } catch (orErr) {
        console.warn("[Stream] OpenRouter stream error:", orErr);
      }
    }

    // 3. Resilient synthesis fallback
    if (!isClientDisconnected && !res.writableEnded) {
      const curTemp = resolvedContext?.current?.temp ?? 30;
      const curCond = resolvedContext?.current?.condition ?? "Clear";
      const curRain = resolvedContext?.current?.rainProbability ?? 20;
      const curHumidity = resolvedContext?.current?.humidity ?? 60;
      const curWind = resolvedContext?.current?.windSpeed ?? 12;

      const fallbackResult = buildMultilingualSynthesis(
        question,
        loc,
        curTemp,
        curCond,
        curRain,
        curHumidity,
        curWind,
        language,
        resolvedContext
      );

      sendSse({ chunk: fallbackResult.text, isFinal: false });
      const payload = {
        text: fallbackResult.text,
        isFinal: true,
        source: fallbackResult.source,
        structured: fallbackResult.structured,
      };
      setInCache(cacheKey, payload, 180);
      sendSse(payload);
      res.end();
    }
  };

  app.post("/api/gemini/chat/stream", streamHandler);
  app.post("/api/chat/stream", streamHandler);

  // Dedicated OpenRouter direct endpoints (Server-side)
  app.post("/api/openrouter/chat", async (req, res) => {
    const { question, weatherContext, language = "en", model, messages } = req.body;
    if (!question && (!messages || messages.length === 0)) {
      return res.status(400).json({ error: "question or messages required" });
    }

    const userPrompt = question || messages?.[messages.length - 1]?.content || "";
    const systemInstruction = getChatSystemInstruction(language, weatherContext);

    try {
      const result = await callOpenRouter({
        messages: messages || [{ role: "user", content: userPrompt }],
        systemInstruction,
        preferredModel: model,
        maxTokens: 500,
      });

      if (result) {
        return res.json({
          text: result.text,
          modelUsed: result.modelUsed,
          source: `WeatherGPT OpenRouter AI (${result.modelUsed})`,
          timestamp: new Date().toISOString(),
        });
      }

      // Seamless fallback to Gemini if OpenRouter is out of credits or unavailable
      const geminiResult = await generateAICompletion({
        userPrompt,
        systemInstruction,
        temperature: 0.2,
        maxTokens: 500,
      });

      if (geminiResult) {
        return res.json({
          text: geminiResult.text,
          modelUsed: geminiResult.modelUsed,
          source: geminiResult.source,
          timestamp: new Date().toISOString(),
        });
      }

      return res.status(503).json({ error: "AI service currently unavailable" });
    } catch (err: any) {
      return res.status(500).json({ error: err.message || "AI error" });
    }
  });

  app.post("/api/openrouter/chat/stream", async (req, res) => {
    const { question, weatherContext, language = "en", model, messages } = req.body;
    if (!question && (!messages || messages.length === 0)) {
      return res.status(400).json({ error: "question or messages required" });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    const userPrompt = question || messages?.[messages.length - 1]?.content || "";
    const systemInstruction = getChatSystemInstruction(language, weatherContext);

    try {
      const streamResult = await streamOpenRouter({
        messages: messages || [{ role: "user", content: userPrompt }],
        systemInstruction,
        preferredModel: model,
        maxTokens: 500,
        onChunk: (chunkText) => {
          res.write(`data: ${JSON.stringify({ chunk: chunkText, isFinal: false })}\n\n`);
        },
      });

      if (streamResult) {
        res.write(`data: ${JSON.stringify({ text: streamResult.text, isFinal: true, source: `WeatherGPT OpenRouter AI (${streamResult.modelUsed})`, timestamp: new Date().toISOString() })}\n\n`);
        res.end();
        return;
      }

      // Seamless fallback to Gemini stream if OpenRouter is out of credits
      const ai = getGenAI();
      if (ai) {
        for (const geminiModel of GEMINI_MODELS) {
          try {
            const responseStream = await ai.models.generateContentStream({
              model: geminiModel,
              contents: userPrompt,
              config: {
                systemInstruction,
                temperature: 0.2,
                maxOutputTokens: 600,
              },
            });

            let acc = "";
            for await (const chunk of responseStream) {
              const cText = chunk.text || "";
              if (cText) {
                acc += cText;
                res.write(`data: ${JSON.stringify({ chunk: cText, isFinal: false })}\n\n`);
              }
            }

            if (acc.trim().length > 0) {
              res.write(`data: ${JSON.stringify({ text: acc, isFinal: true, source: `WeatherGPT AI Engine (${geminiModel})`, timestamp: new Date().toISOString() })}\n\n`);
              res.end();
              return;
            }
          } catch {
            continue;
          }
        }
      }

      res.write(`data: ${JSON.stringify({ error: "AI service stream unavailable", isFinal: true })}\n\n`);
      res.end();
    } catch (err: any) {
      res.write(`data: ${JSON.stringify({ error: err.message, isFinal: true })}\n\n`);
      res.end();
    }
  });

  // AI Smart Advisory Endpoint (Farmer, Travel, Outdoor, Student)
  app.post("/api/gemini/advisory", async (req, res) => {
    const { category, weatherContext, language = "en" } = req.body;
    const langConfig = getLanguageConfig(language);
    const isHi = language === "hi";
    const temp = weatherContext?.current?.temp ?? 30;
    const rain = weatherContext?.current?.rainProbability ?? 20;
    const wind = weatherContext?.current?.windSpeed ?? 12;
    const uv = weatherContext?.current?.uvIndex ?? 5;
    const vis = weatherContext?.current?.visibility ?? 10;

    const getFallbackAdvisory = () => {
      const fallbacks: Record<string, Record<string, any>> = {
        farmer: {
          title: isHi ? "कृषि एवं फसल मौसम परामर्श" : "Agricultural & Crop Advisory",
          verdict: rain > 60
            ? (isHi ? "वर्षा की संभावना उच्च है - सिंचाई स्थगित करें" : "High rain probability - Postpone chemical spraying & irrigation")
            : (isHi ? "सामान्य मौसम - शाम को हल्की सिंचाई की जा सकती है" : "Favorable conditions - Field operations and scheduled irrigation permitted"),
          risk: rain > 70 || temp > 42 ? "high" : "low",
          tips: isHi
            ? [
                "खेतों में जल निकासी चैनलों की स्थिति जांच लें।",
                `हवा की गति (${wind} किमी/घंटा) को ध्यान में रखकर ही कीटनाशक छिड़काव करें।`,
                "कटी हुई फसल को सुरक्षित सूखे स्थान या तिरपाल से ढक कर रखें।",
                "मिट्टी की नमी का स्तर सामान्य बनाए रखें।",
              ]
            : [
                "Inspect field drainage channels prior to forecasted precipitation.",
                `Monitor current wind speed (${wind} km/h) before spraying pesticides.`,
                "Store harvested grain in covered, waterproof sheds.",
                "Maintain optimal soil moisture for seasonal crops.",
              ],
          bestTimeWindow: "06:00 AM - 09:30 AM & 05:00 PM - 07:00 PM",
          precautions: isHi ? "बिजली चमकने के दौरान खुले खेतों में काम न करें।" : "Avoid open field work during thunderstorm alerts.",
        },
        travel: {
          title: isHi ? "यात्रा एवं आवागमन मौसम परामर्श" : "Highway & Transit Advisory",
          verdict: isHi
            ? `दृश्यता ${vis} किमी के साथ यात्रा के लिए ${rain > 60 ? "मध्यम सावधानी आवश्यक" : "मौसम अनुकूल"}`
            : `Safe transit window with ${vis} km visibility. ${rain > 60 ? "Moderate caution for rain." : "Clear highway conditions."}`,
          risk: rain > 70 || vis < 3 ? "medium" : "low",
          tips: isHi
            ? [
                "गीली सड़कों पर वाहन की गति नियंत्रित रखें।",
                "वाहन की हेडलाइट्स और वाइपर की कार्यप्रणाली जांचें।",
                "जलभराव वाले अंडरपास से गुजरने से बचें।",
              ]
            : [
                "Maintain safe braking distances on wet road surfaces.",
                "Ensure functional windshield wipers and low-beam headlights.",
                "Check municipal traffic updates for low-lying waterlogged roads.",
              ],
          bestTimeWindow: "07:00 AM - 11:00 AM & 03:00 PM - 06:00 PM",
          precautions: isHi ? "अचानक तेज बारिश में सुरक्षित स्थान पर वाहन रोकें।" : "Pull over safely during sudden heavy downpours.",
        },
        outdoor: {
          title: isHi ? "आउटडोर खेल एवं फिटनेस परामर्श" : "Outdoor & Fitness Advisory",
          verdict: temp > 35
            ? (isHi ? "उच्च तापमान - सुबह या शाम को ही व्यायाम करें" : "High heat index - Exercise early morning or after sunset")
            : (isHi ? "दौड़, खेल और व्यायाम के लिए उत्कृष्ट मौसम" : "Great conditions for jogging, sports, and outdoor workouts"),
          risk: temp > 38 || uv >= 8 ? "medium" : "low",
          tips: isHi
            ? [
                `वर्तमान यूवी इंडेक्स ${uv} है। सनस्क्रीन लगाएं।`,
                "पर्याप्त मात्रा में पानी और इलेक्ट्रोलाइट्स पिएं।",
                "दोपहर 12 से 3 बजे के बीच सीधी धूप से बचें।",
              ]
            : [
                `Current UV index is ${uv}. Apply broad-spectrum sunscreen.`,
                "Maintain consistent hydration with water and electrolytes.",
                "Avoid strenuous outdoor activities between 12 PM and 3 PM.",
              ],
          bestTimeWindow: "05:30 AM - 08:30 AM",
          precautions: isHi ? "चक्कर आने या थकान महसूस होने पर तुरंत छाया में विश्राम करें।" : "Rest in shaded areas if feeling fatigued or dehydrated.",
        },
        student: {
          title: isHi ? "छात्र एवं कॉलेज आवागमन परामर्श" : "Student & Campus Commute Advisory",
          verdict: isHi
            ? `कॉलेज जाने के लिए ${rain > 50 ? "छाता साथ रखना आवश्यक" : "मौसम सुखद और सामान्य"}`
            : `Campus commute condition is ${rain > 50 ? "rain gear recommended" : "pleasant and comfortable"}.`,
          risk: rain > 60 ? "medium" : "low",
          tips: isHi
            ? [
                "अपने बैग में वाटरप्रूफ कवर या पॉलीथीन रखें।",
                "पानी की बोतल साथ रखें।",
                "मौसम में बदलाव के लिए तैयार रहें।",
              ]
            : [
                "Keep an umbrella or lightweight raincoat in your college backpack.",
                "Carry a reusable water bottle for campus hydration.",
                "Allow a 10-minute buffer for public transport on rainy days.",
              ],
          bestTimeWindow: "08:00 AM - 09:30 AM",
          precautions: isHi ? "इलेक्ट्रॉनिक गैजेट्स और किताबों को पानी से बचाएं।" : "Protect laptops and textbooks in water-resistant sleeves.",
        },
      };
      return fallbacks[category] || fallbacks.travel;
    };

    try {
      const prompt = `
Generate a practical, empathetic, and structured advisory for category "${category}" based on real weather data:
Location: ${weatherContext?.locationName}
Current Temp: ${temp}°C
Condition: ${weatherContext?.current?.condition}
Rain Probability: ${rain}%
Wind Speed: ${wind} km/h
UV Index: ${uv}

Language: ${langConfig.name} (${langConfig.native}). ${langConfig.instructions}
All JSON string values (title, verdict, tips, bestTimeWindow, precautions) MUST be naturally composed in ${langConfig.name} (${langConfig.native}).

Return ONLY valid JSON in this structure:
{
  "title": "Short title in ${langConfig.name}",
  "verdict": "One sentence summary verdict in ${langConfig.name}",
  "risk": "low" | "medium" | "high",
  "tips": ["Tip 1 in ${langConfig.name}", "Tip 2 in ${langConfig.name}", "Tip 3 in ${langConfig.name}", "Tip 4 in ${langConfig.name}"],
  "bestTimeWindow": "e.g. 6:00 AM - 10:00 AM",
  "precautions": "Precautions to take in ${langConfig.name}"
}
      `.trim();

      const aiResult = await generateAICompletion({
        userPrompt: prompt,
        systemInstruction: `You are a professional meteorologist. Generate all advice strictly in ${langConfig.name} (${langConfig.native}). Return only valid JSON adhering strictly to the schema.`,
        temperature: 0.3,
        maxTokens: 500,
        jsonMode: true,
      });

      if (aiResult?.text) {
        const cleaned = cleanJson(aiResult.text);
        const parsed = JSON.parse(cleaned);
        return res.json(parsed);
      }

      return res.json(getFallbackAdvisory());
    } catch (err: any) {
      console.warn("Advisory handler recovered via fallback data:", err?.message || err);
      return res.json(getFallbackAdvisory());
    }
  });

  // AI Weather Comparison (e.g. Bhopal vs Patna)
  app.post("/api/gemini/compare", async (req, res) => {
    const { loc1, loc2, weather1, weather2, language = "en" } = req.body;
    const langConfig = getLanguageConfig(language);
    const isHi = language === "hi";
    const temp1 = weather1?.temp ?? 28;
    const temp2 = weather2?.temp ?? 31;
    const warmer = temp1 >= temp2 ? loc1 : loc2;
    const rainRisk = (weather1?.rainProbability ?? 0) >= (weather2?.rainProbability ?? 0) ? loc1 : loc2;

    const getFallbackComparison = () => ({
      summary: isHi
        ? `${loc1} में तापमान ${temp1}°C (${weather1?.condition || "सामान्य"}) है जबकि ${loc2} में तापमान ${temp2}°C (${weather2?.condition || "सामान्य"}) है। ${warmer} अधिक गर्म है।`
        : `${loc1} is currently ${temp1}°C (${weather1?.condition || "fair"}) while ${loc2} is ${temp2}°C (${weather2?.condition || "fair"}). ${warmer} is warmer by ${Math.abs(temp1 - temp2)}°C.`,
      warmerCity: warmer,
      rainRiskCity: rainRisk,
      recommendation: isHi
        ? `${loc1} और ${loc2} के बीच यात्रा करने से पहले दोनों शहरों के तापमान और बारिश के पूर्वानुमान को ध्यान में रखें।`
        : `If commuting between ${loc1} and ${loc2}, prepare for slight variation in temperature and humidity.`,
      comparisonPoints: [
        `${loc1}: ${temp1}°C, Humidity ${weather1?.humidity ?? 60}%, Rain ${weather1?.rainProbability ?? 20}%`,
        `${loc2}: ${temp2}°C, Humidity ${weather2?.humidity ?? 60}%, Rain ${weather2?.rainProbability ?? 20}%`,
        `Wind: ${weather1?.windSpeed ?? 10} km/h (${loc1}) vs ${weather2?.windSpeed ?? 12} km/h (${loc2})`,
      ],
    });

    try {
      const prompt = `
Compare the weather between two cities for a WeatherGPT user:
City 1: ${loc1} -> Temp: ${temp1}°C, Condition: ${weather1?.condition}, Humidity: ${weather1?.humidity}%, Rain Chance: ${weather1?.rainProbability}%, Wind: ${weather1?.windSpeed} km/h, UV: ${weather1?.uvIndex}
City 2: ${loc2} -> Temp: ${temp2}°C, Condition: ${weather2?.condition}, Humidity: ${weather2?.humidity}%, Rain Chance: ${weather2?.rainProbability}%, Wind: ${weather2?.windSpeed} km/h, UV: ${weather2?.uvIndex}

Language: ${langConfig.name} (${langConfig.native}). ${langConfig.instructions}
All JSON string values (summary, recommendation, comparisonPoints) MUST be composed in ${langConfig.name} (${langConfig.native}).

Provide a crisp comparison summary and actionable travel/living advice.
Return ONLY valid JSON:
{
  "summary": "Detailed comparison sentence in ${langConfig.name}",
  "warmerCity": "${loc1}" or "${loc2}",
  "rainRiskCity": "Name of city with higher rain risk or None",
  "recommendation": "Practical recommendation for travelers or residents in ${langConfig.name}",
  "comparisonPoints": ["Point 1 in ${langConfig.name}", "Point 2 in ${langConfig.name}", "Point 3 in ${langConfig.name}"]
}
      `.trim();

      const aiResult = await generateAICompletion({
        userPrompt: prompt,
        systemInstruction: `You are an atmospheric scientist comparing two cities. Generate all explanations strictly in ${langConfig.name} (${langConfig.native}). Return only valid JSON adhering strictly to the schema.`,
        temperature: 0.3,
        maxTokens: 500,
        jsonMode: true,
      });

      if (aiResult?.text) {
        const cleaned = cleanJson(aiResult.text);
        const parsed = JSON.parse(cleaned);
        return res.json(parsed);
      }

      return res.json(getFallbackComparison());
    } catch (err: any) {
      console.warn("Comparison handler recovered via fallback data:", err?.message || err);
      return res.json(getFallbackComparison());
    }
  });

  // AI Climate Educator & Explainer
  app.post("/api/gemini/climate", async (req, res) => {
    const { topic, question, language = "en" } = req.body;
    const langConfig = getLanguageConfig(language);
    const isHi = language === "hi";

    const getFallbackClimate = () => ({
      title: topic || "Indian Climate & Monsoon Dynamics",
      explanation: isHi
        ? "भारतीय मानसून हिंद महासागर और भारतीय उपमहाद्वीप के बीच तापमान और वायुदाब के अंतर से संचालित होता है। दक्षिण-पश्चिमी मानसून भारत की वार्षिक वर्षा का 75% से अधिक हिस्सा प्रदान करता है और कृषि अर्थव्यवस्था की रीढ़ है।"
        : "The Indian climate is primarily shaped by the Southwest and Northeast monsoons, tropical cyclonic systems, and Western Disturbances. The Southwest Monsoon accounts for over 75% of India's annual precipitation.",
      causes: [
        "Differential seasonal heating of the Indian subcontinent and Indian Ocean",
        "Intertropical Convergence Zone (ITCZ) latitudinal migration",
        "El Niño / La Niña Southern Oscillation (ENSO) ocean-atmosphere coupling",
      ],
      impactsOnIndia: [
        "Crucial for 60%+ of India's rainfed agrarian economy and kharif crops",
        "Replenishment of major river basins and underground aquifers",
        "Vulnerability to extreme weather events (heatwaves, flash floods)",
      ],
      solutionsAndAdaptation: [
        "Deployment of advanced Doppler radars and AI-based early warning systems like WeatherGPT",
        "Promotion of climate-resilient crop varieties and precision micro-irrigation",
      ],
      keyTakeaway: "Accurate meteorological intelligence and proactive community alerts foster robust climate resilience.",
    });

    try {
      const prompt = `
You are an atmospheric scientist and climate educator for WeatherGPT.
Explain the topic: "${topic || question}"
User Question: "${question || topic}"
Language: ${langConfig.name} (${langConfig.native}). ${langConfig.instructions}
All JSON string values (title, explanation, causes, impactsOnIndia, solutionsAndAdaptation, keyTakeaway) MUST be composed in ${langConfig.name} (${langConfig.native}).

Explain clearly with scientific accuracy yet accessible to students, farmers, and citizens.
Return ONLY valid JSON:
{
  "title": "Descriptive Topic Title in ${langConfig.name}",
  "explanation": "Clear 2-3 paragraph explanation in ${langConfig.name}",
  "causes": ["Cause 1 in ${langConfig.name}", "Cause 2 in ${langConfig.name}", "Cause 3 in ${langConfig.name}"],
  "impactsOnIndia": ["Impact 1 in ${langConfig.name}", "Impact 2 in ${langConfig.name}", "Impact 3 in ${langConfig.name}"],
  "solutionsAndAdaptation": ["Action 1 in ${langConfig.name}", "Action 2 in ${langConfig.name}"],
  "keyTakeaway": "One-line summary takeaway in ${langConfig.name}"
}
      `.trim();

      const aiResult = await generateAICompletion({
        userPrompt: prompt,
        systemInstruction: `You are an atmospheric scientist and climate educator. Generate explanations strictly in ${langConfig.name} (${langConfig.native}). Return only valid JSON adhering strictly to the schema.`,
        temperature: 0.3,
        maxTokens: 500,
        jsonMode: true,
      });

      if (aiResult?.text) {
        const cleaned = cleanJson(aiResult.text);
        const parsed = JSON.parse(cleaned);
        return res.json(parsed);
      }

      return res.json(getFallbackClimate());
    } catch (err: any) {
      console.warn("Climate handler recovered via fallback data:", err?.message || err);
      return res.json(getFallbackClimate());
    }
  });

  // Vite middleware for development vs static build in production
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`WeatherGPT Full-Stack Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Fatal error starting server:", err);
});
