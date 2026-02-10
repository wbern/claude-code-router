import { ProxyAgent } from "undici";
import { UnifiedChatRequest } from "../types/llm";

const MAX_RETRIES = 2;
const INITIAL_BACKOFF_MS = 1000;

/**
 * How long to wait for response headers before aborting. Gemini sometimes
 * accepts the TCP connection but never responds, hanging for 5 minutes
 * (undici's default headersTimeout). 90 seconds is what coffeegrind123's
 * proxy uses; Gemini CLI plans 60s. We use 90s as a conservative default.
 */
const CONNECT_TIMEOUT_MS = 90_000;

/**
 * Check if an HTTP status code is retryable.
 * 429 = rate limited, 500/502/503/504 = transient server errors.
 */
function isRetryableStatus(status: number): boolean {
  return status === 429 || (status >= 500 && status <= 504);
}

/**
 * Check if a fetch error is a transient network failure worth retrying.
 * These are infrastructure-level failures where the server never responded.
 */
function isRetryableNetworkError(error: any): boolean {
  const code = error?.code;
  return (
    code === "UND_ERR_HEADERS_TIMEOUT" ||
    code === "UND_ERR_CONNECT_TIMEOUT" ||
    code === "UND_ERR_SOCKET" ||
    code === "ECONNRESET" ||
    code === "ECONNREFUSED" ||
    code === "ETIMEDOUT"
  );
}

/**
 * Parse the Retry-After header value into milliseconds.
 * Supports both seconds (integer) and HTTP-date formats.
 */
function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = parseInt(header, 10);
  if (!isNaN(seconds)) return Math.max(INITIAL_BACKOFF_MS, seconds * 1000);
  const date = Date.parse(header);
  if (!isNaN(date)) return Math.max(INITIAL_BACKOFF_MS, date - Date.now());
  return null;
}

interface BodyRetryInfo {
  delayMs: number | null;
  isDailyQuota: boolean;
}

/**
 * Parse retry info from the response body. Gemini returns retryDelay in
 * error.details[] (e.g., {"retryDelay": "54s"}) rather than in the standard
 * Retry-After HTTP header.
 *
 * Also checks quotaId to distinguish per-minute rate limits (retryable) from
 * per-day quota exhaustion (not retryable — retryDelay is meaningless for
 * daily limits, confirmed by Google engineer in developer forums).
 */
function parseBodyRetryInfo(bodyText: string): BodyRetryInfo {
  try {
    const json = JSON.parse(bodyText);
    const details = json?.error?.details;
    if (!Array.isArray(details)) return { delayMs: null, isDailyQuota: false };

    let delayMs: number | null = null;
    let isDailyQuota = false;

    for (const detail of details) {
      // RetryInfo detail: { "@type": "...RetryInfo", "retryDelay": "54s" }
      const delay = detail?.retryDelay;
      if (typeof delay === "string") {
        const seconds = parseFloat(delay);
        if (!isNaN(seconds) && seconds > 0) {
          delayMs = Math.max(INITIAL_BACKOFF_MS, seconds * 1000);
        }
      }
      // QuotaFailure detail has metadata.quotaId indicating limit type
      const quotaId = detail?.metadata?.quotaId;
      if (typeof quotaId === "string" && quotaId.includes("PerDay")) {
        isDailyQuota = true;
      }
    }

    return { delayMs, isDailyQuota };
  } catch {
    return { delayMs: null, isDailyQuota: false };
  }
}

/**
 * Add 10-30% random jitter to a backoff delay to avoid thundering herd.
 * Recommended by Google's official retry documentation.
 */
function withJitter(delayMs: number): number {
  const jitter = delayMs * (0.1 + Math.random() * 0.2);
  return Math.round(delayMs + jitter);
}

export async function sendUnifiedRequest(
  url: URL | string,
  request: UnifiedChatRequest,
  config: any,
  context: any,
  logger?: any
): Promise<Response> {
  const headers = new Headers({
    "Content-Type": "application/json",
  });
  if (config.headers) {
    Object.entries(config.headers).forEach(([key, value]) => {
      if (value) {
        headers.set(key, value as string);
      }
    });
  }

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: headers,
    body: JSON.stringify(request),
    // signal is set per-attempt below
  };

  if (config.httpsProxy) {
    (fetchOptions as any).dispatcher = new ProxyAgent(
      new URL(config.httpsProxy).toString()
    );
  }

  const requestUrl = typeof url === "string" ? url : url.toString();

  logger?.debug(
    {
      reqId: context.req.id,
      request: fetchOptions,
      headers: Object.fromEntries(headers.entries()),
      requestUrl,
      useProxy: config.httpsProxy,
    },
    "final request"
  );

  // Retry loop for transient failures (network errors, 429, 5xx)
  let lastResponse: Response | undefined;
  let lastError: any = undefined;
  let bodyRetryDelayMs: number | null = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      const headerRetryMs = lastResponse
        ? parseRetryAfter(lastResponse.headers?.get("retry-after") ?? null)
        : null;
      const baseBackoff =
        headerRetryMs ??
        bodyRetryDelayMs ??
        INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
      const backoff = withJitter(baseBackoff);
      const reason = lastError
        ? `network error: ${lastError.code || lastError.message}`
        : `status=${lastResponse?.status}`;
      logger?.warn?.(
        `[Request] Retrying after ${backoff}ms (attempt ${attempt + 1}/${MAX_RETRIES + 1}, ${reason})`
      );
      await new Promise((resolve) => setTimeout(resolve, backoff));
      bodyRetryDelayMs = null;
      lastError = undefined;
    }

    // Per-attempt abort controller with a 90s headers timeout.
    // This replaces the old 60-minute AbortSignal.timeout which let requests
    // hang for 5 minutes waiting on undici's headersTimeout.
    const attemptAC = new AbortController();

    // Forward caller's abort signal (e.g., user pressing Ctrl+C)
    let callerAbortHandler: (() => void) | null = null;
    if (config.signal) {
      if (config.signal.aborted) {
        throw config.signal.reason ?? new DOMException("Aborted", "AbortError");
      }
      callerAbortHandler = () => attemptAC.abort(config.signal.reason);
      config.signal.addEventListener("abort", callerAbortHandler, { once: true });
    }

    // Short timeout to receive response headers. Once headers arrive and
    // fetch() resolves, this timer is cleared — body streaming is not affected.
    const connectTimer = setTimeout(
      () => attemptAC.abort(new Error("CONNECT_TIMEOUT")),
      CONNECT_TIMEOUT_MS
    );

    const cleanup = () => {
      clearTimeout(connectTimer);
      if (callerAbortHandler && config.signal) {
        config.signal.removeEventListener("abort", callerAbortHandler);
      }
    };

    try {
      lastResponse = await fetch(requestUrl, {
        ...fetchOptions,
        signal: attemptAC.signal,
      });
      cleanup();
    } catch (error: any) {
      cleanup();

      // Caller-initiated abort (Ctrl+C) — propagate immediately, don't retry
      if (config.signal?.aborted) {
        throw error;
      }

      // Our connect timeout or transient network error — retry if attempts remain
      const isOurTimeout = error.name === "AbortError";
      if ((isRetryableNetworkError(error) || isOurTimeout) && attempt < MAX_RETRIES) {
        lastError = error;
        lastResponse = undefined;
        continue;
      }
      throw error;
    }

    // Don't retry streaming responses (can't re-read the body) or successful ones
    if (lastResponse.ok || request.stream) {
      return lastResponse;
    }

    if (!isRetryableStatus(lastResponse.status)) {
      return lastResponse;
    }

    // On last attempt, return with body intact for the caller to read
    if (attempt === MAX_RETRIES) {
      return lastResponse;
    }

    // Drain response body to release TCP connection, and extract retry info.
    // For 429s, Gemini embeds retryDelay and quotaId in the error body.
    try {
      const bodyText = await lastResponse.text();
      const retryInfo = parseBodyRetryInfo(bodyText);

      if (retryInfo.isDailyQuota) {
        // Daily quota exhausted — retrying is pointless, it won't reset
        // until midnight Pacific. Return a synthetic response so the caller
        // sees the 429 and can handle it (fallback or error to user).
        logger?.error?.(
          `[Request] Daily quota exhausted (429 with PerDay quotaId), not retrying`
        );
        return new Response(bodyText, {
          status: lastResponse.status,
          statusText: lastResponse.statusText,
          headers: lastResponse.headers,
        });
      }

      bodyRetryDelayMs = retryInfo.delayMs;
    } catch {
      /* ignore drain errors */
    }
  }

  // If all retries were network errors, throw the last one
  if (lastError && !lastResponse) {
    throw lastError;
  }

  return lastResponse!;
}
