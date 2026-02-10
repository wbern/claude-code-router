import { ProxyAgent } from "undici";
import { UnifiedChatRequest } from "../types/llm";

const MAX_RETRIES = 2;
const INITIAL_BACKOFF_MS = 1000;

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

/**
 * Try to extract a retry delay from the response body.
 * Gemini returns retryDelay in error.details[] (e.g., {"retryDelay": "54s"})
 * rather than in the standard Retry-After HTTP header.
 */
function parseBodyRetryDelay(bodyText: string): number | null {
  try {
    const json = JSON.parse(bodyText);
    const details = json?.error?.details;
    if (!Array.isArray(details)) return null;
    for (const detail of details) {
      const delay = detail?.retryDelay;
      if (typeof delay === "string") {
        const seconds = parseFloat(delay);
        if (!isNaN(seconds) && seconds > 0) {
          return Math.max(INITIAL_BACKOFF_MS, seconds * 1000);
        }
      }
    }
  } catch {
    /* not parseable JSON */
  }
  return null;
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

  const abortController = config.signal ? new AbortController() : null;
  const timeoutSignal = AbortSignal.timeout(config.TIMEOUT ?? 60 * 1000 * 60);
  let combinedSignal: AbortSignal;

  if (config.signal && abortController) {
    const abortHandler = () => abortController.abort();
    config.signal.addEventListener("abort", abortHandler, { once: true });
    timeoutSignal.addEventListener("abort", abortHandler, { once: true });
    combinedSignal = abortController.signal;
  } else {
    combinedSignal = timeoutSignal;
  }

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: headers,
    body: JSON.stringify(request),
    signal: combinedSignal,
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
      const backoff =
        headerRetryMs ??
        bodyRetryDelayMs ??
        INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
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

    // Wrap fetch to catch network-level errors (HeadersTimeout, connection reset, etc.)
    try {
      lastResponse = await fetch(requestUrl, fetchOptions);
    } catch (error: any) {
      if (isRetryableNetworkError(error) && attempt < MAX_RETRIES) {
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

    // Drain response body to release TCP connection, and extract any retry
    // delay hint from the body (Gemini sends retryDelay in error.details[]
    // rather than in the standard Retry-After HTTP header)
    try {
      const bodyText = await lastResponse.text();
      bodyRetryDelayMs = parseBodyRetryDelay(bodyText);
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
