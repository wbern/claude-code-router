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
 * Parse the Retry-After header value into milliseconds.
 * Supports both seconds (integer) and HTTP-date formats.
 */
function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = parseInt(header, 10);
  if (!isNaN(seconds)) return Math.max(INITIAL_BACKOFF_MS, seconds * 1000);
  const date = Date.parse(header);
  if (!isNaN(date)) return Math.max(0, date - Date.now());
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

  // Retry loop for transient failures (429, 5xx)
  let lastResponse: Response | undefined;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      const retryAfter = parseRetryAfter(
        lastResponse?.headers?.get("retry-after") ?? null
      );
      const backoff =
        retryAfter ?? INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
      logger?.warn?.(
        `[Request] Retrying after ${backoff}ms (attempt ${attempt + 1}/${MAX_RETRIES + 1}, status=${lastResponse?.status})`
      );
      await new Promise((resolve) => setTimeout(resolve, backoff));
    }

    lastResponse = await fetch(requestUrl, fetchOptions);

    // Don't retry streaming responses (can't re-read the body) or successful ones
    if (lastResponse.ok || request.stream) {
      return lastResponse;
    }

    if (!isRetryableStatus(lastResponse.status)) {
      return lastResponse;
    }

    // Drain response body to release the TCP connection before retrying
    try { await lastResponse.text(); } catch { /* ignore */ }

    // On last attempt, return whatever we got
    if (attempt === MAX_RETRIES) {
      return lastResponse;
    }
  }

  return lastResponse!;
}
