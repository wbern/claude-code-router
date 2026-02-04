import { LLMProvider, UnifiedChatRequest } from "../types/llm";
import { Transformer } from "../types/transformer";
import {
  buildRequestBody,
  transformRequestOut,
  transformResponseOut,
} from "../utils/gemini.util";
import { execSync } from "child_process";

// Cache for Keychain lookup (avoid repeated subprocess calls)
let cachedKeychainKey: string | null = null;

/**
 * Attempt to read API key from macOS Keychain.
 * Returns null on non-macOS or if not found.
 */
function getApiKeyFromKeychain(): string | null {
  if (process.platform !== "darwin") return null;
  if (cachedKeychainKey) return cachedKeychainKey;

  try {
    const result = execSync(
      'security find-generic-password -s "claude-code-router" -a "gemini-api-key" -w 2>/dev/null',
      { encoding: "utf8", timeout: 5000 }
    ).trim();

    if (result) {
      cachedKeychainKey = result;
      return result;
    }
  } catch {
    // Keychain entry not found - fall through
  }
  return null;
}

/**
 * Resolve API key with fallback chain: env var > Keychain > config.
 * Config may contain placeholder like "FROM_KEYCHAIN" which should not be used.
 */
function resolveApiKey(providerApiKey: string): string {
  const envKey = process.env.GEMINI_API_KEY?.trim();
  if (envKey) return envKey;

  const keychainKey = getApiKeyFromKeychain();
  if (keychainKey) return keychainKey;

  // Fallback to config, but skip placeholders
  const isPlaceholder =
    !providerApiKey ||
    providerApiKey === "FROM_KEYCHAIN" ||
    providerApiKey.startsWith("YOUR_");
  if (!isPlaceholder) return providerApiKey;

  throw new Error(
    "No Gemini API key found. Set GEMINI_API_KEY env var or store in macOS Keychain with: " +
      'security add-generic-password -s "claude-code-router" -a "gemini-api-key" -w "YOUR_KEY"'
  );
}

export class GeminiTransformer implements Transformer {
  name = "gemini";

  endPoint = "/v1beta/models/:modelAndAction";

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider
  ): Promise<Record<string, any>> {
    const apiKey = resolveApiKey(provider.apiKey);

    return {
      body: buildRequestBody(request),
      config: {
        url: new URL(
          `./${request.model}:${
            request.stream ? "streamGenerateContent?alt=sse" : "generateContent"
          }`,
          provider.baseUrl
        ),
        headers: {
          "x-goog-api-key": apiKey,
          Authorization: undefined,
        },
      },
    };
  }

  transformRequestOut = transformRequestOut;

  async transformResponseOut(response: Response): Promise<Response> {
    return transformResponseOut(response, this.name, this.logger);
  }
}
