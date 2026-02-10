import { UnifiedChatRequest, UnifiedMessage } from "../types/llm";
import { Content, ContentListUnion, Part, ToolListUnion } from "@google/genai";

export function cleanupParameters(obj: any, keyName?: string): void {
  if (!obj || typeof obj !== "object") {
    return;
  }

  if (Array.isArray(obj)) {
    obj.forEach((item) => {
      cleanupParameters(item);
    });
    return;
  }

  const validFields = new Set([
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "propertyOrdering",
    "default",
    "items",
    "minimum",
    "maximum",
  ]);

  if (keyName !== "properties") {
    Object.keys(obj).forEach((key) => {
      if (!validFields.has(key)) {
        delete obj[key];
      }
    });
  }

  if (obj.enum && obj.type !== "string") {
    delete obj.enum;
  }

  if (
    obj.type === "string" &&
    obj.format &&
    !["enum", "date-time"].includes(obj.format)
  ) {
    delete obj.format;
  }

  Object.keys(obj).forEach((key) => {
    cleanupParameters(obj[key], key);
  });
}

// Type enum equivalent in JavaScript
const Type = {
  TYPE_UNSPECIFIED: "TYPE_UNSPECIFIED",
  STRING: "STRING",
  NUMBER: "NUMBER",
  INTEGER: "INTEGER",
  BOOLEAN: "BOOLEAN",
  ARRAY: "ARRAY",
  OBJECT: "OBJECT",
  NULL: "NULL",
};

/**
 * Transform the type field from an array of types to an array of anyOf fields.
 * @param {string[]} typeList - List of types
 * @param {Object} resultingSchema - The schema object to modify
 */
function flattenTypeArrayToAnyOf(
  typeList: Array<string>,
  resultingSchema: any
): void {
  if (typeList.includes("null")) {
    resultingSchema["nullable"] = true;
  }
  const listWithoutNull = typeList.filter((type) => type !== "null");

  if (listWithoutNull.length === 1) {
    const upperCaseType = listWithoutNull[0].toUpperCase();
    resultingSchema["type"] = Object.values(Type).includes(upperCaseType)
      ? upperCaseType
      : Type.TYPE_UNSPECIFIED;
  } else {
    resultingSchema["anyOf"] = [];
    for (const i of listWithoutNull) {
      const upperCaseType = i.toUpperCase();
      resultingSchema["anyOf"].push({
        type: Object.values(Type).includes(upperCaseType)
          ? upperCaseType
          : Type.TYPE_UNSPECIFIED,
      });
    }
  }
}

/**
 * Process a JSON schema to make it compatible with the GenAI API
 * @param {Object} _jsonSchema - The JSON schema to process
 * @returns {Object} - The processed schema
 */
function processJsonSchema(_jsonSchema: any): any {
  const genAISchema = {};
  const schemaFieldNames = ["items"];
  const listSchemaFieldNames = ["anyOf"];
  const dictSchemaFieldNames = ["properties"];

  if (_jsonSchema["type"] && _jsonSchema["anyOf"]) {
    throw new Error("type and anyOf cannot be both populated.");
  }

  /*
  This is to handle the nullable array or object. The _jsonSchema will
  be in the format of {anyOf: [{type: 'null'}, {type: 'object'}]}. The
  logic is to check if anyOf has 2 elements and one of the element is null,
  if so, the anyOf field is unnecessary, so we need to get rid of the anyOf
  field and make the schema nullable. Then use the other element as the new
  _jsonSchema for processing. This is because the backend doesn't have a null
  type.
  */
  const incomingAnyOf = _jsonSchema["anyOf"];
  if (
    incomingAnyOf != null &&
    Array.isArray(incomingAnyOf) &&
    incomingAnyOf.length == 2
  ) {
    if (incomingAnyOf[0] && incomingAnyOf[0]["type"] === "null") {
      genAISchema["nullable"] = true;
      _jsonSchema = incomingAnyOf[1];
    } else if (incomingAnyOf[1] && incomingAnyOf[1]["type"] === "null") {
      genAISchema["nullable"] = true;
      _jsonSchema = incomingAnyOf[0];
    }
  }

  if (_jsonSchema["type"] && Array.isArray(_jsonSchema["type"])) {
    flattenTypeArrayToAnyOf(_jsonSchema["type"], genAISchema);
  }

  for (const [fieldName, fieldValue] of Object.entries(_jsonSchema)) {
    // Skip if the fieldValue is undefined or null.
    if (fieldValue == null) {
      continue;
    }

    if (fieldName == "type") {
      if (fieldValue === "null") {
        throw new Error(
          "type: null can not be the only possible type for the field."
        );
      }
      if (Array.isArray(fieldValue)) {
        // we have already handled the type field with array of types in the
        // beginning of this function.
        continue;
      }
      const upperCaseValue = fieldValue.toUpperCase();
      genAISchema["type"] = Object.values(Type).includes(upperCaseValue)
        ? upperCaseValue
        : Type.TYPE_UNSPECIFIED;
    } else if (schemaFieldNames.includes(fieldName)) {
      genAISchema[fieldName] = processJsonSchema(fieldValue);
    } else if (listSchemaFieldNames.includes(fieldName)) {
      const listSchemaFieldValue = [];
      for (const item of fieldValue) {
        if (item["type"] == "null") {
          genAISchema["nullable"] = true;
          continue;
        }
        listSchemaFieldValue.push(processJsonSchema(item));
      }
      genAISchema[fieldName] = listSchemaFieldValue;
    } else if (dictSchemaFieldNames.includes(fieldName)) {
      const dictSchemaFieldValue = {};
      for (const [key, value] of Object.entries(fieldValue)) {
        dictSchemaFieldValue[key] = processJsonSchema(value);
      }
      genAISchema[fieldName] = dictSchemaFieldValue;
    } else {
      // additionalProperties is not included in JSONSchema, skipping it.
      if (fieldName === "additionalProperties") {
        continue;
      }
      genAISchema[fieldName] = fieldValue;
    }
  }
  return genAISchema;
}

/**
 * Transform a tool object
 * @param {Object} tool - The tool object to transform
 * @returns {Object} - The transformed tool object
 */
export function tTool(tool: any): any {
  if (tool.functionDeclarations) {
    for (const functionDeclaration of tool.functionDeclarations) {
      if (functionDeclaration.parameters) {
        if (!Object.keys(functionDeclaration.parameters).includes("$schema")) {
          functionDeclaration.parameters = processJsonSchema(
            functionDeclaration.parameters
          );
        } else {
          if (!functionDeclaration.parametersJsonSchema) {
            functionDeclaration.parametersJsonSchema =
              functionDeclaration.parameters;
            delete functionDeclaration.parameters;
          }
        }
      }
      if (functionDeclaration.response) {
        if (!Object.keys(functionDeclaration.response).includes("$schema")) {
          functionDeclaration.response = processJsonSchema(
            functionDeclaration.response
          );
        } else {
          if (!functionDeclaration.responseJsonSchema) {
            functionDeclaration.responseJsonSchema =
              functionDeclaration.response;
            delete functionDeclaration.response;
          }
        }
      }
    }
  }
  return tool;
}

/**
 * Error patterns that indicate a model is stuck in an Edit tool loop.
 * These are error messages returned by Claude Code's Edit tool.
 */
const EDIT_LOOP_ERROR_PATTERNS = [
  "old_string and new_string are exactly the same",
  "No changes to make",
];

/**
 * Substrings that indicate a tool result is an error rather than success.
 */
const ERROR_INDICATORS = [
  "Error:",
  "Error ",
  "error:",
  "ENOENT",
  "EACCES",
  "EPERM",
  "failed",
  "FAILED",
  "not found",
  "Permission denied",
  "Operation not permitted",
];

/**
 * Extract text content from a message's content field.
 */
function getToolResultText(
  content: string | null | Array<any>
): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((c: any) => c.text || "").join(" ");
  }
  return "";
}

/**
 * Detect if the model is stuck in a tool usage loop.
 * Scans recent messages for repeated tool failures and returns an
 * appropriate hint:
 *   - Edit-specific hint when the Edit tool is called with identical strings
 *   - Generic hint when any tool keeps erroring, nudging the model to try
 *     a different non-destructive approach or inform the user it's stuck
 *
 * Returns a hint message if a loop is detected, null otherwise.
 */
function detectToolLoops(messages: UnifiedMessage[]): string | null {
  const recentMessages = messages.slice(-20);

  let editSameContentCount = 0;
  let genericErrorCount = 0;

  for (const msg of recentMessages) {
    if (msg.role === "tool") {
      const text = getToolResultText(msg.content);

      if (EDIT_LOOP_ERROR_PATTERNS.some((p) => text.includes(p))) {
        editSameContentCount++;
      } else if (ERROR_INDICATORS.some((p) => text.includes(p))) {
        genericErrorCount++;
      }
    }
  }

  // Specific hint for Edit tool same-content loop (lower threshold — very specific pattern)
  if (editSameContentCount >= 2) {
    return (
      "IMPORTANT: Your last Edit/Update attempts failed because old_string and new_string were identical. " +
      "The Edit tool requires old_string to exactly match existing file text, and new_string must be DIFFERENT from old_string. " +
      "If you cannot express the change correctly with Edit, use the Write tool to replace the entire file contents instead."
    );
  }

  // Generic hint for any repeated tool errors (higher threshold to avoid false positives)
  if (genericErrorCount >= 3) {
    return (
      "IMPORTANT: You appear to be encountering repeated tool errors. " +
      "Stop retrying the same failing approach. Instead, try a DIFFERENT, non-destructive method to accomplish your goal. " +
      "If you cannot find an approach that works, tell the user what you were trying to do and that you are unable to proceed — " +
      "do not keep retrying the same operation."
    );
  }

  return null;
}

export function buildRequestBody(
  request: UnifiedChatRequest
): Record<string, any> {
  const tools = [];
  const functionDeclarations = request.tools
    ?.filter((tool) => tool.function.name !== "web_search")
    ?.map((tool) => {
      return {
        name: tool.function.name,
        description: tool.function.description,
        parametersJsonSchema: tool.function.parameters,
      };
    });
  if (functionDeclarations?.length) {
    tools.push(
      tTool({
        functionDeclarations,
      })
    );
  }
  const webSearch = request.tools?.find(
    (tool) => tool.function.name === "web_search"
  );
  if (webSearch) {
    tools.push({
      googleSearch: {},
    });
  }

  const contents: any[] = [];
  const toolResponses = request.messages.filter((item) => item.role === "tool");
  request.messages
    .filter((item) => item.role !== "tool")
    .forEach((message: UnifiedMessage) => {
      let role: "user" | "model";
      if (message.role === "assistant") {
        role = "model";
      } else if (["user", "system"].includes(message.role)) {
        role = "user";
      } else {
        role = "user"; // Default to user if role is not recognized
      }
      const parts = [];
      if (typeof message.content === "string") {
        const part: any = {
          text: message.content,
        };
        if (message?.thinking?.signature) {
          part.thoughtSignature = message.thinking.signature;
        }
        parts.push(part);
      } else if (Array.isArray(message.content)) {
        parts.push(
          ...message.content.map((content) => {
            if (content.type === "text") {
              return {
                text: content.text || "",
              };
            }
            if (content.type === "image_url") {
              if (content.image_url.url.startsWith("http")) {
                return {
                  file_data: {
                    mime_type: content.media_type,
                    file_uri: content.image_url.url,
                  },
                };
              } else {
                return {
                  inlineData: {
                    mime_type: content.media_type,
                    data:
                      content.image_url.url?.split(",")?.pop() ||
                      content.image_url.url,
                  },
                };
              }
            }
          })
        );
      } else if (message.content && typeof message.content === "object") {
        // Object like { text: "..." }
        if (message.content.text) {
          parts.push({ text: message.content.text });
        } else {
          parts.push({ text: JSON.stringify(message.content) });
        }
      }

      if (Array.isArray(message.tool_calls)) {
        parts.push(
          ...message.tool_calls.map((toolCall, index) => {
            return {
              functionCall: {
                id:
                  toolCall.id ||
                  `tool_${Math.random().toString(36).substring(2, 15)}`,
                name: toolCall.function.name,
                args: JSON.parse(toolCall.function.arguments || "{}"),
              },
              thoughtSignature:
                index === 0 && message.thinking?.signature
                  ? message.thinking?.signature
                  : undefined,
            };
          })
        );
      }

      if (parts.length === 0) {
        parts.push({ text: "" });
      }

      contents.push({
        role,
        parts,
      });

      if (role === "model" && message.tool_calls) {
        const functionResponses = message.tool_calls.map((tool) => {
          const response = toolResponses.find(
            (item) => item.tool_call_id === tool.id
          );
          return {
            functionResponse: {
              name: tool?.function?.name,
              response: { result: response?.content },
            },
          };
        });
        contents.push({
          role: "user",
          parts: functionResponses,
        });
      }
    });

  // Detect tool usage loops and inject a hint to help the model recover
  const loopHint = detectToolLoops(request.messages);
  if (loopHint) {
    const lastContent = contents[contents.length - 1];
    if (lastContent && lastContent.role === "user") {
      // Append to existing user message to maintain role alternation
      lastContent.parts.push({ text: loopHint });
    } else {
      contents.push({
        role: "user",
        parts: [{ text: loopHint }],
      });
    }
  }

  const generationConfig: any = {};

  // Gemini 3 models require temperature=1.0 to prevent infinite thinking loops.
  // Lower temperatures cause the model to get trapped in deterministic verification loops.
  // See: https://ai.google.dev/gemini-api/docs/gemini-3
  if (request.model.includes("gemini-3")) {
    generationConfig.temperature = 1.0;
  }

  if (
    request.reasoning &&
    request.reasoning.effort &&
    request.reasoning.effort !== "none"
  ) {
    generationConfig.thinkingConfig = {
      includeThoughts: true,
    };
    if (request.model.includes("gemini-3")) {
      // Gemini 3 Pro only supports LOW and HIGH.
      // Gemini 3 Flash supports LOW, MEDIUM, HIGH.
      // Map Claude Code's effort values to valid levels per model.
      const effort = request.reasoning.effort;
      if (request.model.includes("pro")) {
        generationConfig.thinkingConfig.thinkingLevel =
          effort === "high" ? "HIGH" : "LOW";
      } else {
        generationConfig.thinkingConfig.thinkingLevel =
          effort === "high" ? "HIGH" : effort === "medium" ? "MEDIUM" : "LOW";
      }
    } else {
      const thinkingBudgets = request.model.includes("pro")
        ? [128, 32768]
        : [0, 24576];
      let thinkingBudget;
      const max_tokens = request.reasoning.max_tokens;
      if (typeof max_tokens !== "undefined") {
        if (
          max_tokens >= thinkingBudgets[0] &&
          max_tokens <= thinkingBudgets[1]
        ) {
          thinkingBudget = max_tokens;
        } else if (max_tokens < thinkingBudgets[0]) {
          thinkingBudget = thinkingBudgets[0];
        } else if (max_tokens > thinkingBudgets[1]) {
          thinkingBudget = thinkingBudgets[1];
        }
        generationConfig.thinkingConfig.thinkingBudget = thinkingBudget;
      }
    }
  }

  const body = {
    contents,
    tools: tools.length ? tools : undefined,
    generationConfig,
    systemInstruction: {
      role: "user",
      parts: [
        {
          text: [
            "<role>",
            "You are a coding assistant operating inside Claude Code, a CLI tool for software development.",
            "</role>",
            "",
            "<tool-guidance>",
            "The Edit tool performs exact string replacement in files:",
            "- old_string must EXACTLY match text currently in the file, including whitespace and indentation",
            "- new_string must be DIFFERENT from old_string — identical strings will always fail",
            "- Read a file before editing it to ensure you have the current contents",
            "- If Edit fails, use the Write tool to replace the entire file instead",
            "</tool-guidance>",
            "",
            "<constraints>",
            "If a tool operation fails twice with the same error, switch to a different non-destructive approach.",
            "If no approach works, clearly tell the user what you attempted and that you cannot proceed — do not keep retrying the same failing operation.",
            "</constraints>",
          ].join("\n"),
        },
      ],
    },
  };

  if (request.tool_choice) {
    const toolConfig = {
      functionCallingConfig: {},
    };
    if (request.tool_choice === "auto") {
      toolConfig.functionCallingConfig.mode = "auto";
    } else if (request.tool_choice === "none") {
      toolConfig.functionCallingConfig.mode = "none";
    } else if (request.tool_choice === "required") {
      toolConfig.functionCallingConfig.mode = "any";
    } else if (request.tool_choice?.function?.name) {
      toolConfig.functionCallingConfig.mode = "any";
      toolConfig.functionCallingConfig.allowedFunctionNames = [
        request.tool_choice?.function?.name,
      ];
    }
    body.toolConfig = toolConfig;
  }

  return body;
}

export function transformRequestOut(
  request: Record<string, any>
): UnifiedChatRequest {
  const contents: ContentListUnion = request.contents;
  const tools: ToolListUnion = request.tools;
  const model: string = request.model;
  const max_tokens: number | undefined = request.max_tokens;
  const temperature: number | undefined = request.temperature;
  const stream: boolean | undefined = request.stream;
  const tool_choice: "auto" | "none" | string | undefined = request.tool_choice;

  const unifiedChatRequest: UnifiedChatRequest = {
    messages: [],
    model,
    max_tokens,
    temperature,
    stream,
    tool_choice,
  };

  if (Array.isArray(contents)) {
    contents.forEach((content) => {
      if (typeof content === "string") {
        unifiedChatRequest.messages.push({
          role: "user",
          content,
        });
      } else if (typeof (content as Part).text === "string") {
        unifiedChatRequest.messages.push({
          role: "user",
          content: (content as Part).text || null,
        });
      } else if ((content as Content).role === "user") {
        unifiedChatRequest.messages.push({
          role: "user",
          content:
            (content as Content)?.parts?.map((part: Part) => ({
              type: "text",
              text: part.text || "",
            })) || [],
        });
      } else if ((content as Content).role === "model") {
        unifiedChatRequest.messages.push({
          role: "assistant",
          content:
            (content as Content)?.parts?.map((part: Part) => ({
              type: "text",
              text: part.text || "",
            })) || [],
        });
      }
    });
  }

  if (Array.isArray(tools)) {
    unifiedChatRequest.tools = [];
    tools.forEach((tool) => {
      if (Array.isArray(tool.functionDeclarations)) {
        tool.functionDeclarations.forEach((tool) => {
          unifiedChatRequest.tools!.push({
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.parameters,
            },
          });
        });
      }
    });
  }

  return unifiedChatRequest;
}

/**
 * Get the correct finish_reason, converting "stop" to "tool_calls" when tool calls are present.
 * Gemini sends "stop" even when returning tool calls, but OpenAI/Anthropic format expects "tool_calls".
 */
function getFinishReason(candidate: any, hasToolCalls: boolean): string | null {
  const reason = candidate?.finishReason?.toLowerCase() || null;
  if (hasToolCalls && reason === "stop") {
    return "tool_calls";
  }
  return reason;
}

/**
 * Detect if a request is a "suggestion mode" request from Claude Code.
 * These are short requests that predict what the user might type next.
 * They complete quickly and can cause race conditions with longer subagent requests.
 */
function isSuggestionModeRequest(context?: any): boolean {
  try {
    const body = context?.req?.body;
    if (!body?.messages) return false;

    // Check if any message contains the SUGGESTION MODE marker
    return body.messages.some((msg: any) => {
      if (typeof msg.content === 'string') {
        return msg.content.includes('[SUGGESTION MODE:');
      }
      if (Array.isArray(msg.content)) {
        return msg.content.some((c: any) =>
          typeof c.text === 'string' && c.text.includes('[SUGGESTION MODE:')
        );
      }
      return false;
    });
  } catch {
    return false;
  }
}

/**
 * Delay helper for suggestion mode responses.
 * This gives subagents time to complete before the suggestion response
 * potentially triggers the UI to think the turn is complete.
 */
const SUGGESTION_MODE_DELAY_MS = 3000;

export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any,
  context?: any
): Promise<Response> {
  const isSuggestion = isSuggestionModeRequest(context);
  if (isSuggestion) {
    logger?.debug?.('[Gemini] Detected suggestion mode request, will delay response');
  }
  if (response.headers.get("Content-Type")?.includes("application/json")) {
    const jsonResponse: any = await response.json();
    logger?.debug({ response: jsonResponse }, `${providerName} response:`);

    // Extract thinking content from parts with thought: true
    let thinkingContent = "";
    let thinkingSignature = "";

    const parts = jsonResponse.candidates[0]?.content?.parts || [];
    const nonThinkingParts: Part[] = [];

    for (const part of parts) {
      if (part.text && part.thought === true) {
        thinkingContent += part.text;
      } else {
        nonThinkingParts.push(part);
      }
    }

    // Get thoughtSignature from functionCall args or usageMetadata
    thinkingSignature = parts.find(
      (part: any) => part.thoughtSignature
    )?.thoughtSignature;

    const tool_calls =
      nonThinkingParts
        ?.filter((part: Part) => part.functionCall)
        ?.map((part: Part) => ({
          id:
            part.functionCall?.id ||
            `tool_${Math.random().toString(36).substring(2, 15)}`,
          type: "function",
          function: {
            name: part.functionCall?.name,
            arguments: JSON.stringify(part.functionCall?.args || {}),
          },
        })) || [];

    const textContent =
      nonThinkingParts
        ?.filter((part: Part) => part.text)
        ?.map((part: Part) => part.text)
        ?.join("\n") || "";

    const res = {
      id: jsonResponse.responseId,
      choices: [
        {
          finish_reason: getFinishReason(jsonResponse.candidates[0], tool_calls.length > 0),
          index: 0,
          message: {
            content: textContent,
            role: "assistant",
            tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
            // Add thinking as separate field if available (only if there's actual content)
            ...(thinkingSignature && thinkingContent && {
              thinking: {
                content: thinkingContent,
                signature: thinkingSignature,
              },
            }),
          },
        },
      ],
      created: parseInt(new Date().getTime() / 1000 + "", 10),
      model: jsonResponse.modelVersion,
      object: "chat.completion",
      usage: {
        completion_tokens:
          jsonResponse.usageMetadata?.candidatesTokenCount || 0,
        prompt_tokens: jsonResponse.usageMetadata?.promptTokenCount || 0,
        prompt_tokens_details: {
          cached_tokens:
            jsonResponse.usageMetadata?.cachedContentTokenCount || 0,
        },
        total_tokens: jsonResponse.usageMetadata?.totalTokenCount || 0,
        output_tokens_details: {
          reasoning_tokens: jsonResponse.usageMetadata?.thoughtsTokenCount || 0,
        },
      },
    };
    // Delay suggestion mode responses to prevent race conditions with subagents
    if (isSuggestion) {
      logger?.debug?.(`[Gemini] Delaying suggestion mode JSON response by ${SUGGESTION_MODE_DELAY_MS}ms`);
      await new Promise(resolve => setTimeout(resolve, SUGGESTION_MODE_DELAY_MS));
    }

    return new Response(JSON.stringify(res), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } else if (response.headers.get("Content-Type")?.includes("stream")) {
    if (!response.body) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let signatureSent = false;
    let contentSent = false;
    let hasThinkingContent = false;
    let pendingContent = "";
    let contentIndex = 0;
    let toolCallIndex = -1;

    const stream = new ReadableStream({
      async start(controller) {
        const processLine = async (
          line: string,
          controller: ReadableStreamDefaultController
        ) => {
          if (line.startsWith("data: ")) {
            const chunkStr = line.slice(6).trim();
            if (chunkStr) {
              logger?.debug({ chunkStr }, `${providerName} chunk:`);
              try {
                const chunk = JSON.parse(chunkStr);

                // Check if chunk has valid structure
                if (!chunk.candidates || !chunk.candidates[0]) {
                  logger?.debug({ chunkStr }, `Invalid chunk structure`);
                  return;
                }

                const candidate = chunk.candidates[0];
                const parts = candidate.content?.parts || [];

                parts
                  .filter((part: any) => part.text && part.thought === true)
                  .forEach((part: any) => {
                    if (!hasThinkingContent) {
                      hasThinkingContent = true;
                    }
                    const thinkingChunk = {
                      choices: [
                        {
                          delta: {
                            role: "assistant",
                            content: null,
                            thinking: {
                              content: part.text,
                            },
                          },
                          finish_reason: null,
                          index: contentIndex,
                          logprobs: null,
                        },
                      ],
                      created: parseInt(new Date().getTime() / 1000 + "", 10),
                      id: chunk.responseId || "",
                      model: chunk.modelVersion || "",
                      object: "chat.completion.chunk",
                      system_fingerprint: "fp_a49d71b8a1",
                    };
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify(thinkingChunk)}\n\n`
                      )
                    );
                  });

                let signature = parts.find(
                  (part: Part) => part.thoughtSignature
                )?.thoughtSignature;
                if (signature && !signatureSent) {
                  // Only send thinking chunk if we have actual thinking content
                  // (skip empty thinking to avoid UI showing "Thinking... (no content)")
                  const signatureChunk = {
                    choices: [
                      {
                        delta: {
                          role: "assistant",
                          content: null,
                          thinking: {
                            signature,
                          },
                        },
                        finish_reason: null,
                        index: contentIndex,
                        logprobs: null,
                      },
                    ],
                    created: parseInt(new Date().getTime() / 1000 + "", 10),
                    id: chunk.responseId || "",
                    model: chunk.modelVersion || "",
                    object: "chat.completion.chunk",
                    system_fingerprint: "fp_a49d71b8a1",
                  };
                  controller.enqueue(
                    encoder.encode(
                      `data: ${JSON.stringify(signatureChunk)}\n\n`
                    )
                  );
                  signatureSent = true;
                  contentIndex++;
                  if (pendingContent) {
                    const res = {
                      choices: [
                        {
                          delta: {
                            role: "assistant",
                            content: pendingContent,
                          },
                          finish_reason: null,
                          index: contentIndex,
                          logprobs: null,
                        },
                      ],
                      created: parseInt(new Date().getTime() / 1000 + "", 10),
                      id: chunk.responseId || "",
                      model: chunk.modelVersion || "",
                      object: "chat.completion.chunk",
                      system_fingerprint: "fp_a49d71b8a1",
                    };

                    controller.enqueue(
                      encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                    );

                    pendingContent = "";
                    if (!contentSent) {
                      contentSent = true;
                    }
                  }
                }

                const tool_calls = parts
                  .filter((part: Part) => part.functionCall)
                  .map((part: Part) => ({
                    id:
                      part.functionCall?.id ||
                      `ccr_tool_${Math.random().toString(36).substring(2, 15)}`,
                    type: "function",
                    function: {
                      name: part.functionCall?.name,
                      arguments: JSON.stringify(part.functionCall?.args || {}),
                    },
                  }));

                const textContent = parts
                  .filter((part: Part) => part.text && part.thought !== true)
                  .map((part: Part) => part.text)
                  .join("\n");

                // Skip sending empty content placeholder - let actual content through
                // (removed "(no content)" fallback that was confusing UI)

                if (hasThinkingContent && textContent && !signatureSent) {
                  if (chunk.modelVersion.includes("3")) {
                    pendingContent += textContent;
                    return;
                  } else {
                    const signatureChunk = {
                      choices: [
                        {
                          delta: {
                            role: "assistant",
                            content: null,
                            thinking: {
                              signature: `ccr_${+new Date()}`,
                            },
                          },
                          finish_reason: null,
                          index: contentIndex,
                          logprobs: null,
                        },
                      ],
                      created: parseInt(new Date().getTime() / 1000 + "", 10),
                      id: chunk.responseId || "",
                      model: chunk.modelVersion || "",
                      object: "chat.completion.chunk",
                      system_fingerprint: "fp_a49d71b8a1",
                    };
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify(signatureChunk)}\n\n`
                      )
                    );
                    signatureSent = true;
                  }
                }

                if (textContent) {
                  if (!pendingContent) contentIndex++;
                  const res = {
                    choices: [
                      {
                        delta: {
                          role: "assistant",
                          content: textContent,
                        },
                        finish_reason: getFinishReason(candidate, tool_calls.length > 0),
                        index: contentIndex,
                        logprobs: null,
                      },
                    ],
                    created: parseInt(new Date().getTime() / 1000 + "", 10),
                    id: chunk.responseId || "",
                    model: chunk.modelVersion || "",
                    object: "chat.completion.chunk",
                    system_fingerprint: "fp_a49d71b8a1",
                    usage: {
                      completion_tokens:
                        chunk.usageMetadata?.candidatesTokenCount || 0,
                      prompt_tokens: chunk.usageMetadata?.promptTokenCount || 0,
                      prompt_tokens_details: {
                        cached_tokens:
                          chunk.usageMetadata?.cachedContentTokenCount || 0,
                      },
                      total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                      output_tokens_details: {
                        reasoning_tokens:
                          chunk.usageMetadata?.thoughtsTokenCount || 0,
                      },
                    },
                  };

                  if (candidate?.groundingMetadata?.groundingChunks?.length) {
                    (res.choices[0].delta as any).annotations =
                      candidate.groundingMetadata.groundingChunks.map(
                        (groundingChunk: any, index: number) => {
                          const support =
                            candidate?.groundingMetadata?.groundingSupports?.filter(
                              (item: any) =>
                                item.groundingChunkIndices?.includes(index)
                            );
                          return {
                            type: "url_citation",
                            url_citation: {
                              url: groundingChunk?.web?.uri || "",
                              title: groundingChunk?.web?.title || "",
                              content: support?.[0]?.segment?.text || "",
                              start_index:
                                support?.[0]?.segment?.startIndex || 0,
                              end_index: support?.[0]?.segment?.endIndex || 0,
                            },
                          };
                        }
                      );
                  }
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                  );

                  if (!contentSent && textContent) {
                    contentSent = true;
                  }
                }

                if (tool_calls.length > 0) {
                  tool_calls.forEach((tool) => {
                    contentIndex++;
                    toolCallIndex++;
                    const res = {
                      choices: [
                        {
                          delta: {
                            role: "assistant",
                            tool_calls: [
                              {
                                ...tool,
                                index: toolCallIndex,
                              },
                            ],
                          },
                          finish_reason: getFinishReason(candidate, true),
                          index: contentIndex,
                          logprobs: null,
                        },
                      ],
                      created: parseInt(new Date().getTime() / 1000 + "", 10),
                      id: chunk.responseId || "",
                      model: chunk.modelVersion || "",
                      object: "chat.completion.chunk",
                      system_fingerprint: "fp_a49d71b8a1",
                      usage: {
                        completion_tokens:
                          chunk.usageMetadata?.candidatesTokenCount || 0,
                        prompt_tokens:
                          chunk.usageMetadata?.promptTokenCount || 0,
                        prompt_tokens_details: {
                          cached_tokens:
                            chunk.usageMetadata?.cachedContentTokenCount || 0,
                        },
                        total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                        output_tokens_details: {
                          reasoning_tokens:
                            chunk.usageMetadata?.thoughtsTokenCount || 0,
                        },
                      },
                    };

                    if (candidate?.groundingMetadata?.groundingChunks?.length) {
                      (res.choices[0].delta as any).annotations =
                        candidate.groundingMetadata.groundingChunks.map(
                          (groundingChunk: any, index: number) => {
                            const support =
                              candidate?.groundingMetadata?.groundingSupports?.filter(
                                (item: any) =>
                                  item.groundingChunkIndices?.includes(index)
                              );
                            return {
                              type: "url_citation",
                              url_citation: {
                                url: groundingChunk?.web?.uri || "",
                                title: groundingChunk?.web?.title || "",
                                content: support?.[0]?.segment?.text || "",
                                start_index:
                                  support?.[0]?.segment?.startIndex || 0,
                                end_index: support?.[0]?.segment?.endIndex || 0,
                              },
                            };
                          }
                        );
                    }
                    controller.enqueue(
                      encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                    );
                  });

                  if (!contentSent && textContent) {
                    contentSent = true;
                  }
                }
              } catch (error: any) {
                logger?.error(
                  `Error parsing ${providerName} stream chunk`,
                  chunkStr,
                  error.message
                );
              }
            }
          }
        };

        const reader = response.body!.getReader();
        let buffer = "";
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              if (buffer) {
                await processLine(buffer, controller);
              }
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");

            buffer = lines.pop() || "";

            for (const line of lines) {
              await processLine(line, controller);
            }
          }
        } catch (error: any) {
          // Gracefully handle stream interruptions (e.g., ERR_STREAM_PREMATURE_CLOSE)
          // instead of crashing the entire response. Emit [DONE] so the client
          // receives whatever content was already streamed.
          if (
            error?.code === "ERR_STREAM_PREMATURE_CLOSE" ||
            error?.message?.includes("premature close") ||
            error?.message?.includes("aborted")
          ) {
            logger?.warn?.(
              `[Gemini] Stream interrupted (${error.code || error.message}), closing gracefully`
            );
            try {
              controller.enqueue(
                encoder.encode("data: [DONE]\n\n")
              );
            } catch {
              // Controller may already be in error state
            }
          } else {
            controller.error(error);
          }
        } finally {
          // Delay suggestion mode responses to prevent race conditions with subagents
          if (isSuggestion) {
            logger?.debug?.(`[Gemini] Delaying suggestion mode streaming response by ${SUGGESTION_MODE_DELAY_MS}ms`);
            await new Promise(resolve => setTimeout(resolve, SUGGESTION_MODE_DELAY_MS));
          }
          controller.close();
        }
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }
}
