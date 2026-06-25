// Background service worker for handling messages between popup, content script, and backend.

// Backend API base URL (user-selected integration port).
const API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_BIAS_TYPES = ["gender"];

// Keep latest automatic capture for popup redisplay after open/refresh.
let lastLLMTurn = null;

function getStoredBiasTypes() {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve(DEFAULT_BIAS_TYPES);
      return;
    }

    chrome.storage.local.get({ bias_types: DEFAULT_BIAS_TYPES }, (result) => {
      const stored = Array.isArray(result?.bias_types) && result.bias_types.length > 0
        ? result.bias_types
        : DEFAULT_BIAS_TYPES;
      resolve(stored);
    });
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "GET_LAST_LLM_TURN") {
    sendResponse({ ok: true, data: lastLLMTurn });
    return true;
  }

  if (message.type === "ANALYZE_TEXT" || message.type === "NEW_LLM_TURN") {
    void (async () => {
      const {
        text = "",
        question = "",
        prompt = "",
        context = "",
        answer = "",
        mode_speed = "fast",
        mode_depth = "normal",
        bias_types = [],
      } = message.payload || {};

      const normalizedBiasTypes = Array.isArray(bias_types) && bias_types.length > 0
        ? bias_types
        : await getStoredBiasTypes();

      try {
        const res = await fetch(`${API_BASE_URL}/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text,
            question,
            prompt,
            context,
            answer,
            mode_speed,
            mode_depth,
            bias_types: normalizedBiasTypes,
          }),
        });

        if (!res.ok) {
          const bodyText = await res.text();
          throw new Error(bodyText || `HTTP ${res.status}`);
        }

        const data = await res.json();

        // If this came from the popup, respond back to it.
        if (message.type === "ANALYZE_TEXT") {
          sendResponse({ ok: true, data });
        }

        // If this was an automatic LLM turn, push result back into the page
        // and also broadcast it so the popup can show the Q + A and scores.
        if (message.type === "NEW_LLM_TURN") {
          lastLLMTurn = {
            question,
            prompt,
            context,
            answer,
            bias_types: normalizedBiasTypes,
            result: data,
            timestamp: Date.now(),
          };

          const resultPayload = {
            type: "BIAS_RESULT_FOR_LLM_TURN",
            payload: lastLLMTurn,
          };

          if (sender.tab && sender.tab.id) {
            chrome.tabs.sendMessage(sender.tab.id, resultPayload, (resp) => {
              if (chrome.runtime.lastError) {
                console.warn("No content script listener or closed tab:", chrome.runtime.lastError.message);
              }
            });
          }

          chrome.runtime.sendMessage(resultPayload, (resp) => {
            if (chrome.runtime.lastError) {
              console.warn("No runtime listener for BIAS_RESULT_FOR_LLM_TURN:", chrome.runtime.lastError.message);
            }
          });
        }
      } catch (error) {
        console.error("Error calling bias API:", error);
        if (message.type === "ANALYZE_TEXT") {
          sendResponse({ ok: false, error: error.message || "Unknown error" });
        }
      }
    })();

    // Only indicate async response when the caller expects a callback (ANALYZE_TEXT).
    if (message.type === "ANALYZE_TEXT") {
      return true;
    }
  }
});
