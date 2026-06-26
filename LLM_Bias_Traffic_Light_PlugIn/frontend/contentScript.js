// Content script: extracts visible text from the page when requested,
// highlights biased sentences, and (for supported LLM sites) automatically
// captures user prompt + LLM answer pair.

function isBiasExtensionUiElement(el) {
  if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
  if (el.id === "bias-overlay-hub" || el.id === "bias-overlay-show-fab") return true;
  if (el.closest?.("#bias-overlay-hub, #bias-overlay-show-fab, [data-bias-extension-ui]")) return true;
  return false;
}

function normalizeCaptureText(s) {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function isExtensionOrChatUiNoise(text) {
  const t = normalizeCaptureText(text);
  if (!t) return true;
  if (t.includes("bias detector")) return true;
  if (t.includes("bias info")) return true;
  if (t.startsWith("bias:")) return true;
  if (t.includes("bias score")) return true;
  if (t.includes("risk level")) return true;
  if (t.includes("traffic light")) return true;
  if (t.includes("embedder=")) return true;
  if (t.includes("sentence_transformers")) return true;
  if (t.includes("category scores ->")) return true;
  if (t.includes("show bias info")) return true;
  if (t.includes("currently unavailable")) return true;
  if (t.includes("learn more") && t.includes("opens in new tab")) return true;
  if (t.includes("claude is ai and can make mistakes")) return true;
  if (t.includes("please double-check responses")) return true;
  if (/^claude\s+(fable|sonnet|haiku|opus)/i.test((text || "").trim())) return true;
  if (t.includes("fable") && t.includes("unavailable")) return true;
  if (t.includes("scan page")) return true;
  if (t.includes("last captured interaction")) return true;
  if (t === "share" || t === "reply..." || t === "copy") return true;
  if (t.includes("retry") && t.length < 24) return true;
  if (t.includes("edit") && t.length < 20) return true;
  if (/^\d+\.\d{2}\s*·\s*(low|medium|high)$/i.test(t)) return true;
  return false;
}

function sanitizeCapturedMessageText(text) {
  if (!text) return "";
  const lines = (text || "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !isExtensionOrChatUiNoise(l));
  return lines.join("\n").trim();
}

function compareDomOrder(a, b) {
  if (a === b) return 0;
  const pos = a.compareDocumentPosition(b);
  if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1;
  if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1;
  return 0;
}

function extractVisibleText() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.parentElement) return NodeFilter.FILTER_REJECT;
      if (isBiasExtensionUiElement(node.parentElement)) return NodeFilter.FILTER_REJECT;
      const style = window.getComputedStyle(node.parentElement);
      if (style && (style.visibility === "hidden" || style.display === "none")) {
        return NodeFilter.FILTER_REJECT;
      }
      const trimmed = node.textContent.trim();
      return trimmed ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });

  const parts = [];
  while (walker.nextNode()) {
    parts.push(walker.currentNode.textContent.trim());
  }

  return parts.join(" ");
}

function extractSelectionPayload() {
  const sel = window.getSelection ? window.getSelection() : null;
  const selectionText = (sel?.toString?.() || "").trim();

  const findClosestMessageLikeContainer = (node) => {
    if (!node) return null;
    const el =
      node.nodeType === Node.ELEMENT_NODE
        ? node
        : node.parentElement || (node.parentNode && node.parentNode.parentElement) || null;
    if (!el) return null;

    const selectors = [
      // ChatGPT
      "div[data-message-author-role]",
      "[data-testid*='message']",
      // Common chat message wrappers
      "[role='listitem']",
      "[role='article']",
      "article",
      "div[class*='message']",
      "div[class*='group']",
      "div[class*='prose']",
      "div[class*='response']",
      "div[class*='prompt']",
    ];

    let cur = el;
    for (let depth = 0; depth < 8 && cur; depth++) {
      if (cur.matches && selectors.some((s) => cur.matches(s))) return cur;
      cur = cur.parentElement;
    }
    return null;
  };

  let expandedText = "";
  try {
    const anchorNode = sel?.anchorNode || null;
    const focusNode = sel?.focusNode || null;
    const anchorContainer = findClosestMessageLikeContainer(anchorNode);
    const focusContainer = findClosestMessageLikeContainer(focusNode);

    if (anchorContainer && focusContainer && anchorContainer !== focusContainer) {
      const range = sel?.rangeCount ? sel.getRangeAt(0) : null;
      if (range) {
        const common =
          range.commonAncestorContainer?.nodeType === Node.ELEMENT_NODE
            ? range.commonAncestorContainer
            : range.commonAncestorContainer?.parentElement || null;
        const commonMessage = findClosestMessageLikeContainer(common);
        expandedText = (commonMessage?.innerText || "").trim();
      }
    } else if (anchorContainer) {
      expandedText = (anchorContainer.innerText || "").trim();
    }
  } catch (e) {
    // best-effort only
  }

  // If selection is small, prefer expanding to a message block if available.
  const text = selectionText.length >= 20 ? selectionText : expandedText || selectionText;
  return { text, selectionText, expandedText };
}

let biasHighlights = [];
let lastSentTurnKey = null;
let lastSentTurnTimestamp = 0;
let lastSentTurnAnswerLen = 0;
let emptyConversationSince = null;
let autoDetectTimer = null;
let lastObservedUrl = location.href;

const AUTO_DETECT_DELAY_MS = 900;
const AUTO_DETECT_LOAD_DELAYS_MS = [600, 1800, 4000, 8000, 12000];

function scheduleAutoDetect(delayMs = AUTO_DETECT_DELAY_MS) {
  if (autoDetectTimer) clearTimeout(autoDetectTimer);
  autoDetectTimer = setTimeout(() => {
    autoDetectTimer = null;
    detectAndSendLatestTurn();
  }, delayMs);
}

function resetCaptureStateForNavigation() {
  lastSentTurnKey = null;
  lastSentTurnTimestamp = 0;
  lastSentTurnAnswerLen = 0;
  emptyConversationSince = null;
}

function onAppRouteMaybeChanged() {
  if (location.href === lastObservedUrl) return;
  lastObservedUrl = location.href;
  resetCaptureStateForNavigation();
  scheduleAutoDetect(500);
}

function installRouteChangeListeners() {
  if (installRouteChangeListeners.installed) return;
  installRouteChangeListeners.installed = true;

  const origPushState = history.pushState;
  const origReplaceState = history.replaceState;
  history.pushState = function (...args) {
    origPushState.apply(this, args);
    onAppRouteMaybeChanged();
  };
  history.replaceState = function (...args) {
    origReplaceState.apply(this, args);
    onAppRouteMaybeChanged();
  };
  window.addEventListener("popstate", onAppRouteMaybeChanged);
}

function clearCapturedTurnState() {
  lastSentTurnKey = null;
  lastSentTurnTimestamp = 0;
  lastSentTurnAnswerLen = 0;
  emptyConversationSince = null;

  if (window.BiasOverlayHub?.clear) {
    window.BiasOverlayHub.clear();
  }
  highlightSentences([]);

  chrome.runtime.sendMessage({ type: "CLEAR_LLM_TURN" }, () => {
    if (chrome.runtime.lastError) {
      // Background may be unavailable during extension reload.
    }
  });
}

function noteEmptyConversation() {
  if (!lastSentTurnKey) return;

  if (!emptyConversationSince) {
    emptyConversationSince = Date.now();
    return;
  }

  if (Date.now() - emptyConversationSince >= 500) {
    clearCapturedTurnState();
  }
}

function isSupportedLLMHost(hostname = location.hostname || "") {
  return (
    hostname.includes("chatgpt.com") ||
    hostname.includes("openai.com") ||
    hostname.includes("gemini.google.com") ||
    hostname.includes("ai.google.com") ||
    hostname.includes("claude.ai") ||
    hostname.includes("deepseek") ||
    hostname.includes("deepthink")
  );
}

function isDeepseekHost(hostname = location.hostname || "") {
  return hostname.includes("deepseek") || hostname.includes("deepthink");
}

function isInsideDeepseekSidebar(el) {
  if (!el?.closest) return false;
  return !!el.closest(
    'aside, nav, [class*="sidebar" i], [class*="side-bar" i], [class*="SideBar"], [class*="session-list" i], [class*="history-list" i]'
  );
}

function getDeepseekChatRoot() {
  const selectors = [
    '[class*="ds-scroll-area"]',
    '[class*="chat-content"]',
    '[class*="conversation-content"]',
    '[class*="message-list"]',
    "main",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && !isInsideDeepseekSidebar(el)) return el;
  }
  const main = document.querySelector("main");
  if (main) return main;
  return document.body;
}

function isDeepseekUiNoise(text) {
  if (isExtensionOrChatUiNoise(text)) return true;
  const tl = normalizeCaptureText(text);
  if (!tl) return true;

  const blockedExact = [
    "deepthink",
    "deepthink search",
    "deep thinking",
    "search",
    "new chat",
    "message deepseek",
    "how can i help you today",
    "how can i help",
  ];
  if (blockedExact.includes(tl)) return true;

  if (tl.includes("new chat") || tl.includes("regenerate") || tl.includes("sign in")) return true;
  if (tl.includes("one more step before you proceed") || tl.includes("captcha")) return true;
  if (tl.includes("message deepseek")) return true;
  if (tl.includes("upload") && tl.length < 40) return true;
  if (tl.includes("ai-generated") && tl.includes("reference")) return true;
  if (tl.includes("for reference only")) return true;
  if (tl.length <= 24 && (tl.includes("deepthink") || tl.includes("deep thinking"))) return true;
  return false;
}

function isNumericOrUiGarbage(text) {
  const t = (text || "").trim();
  if (!t) return true;
  if (/^\d+$/.test(t)) return true;
  if (/^\d+[\.,]\d+$/.test(t)) return true;
  if (/^\d+\s*(messages?|tokens?|chars?)?$/i.test(t)) return true;
  return false;
}

function normalizePairOverlap(a, b) {
  const aa = normalizeCaptureText(a);
  const bb = normalizeCaptureText(b);
  if (!aa || !bb) return 0;
  if (aa === bb) return 1;
  const short = aa.length <= bb.length ? aa : bb;
  const long = aa.length > bb.length ? aa : bb;
  if (short.length < 20) return 0;
  if (long.includes(short)) return short.length / long.length;
  return 0;
}

function looksQuestionLikeText(s) {
  const t = (s || "").trim();
  if (!t) return false;
  return /\?\s*$/.test(t) || t.includes("?");
}

function passesDeepseekTurnValidation(prompt, answer) {
  if (!hasMeaningfulConversationPair({ prompt, answer })) return false;

  const promptText = (prompt || "").trim();
  const answerText = (answer || "").trim();
  if (promptText.length < 3 || answerText.length < 8) return false;
  if (isDeepseekUiNoise(promptText) || isDeepseekUiNoise(answerText)) return false;
  if (isNumericOrUiGarbage(promptText) || isNumericOrUiGarbage(answerText)) return false;

  const tlPrompt = normalizeCaptureText(promptText);
  const tlAnswer = normalizeCaptureText(answerText);
  if (tlPrompt.includes("deepthink") && tlPrompt.includes("search")) return false;
  if (tlPrompt === "deepthink" || tlPrompt === "deepthink search" || tlPrompt === "deep thinking") {
    return false;
  }
  if (tlAnswer.includes("ai-generated") && tlAnswer.includes("reference") && tlAnswer.length < 120) {
    return false;
  }
  if (tlAnswer.includes("for reference only") && tlAnswer.length < 120) return false;
  if (normalizePairOverlap(promptText, answerText) >= 0.82) return false;
  if (
    looksQuestionLikeText(answerText) &&
    answerText.length <= promptText.length * 1.05
  ) {
    return false;
  }
  return true;
}

function isDeepseekEmptyChatState() {
  const root = getDeepseekChatRoot();
  const messageSelector = [
    '[data-message-author-role="user"]',
    '[data-message-author-role="assistant"]',
    ".ds-message",
    ".ds-chat-message",
    '[class*="UserMessage"]',
    '[class*="AssistantMessage"]',
  ].join(", ");
  const messageEls = Array.from(root.querySelectorAll(messageSelector)).filter(
    (el) => !isInsideDeepseekSidebar(el) && !isBiasExtensionUiElement(el)
  );
  if (messageEls.length === 0) return true;

  const messages = getDeepseekChatMessagesOrdered();
  return messages.length === 0;
}

function getDeepseekRoleForMessageEl(el) {
  if (!el) return "unknown";

  const authorRole = (el.getAttribute("data-message-author-role") || "").toLowerCase();
  if (authorRole === "user" || authorRole === "assistant") return authorRole;

  const dataRole = (el.getAttribute("data-role") || "").toLowerCase();
  if (dataRole === "user" || dataRole === "assistant") return dataRole;

  const testId = (el.getAttribute("data-testid") || "").toLowerCase();
  if (testId.includes("user") || testId.includes("human")) return "user";
  if (testId.includes("assistant") || testId.includes("model")) return "assistant";

  const cls = String(el.className || "");
  if (/UserMessage|user-message|userMessage/i.test(cls)) return "user";
  if (/AssistantMessage|assistant-message|assistantMessage/i.test(cls)) return "assistant";

  const wrapper = el.closest?.(
    '[data-message-author-role], .ds-message, .ds-chat-message, [class*="UserMessage"], [class*="AssistantMessage"]'
  );
  if (wrapper && wrapper !== el) return getDeepseekRoleForMessageEl(wrapper);

  if (el.matches?.('.ds-markdown, .markdown-body, [class*="markdown"]')) {
    const assistantParent = el.closest(
      '[data-message-author-role="assistant"], [class*="AssistantMessage"], [class*="assistant-message"]'
    );
    if (assistantParent) return "assistant";
  }

  return getRoleForMessageEl(el);
}

function getDeepseekTextForMessageEl(el) {
  if (!el || isBiasExtensionUiElement(el)) return "";
  const markdown = el.querySelector?.('.ds-markdown, .markdown-body, [class*="markdown-body"]');
  if (markdown && !isBiasExtensionUiElement(markdown)) {
    const text = sanitizeCapturedMessageText((markdown.innerText || "").trim());
    if (text) return text;
  }
  return getTextForMessageEl(el);
}

function getDeepseekChatMessagesOrdered() {
  const root = getDeepseekChatRoot();
  const deepseekSelector = [
    '[data-message-author-role="user"]',
    '[data-message-author-role="assistant"]',
    '[data-message-author-role]',
    '[data-role="user"]',
    '[data-role="assistant"]',
    ".ds-message",
    ".ds-chat-message",
    '[class*="UserMessage"]',
    '[class*="AssistantMessage"]',
    '[data-testid*="user-message"]',
    '[data-testid*="assistant-message"]',
  ].join(", ");

  const els = queryTopLevelMessageEls(root, deepseekSelector)
    .filter((el) => !isBiasExtensionUiElement(el) && !isInsideDeepseekSidebar(el))
    .sort(compareDomOrder);

  const seen = new Set();
  const messages = [];

  for (const el of els) {
    const role = getDeepseekRoleForMessageEl(el);
    if (role !== "user" && role !== "assistant") continue;

    const text = getDeepseekTextForMessageEl(el);
    if (!text || text.length < 2 || isDeepseekUiNoise(text)) continue;

    const key = `${role}:${normalizeCaptureText(text).slice(0, 120)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    messages.push({ role, text });
  }

  return messages;
}

function hasMeaningfulConversationPair(pair) {
  if (!pair) return false;
  const prompt = (pair.prompt || "").trim();
  const answer = (pair.answer || "").trim();
  if (!prompt || !answer) return false;
  if (prompt.length < 3 || answer.length < 8) return false;
  if (prompt === answer) return false;

  const normalize = (s) =>
    (s || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  const p = normalize(prompt);
  const a = normalize(answer);
  if (!p || !a) return false;

  const blockedPromptLabels = [
    "new chat",
    "start a new chat",
    "message",
    "ask",
    "search",
  ];
  if (blockedPromptLabels.includes(p)) return false;

  const shortUiNoise = [
    "bias detector",
    "bias score",
    "risk level",
    "scan page",
    "traffic light",
    "embedder=",
    "sentence_transformers",
    "currently unavailable",
    "learn more",
    "opens in new tab",
  ];
  if (shortUiNoise.some((x) => p.includes(x) || a.includes(x))) return false;

  const short = p.length <= a.length ? p : a;
  const long = p.length > a.length ? p : a;
  if (short.length >= 20 && long.includes(short) && short.length / long.length >= 0.82) {
    return false;
  }
  return true;
}

// --- Conversation turn listing (popup picker) ---

function queryTopLevelMessageEls(root, selector) {
  const all = Array.from(root.querySelectorAll(selector));
  return all.filter((el) => !all.some((other) => other !== el && other.contains(el)));
}

function getChatMessagesOrdered() {
  const hostname = location.hostname || "";
  const root = document.querySelector("main") || document.body;

  if (hostname.includes("chatgpt.com") || hostname.includes("openai.com")) {
    return queryTopLevelMessageEls(root, "div[data-message-author-role]")
      .map((el) => ({
        role: (el.getAttribute("data-message-author-role") || "unknown").toLowerCase(),
        text: getTextForMessageEl(el),
      }))
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.text.length > 5);
  }

  if (hostname.includes("claude.ai")) {
    const claudeMessageSelector = [
      "[data-testid='user-message']",
      "[data-testid='human-message']",
      ".font-user-message",
      ".font-claude-response",
      "[data-testid='assistant-message']",
      "[data-testid='ai-message']",
      "[data-testid='message-assistant']",
      "[data-message-author-role]",
    ].join(", ");

    const els = queryTopLevelMessageEls(root, claudeMessageSelector)
      .filter((el) => !isBiasExtensionUiElement(el))
      .sort(compareDomOrder);

    return els
      .map((el) => ({
        role: getClaudeRoleForMessageEl(el),
        text: getTextForMessageEl(el),
      }))
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.text.length > 2)
      .filter((m) => !isExtensionOrChatUiNoise(m.text));
  }

  if (hostname.includes("gemini.google.com") || hostname.includes("ai.google.com")) {
    const geminiSelector = [
      '[data-turn-role]',
      '[data-message-author-role]',
      '[class*="user-query"]',
      '[class*="model-response"]',
      '[class*="conversation-turn"]',
    ].join(", ");
    return queryTopLevelMessageEls(root, geminiSelector)
      .map((el) => ({
        role: getRoleForMessageEl(el),
        text: getTextForMessageEl(el),
      }))
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.text.length > 5);
  }

  if (hostname.includes("deepseek") || hostname.includes("deepthink")) {
    return getDeepseekChatMessagesOrdered();
  }

  const genericSelector = [
    "div[data-message-author-role]",
    "[data-turn-role]",
    "[data-role='user']",
    "[data-role='assistant']",
    "[data-testid='user-message']",
    "[data-testid='assistant-message']",
    "[data-testid*='message']",
  ].join(", ");

  const normalize = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();
  const seen = new Set();
  const messages = [];

  for (const el of queryTopLevelMessageEls(root, genericSelector)) {
    const text = getTextForMessageEl(el);
    if (!text || text.length < 5) continue;
    const role = getRoleForMessageEl(el);
    if (role !== "user" && role !== "assistant") continue;
    const key = `${role}:${normalize(text).slice(0, 120)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    messages.push({ role, text });
  }

  return messages;
}

function pairFromOrderedMessages() {
  const messages = getChatMessagesOrdered();
  if (!Array.isArray(messages) || messages.length < 2) return null;

  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }
  if (lastAssistantIdx === -1) return null;

  let lastUserIdx = -1;
  for (let i = lastAssistantIdx - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx === -1) return null;

  const prompt = (messages[lastUserIdx].text || "").trim();
  const answer = (messages[lastAssistantIdx].text || "").trim();
  if (!prompt || !answer) return null;
  return { prompt, answer };
}

function buildTurnLabel(prompt, question) {
  const source = (question || prompt || "").replace(/\s+/g, " ").trim();
  if (!source) return "Conversation";
  return source.length > 52 ? `${source.slice(0, 52)}…` : source;
}

function buildTurnsFromMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return [];

  const turns = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role !== "user") continue;

    const prompt = sanitizeCapturedMessageText((messages[i].text || "").trim());
    if (!prompt || isExtensionOrChatUiNoise(prompt)) continue;
    const answerParts = [];
    for (let j = i + 1; j < messages.length; j++) {
      if (messages[j].role === "user") break;
      if (messages[j].role === "assistant") {
        const part = sanitizeCapturedMessageText((messages[j].text || "").trim());
        if (part && !isExtensionOrChatUiNoise(part)) answerParts.push(part);
      }
    }

    const answer = answerParts.join("\n\n").trim();
    if (!prompt || !answer) continue;
    if (!hasMeaningfulConversationPair({ prompt, answer })) continue;

    const parsed =
      typeof splitPromptContextAndAnswer === "function"
        ? splitPromptContextAndAnswer(prompt)
        : { question: prompt, context: "", answerFromPrompt: "" };

    turns.push({
      id: turns.length,
      prompt,
      question: parsed.question || prompt,
      context: parsed.context || "",
      answer,
      label: buildTurnLabel(prompt, parsed.question),
    });
  }

  return turns;
}

function getConversationTurnsPayload() {
  const isDeepseek = isDeepseekHost();
  const messages = getChatMessagesOrdered();
  let turns = buildTurnsFromMessages(messages);

  if (isDeepseek) {
    turns = turns.filter((turn) => passesDeepseekTurnValidation(turn.prompt, turn.answer));

    if (turns.length === 0 && !isDeepseekEmptyChatState()) {
      const latest = extractLatestTurnForDeepseek();
      if (latest && passesDeepseekTurnValidation(latest.prompt, latest.answer)) {
        const parsed =
          typeof splitPromptContextAndAnswer === "function"
            ? splitPromptContextAndAnswer(latest.prompt || "")
            : { question: latest.prompt || "", context: "", answerFromPrompt: "" };
        turns = [
          {
            id: 0,
            prompt: latest.prompt || "",
            question: parsed.question || latest.prompt || "",
            context: parsed.context || "",
            answer: latest.answer || "",
            label: buildTurnLabel(latest.prompt, parsed.question),
          },
        ];
      }
    }

    return turns;
  }

  if (turns.length === 0) {
    const latest = extractLatestTurnForCurrentSite();
    if (latest && hasMeaningfulConversationPair(latest)) {
      const parsed =
        typeof splitPromptContextAndAnswer === "function"
          ? splitPromptContextAndAnswer(latest.prompt || "")
          : { question: latest.prompt || "", context: "", answerFromPrompt: "" };
      turns = [
        {
          id: 0,
          prompt: latest.prompt || "",
          question: parsed.question || latest.prompt || "",
          context: parsed.context || "",
          answer: latest.answer || "",
          label: buildTurnLabel(latest.prompt, parsed.question),
        },
      ];
    }
  }

  return turns;
}

function getClaudeRoleForMessageEl(el) {
  if (!el) return "unknown";
  const testId = (el.getAttribute?.("data-testid") || "").toLowerCase();
  if (testId === "user-message" || testId === "human-message" || testId.includes("human")) {
    return "user";
  }
  if (
    testId === "assistant-message" ||
    testId === "ai-message" ||
    testId.includes("assistant") ||
    testId.includes("ai-message")
  ) {
    return "assistant";
  }

  const cls = String(el.className || "").toLowerCase();
  if (cls.includes("font-claude-response") || cls.includes("claude-response")) return "assistant";
  if (cls.includes("font-user-message") || cls.includes("user-message")) return "user";

  const explicit = el.getAttribute?.("data-message-author-role");
  if (explicit) return explicit.toLowerCase();

  return getRoleForMessageEl(el);
}

function getRoleForMessageEl(el) {
  if (!el) return "unknown";
  const testId = (el.getAttribute?.("data-testid") || "").toLowerCase();
  if (testId === "assistant-message" || testId.includes("assistant-message") || testId === "ai-message") {
    return "assistant";
  }
  if (testId === "user-message" || testId === "human-message" || testId.includes("human-message")) {
    return "user";
  }

  const explicit = el.getAttribute?.("data-message-author-role");
  if (explicit) return explicit.toLowerCase();

  const hints = [
    el.getAttribute?.("data-turn-role"),
    el.getAttribute?.("data-role"),
    el.getAttribute?.("data-testid"),
    el.getAttribute?.("aria-label"),
    el.className,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (
    hints.includes("assistant") ||
    hints.includes("model-response") ||
    hints.includes("font-claude-response") ||
    hints.includes("gemini")
  ) {
    return "assistant";
  }
  if (hints.includes("user") || hints.includes("human") || hints.includes("prompt") || hints.includes("query")) {
    return "user";
  }
  return "unknown";
}

function getTextForMessageEl(el) {
  if (!el || isBiasExtensionUiElement(el)) return "";
  const innerSelectors = [
    "[data-message-content-inner]",
    ".standard-markdown",
    ".progressive-markdown",
    ".markdown",
    ".prose",
  ];
  for (const selector of innerSelectors) {
    const inner = el.querySelector?.(selector);
    if (inner && !isBiasExtensionUiElement(inner)) {
      const text = sanitizeCapturedMessageText((inner.innerText || "").trim());
      if (text) return text;
    }
  }
  const content = el.querySelector?.("[data-message-content-inner]") || el;
  return sanitizeCapturedMessageText((content.innerText || "").trim());
}

function clearBiasHighlights() {
  biasHighlights.forEach((span) => {
    const parent = span.parentNode;
    if (!parent) return;
    const text = document.createTextNode(span.textContent);
    parent.replaceChild(text, span);
    parent.normalize();
  });
  biasHighlights = [];
}

function highlightSentenceOnce(sentence) {
  if (!sentence) return;

  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node.parentElement) return NodeFilter.FILTER_REJECT;
        if (isBiasExtensionUiElement(node.parentElement)) return NodeFilter.FILTER_REJECT;
        const style = window.getComputedStyle(node.parentElement);
        if (
          style &&
          (style.visibility === "hidden" || style.display === "none")
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        const trimmed = node.textContent;
        return trimmed && trimmed.includes(sentence)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    }
  );

  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.textContent;
    const index = text.indexOf(sentence);
    if (index === -1) continue;

    const before = text.slice(0, index);
    const match = text.slice(index, index + sentence.length);
    const after = text.slice(index + sentence.length);

    const highlightSpan = document.createElement("span");
    highlightSpan.textContent = match;
    highlightSpan.style.backgroundColor = "#facc15";
    highlightSpan.style.color = "inherit";
    highlightSpan.style.borderRadius = "4px";
    highlightSpan.style.padding = "0 1px";

    const fragment = document.createDocumentFragment();
    if (before) fragment.appendChild(document.createTextNode(before));
    fragment.appendChild(highlightSpan);
    if (after) fragment.appendChild(document.createTextNode(after));

    const parent = node.parentNode;
    if (parent) {
      parent.replaceChild(fragment, node);
      biasHighlights.push(highlightSpan);
    }

    // Only highlight the first occurrence per sentence for now.
    break;
  }
}

function highlightSentences(sentences) {
  clearBiasHighlights();
  if (!Array.isArray(sentences)) return;
  sentences.forEach((s) => highlightSentenceOnce(s));
}

// --- Automatic LLM turn capture for multiple providers (ChatGPT, Gemini, Claude, etc.) ---

function extractLatestTurnForChatGPT() {
  console.debug("extractLatestTurnForChatGPT start");

  const main = document.querySelector("main") || document.body;

  // Prefer role-based extraction to avoid swapping user/assistant bubbles.
  // ChatGPT commonly exposes message roles via `data-message-author-role`.
  const messageEls = Array.from(main.querySelectorAll("div[data-message-author-role]"));
  if (messageEls.length > 0) {
    const messages = messageEls
      .map((el) => {
        const role = el.getAttribute("data-message-author-role");
        // Most message text lives under this container.
        const content = el.querySelector("[data-message-content-inner]") || el;
        const text = (content.innerText || "").trim();
        return { role, text };
      })
      .filter((m) => m.role && m.text && m.text.length > 5);

    // Find the latest assistant message, then the closest user message before it.
    let lastAssistantIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }

    if (lastAssistantIdx !== -1) {
      let lastUserIdx = -1;
      for (let i = lastAssistantIdx - 1; i >= 0; i--) {
        if (messages[i].role === "user") {
          lastUserIdx = i;
          break;
        }
      }

      if (lastUserIdx !== -1) {
        const prompt = messages[lastUserIdx].text;
        const answer = messages[lastAssistantIdx].text;
        if (prompt && answer) {
          console.debug("extractLatestTurnForChatGPT role-based", { prompt, answer });
          return { prompt, answer };
        }
      }
    }
  }

  // Fallback: ChatGPT UI often has grouped conversation bubbles under main.
  const groupNodes = Array.from(
    main.querySelectorAll(
      "[data-testid='message-text'], [data-testid='message-bubble'], div[class*='group'], div[class*='result']"
    )
  );
  const candidateTexts = groupNodes
    .map((node) => node.innerText?.trim())
    .filter((text) => text && text.length > 15);

  if (candidateTexts.length >= 2) {
    const prompt = candidateTexts[candidateTexts.length - 2];
    const answer = candidateTexts[candidateTexts.length - 1];
    if (prompt && answer) {
      console.debug("extractLatestTurnForChatGPT group fallback", { prompt, answer });
      return { prompt, answer };
    }
  }

  // If this still fails, use fallback from all chat-like list items in main.
  const itemNodes = Array.from(
    main.querySelectorAll("[role='listitem'], [role='article'], [data-testid*='message'], [aria-label*='message'], .group, .message")
  );
  const itemTexts = itemNodes.map((n) => n.innerText?.trim()).filter((t) => t && t.length > 12);
  if (itemTexts.length >= 2) {
    const prompt = itemTexts[itemTexts.length - 2];
    const answer = itemTexts[itemTexts.length - 1];
    if (prompt && answer) {
      console.debug("extractLatestTurnForChatGPT item fallback", { prompt, answer });
      return { prompt, answer };
    }
  }

  // Final generic fallback: last two long lines from visible page text.
  const lines = (main.innerText || document.body.innerText || "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 10);
  if (lines.length >= 2) {
    const prompt = lines[lines.length - 2];
    const answer = lines[lines.length - 1];
    if (prompt && answer) {
      console.debug("extractLatestTurnForChatGPT line fallback", { prompt, answer });
      return { prompt, answer };
    }
  }

  return null;
}

function extractLatestTurnForGemini() {
  const root =
    document.querySelector('main [aria-label^="Conversation"], [aria-label^="Conversation"], [role="feed"], main') ||
    document.body;

  const textLooksLikeUiChrome = (text) => {
    const t = (text || "").toLowerCase().trim();
    if (!t || t.length < 8) return true;
    // Reject likely account/email chips and short metadata labels.
    if (/@/.test(t) && t.length < 120) return true;
    const blocked = [
      "bias detector",
      "bias score",
      "bias:",
      "scan page",
      "automatic scan of latest llm reply",
      "last captured interaction",
      "prompt not detected",
      "context not detected",
      "answer not detected",
      "risk level",
      "thinking depth",
      "speed mode",
      "new chat",
      "gem manager",
      "settings",
      "help",
      "privacy",
      "prototype",
      "demander à gemini",
      "gemini peut se tromper",
      "gemini a dit",
      "gemini said",
      "vous avez dit",
      "you said",
      "tools",
    ];
    return blocked.some((s) => t.includes(s));
  };

  const classifyRole = (el) => {
    const roleHints = [
      el.getAttribute("data-turn-role"),
      el.getAttribute("data-message-author-role"),
      el.getAttribute("aria-label"),
      el.getAttribute("data-testid"),
      el.className,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    if (
      roleHints.includes("assistant") ||
      roleHints.includes("model") ||
      roleHints.includes("gemini") ||
      roleHints.includes("response")
    ) {
      return "assistant";
    }
    if (
      roleHints.includes("user") ||
      roleHints.includes("prompt") ||
      roleHints.includes("query")
    ) {
      return "user";
    }
    return null;
  };

  const candidateSelectors = [
    // Gemini/Bard-ish specific hooks first
    '[class*="user-query"]',
    '[class*="query-text"]',
    '[class*="query-content"]',
    '[class*="model-response"]',
    '[class*="response-content"]',
    '[class*="message-content"]',
    '[class*="conversation-turn"]',
    '[data-turn-role]',
    '[data-message-author-role]',
    '[data-testid*="message"]',
    '[data-testid*="chat"]',
    '[role="listitem"]',
    '[role="article"]',
    'div[class*="message"]',
    'div[class*="response"]',
    'div[class*="prompt"]',
    'div[class*="query"]',
    'article',
  ];

  const rawCandidates = Array.from(root.querySelectorAll(candidateSelectors.join(", ")));
  const candidates = rawCandidates
    .map((el) => {
      const text = (el.innerText || "").trim();
      return { text, role: classifyRole(el) };
    })
    .filter((c) => c.text.length >= 12 && c.text.length <= 25000)
    .filter((c) => !textLooksLikeUiChrome(c.text));

  // Deduplicate repeated wrappers that carry identical text.
  const deduped = [];
  for (const c of candidates) {
    if (!deduped.length || deduped[deduped.length - 1].text !== c.text) {
      deduped.push(c);
    }
  }

  const normalizeForCompare = (s) =>
    (s || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();

  const overlapScore = (a, b) => {
    const aa = normalizeForCompare(a);
    const bb = normalizeForCompare(b);
    if (!aa || !bb) return 0;
    if (aa === bb) return 1;
    const minLen = Math.min(aa.length, bb.length);
    if (minLen < 20) return 0;
    const short = aa.length <= bb.length ? aa : bb;
    const long = aa.length > bb.length ? aa : bb;
    if (long.includes(short)) return short.length / long.length;
    return 0;
  };

  const findAssistantAnswerCandidate = (promptText) => {
    const assistantSelectors = [
      '[data-turn-role*="model" i]',
      '[data-turn-role*="assistant" i]',
      '[data-message-author-role*="assistant" i]',
      '[class*="model-response"]',
      '[class*="response-content"]',
      '[class*="assistant"]',
      'message-content[class*="model"]',
    ];
    const blocks = Array.from(root.querySelectorAll(assistantSelectors.join(", ")))
      .map((el) => (el.innerText || "").trim())
      .filter((t) => t.length >= 24 && t.length <= 25000)
      .filter((t) => !textLooksLikeUiChrome(t));

    const unique = [];
    for (const t of blocks) {
      if (!unique.length || normalizeForCompare(unique[unique.length - 1]) !== normalizeForCompare(t)) {
        unique.push(t);
      }
    }

    // Gemini often splits one assistant answer into multiple adjacent chunks.
    // Build merged candidates from the tail to recover the full response.
    const mergedCandidates = [];
    for (let end = unique.length - 1; end >= 0; end--) {
      const maxChunks = 10;  // Increased from 4 to capture more chunks from streaming responses
      for (let chunks = 1; chunks <= maxChunks; chunks++) {
        const start = end - chunks + 1;
        if (start < 0) break;
        const merged = unique.slice(start, end + 1).join("\n\n").trim();
        if (merged.length >= 24 && merged.length <= 80000) {  // Increased from 35000
          mergedCandidates.push(merged);
        }
      }
    }

    // Prefer richer answers (multi-paragraph / longer), while ensuring they are
    // not just a copy of the user prompt/context block.
    let best = null;
    let bestScore = -Infinity;
    const allCandidates = [...mergedCandidates, ...unique];
    for (const candidate of allCandidates) {
      const ov = overlapScore(candidate, promptText);
      if (ov >= 0.7) continue;

      let score = 0;
      const paras = candidate.split(/\n{2,}/).filter((p) => p.trim().length > 0).length;
      score += Math.min(4, paras); // reward multi-paragraph responses
      score += Math.min(6, Math.floor(candidate.length / 220)); // reward fuller answers
      if (candidate.includes("\n")) score += 1;

      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    if (best) return best;

    // Last fallback: latest single assistant block that is not prompt-like.
    for (let i = unique.length - 1; i >= 0; i--) {
      const candidate = unique[i];
      if (overlapScore(candidate, promptText) < 0.7) return candidate;
    }
    return null;
  };

  const finalizePair = (prompt, answer) => {
    if (!prompt || !answer) return null;
    
    // Also clean the prompt to remove "Vous avez dit" / "You said" markers
    const cleanPrompt = (prompt || "")
      .replace(/(?:vous avez dit|you said)\s*[:\-–]?\s*/gi, "")
      .trim();
    
    const cleanAnswer = (answer || "")
      .replace(/(?:gemini a dit|gemini said)\s*[:\-–]?\s*/gi, "")
      .split("\n")
      .map((l) => l.trimEnd())
      .filter((l) => {
        const ll = l.trim().toLowerCase();
        if (!ll) return false;
        if (ll.startsWith("bias score")) return false;
        if (ll.startsWith("risk level")) return false;
        if (ll.startsWith("explanation")) return false;
        if (ll === "0" || ll === "low" || ll === "medium" || ll === "high") return false;
        // Gemini UI footer / controls (Gemini uses these as part of `innerText` sometimes).
        if (ll === "outils" || ll === "rapide") return false;
        if (ll.includes("gemini est une ia")) return false;
        if (ll.includes("votre confidentialité")) return false;
        if (ll.includes("s'ouvre dans une nouvelle fenêtre")) return false;
        if (ll.includes("your privacy") || ll.includes("gemini can") || ll.includes("gemini may")) return false;
        if (ll.startsWith("bias:")) return false;
        // Extension risk badge fragments
        if (ll.startsWith("bias:") || ll.startsWith("risk level:")) return false;
        // Gemini UI markers that should be removed from captured response
        if (ll.startsWith("gemini a dit") || ll.startsWith("gemini said")) return false;
        if (ll.startsWith("vous avez dit") || ll.startsWith("you said")) return false;
        return true;
      })
      .join("\n")
      .trim();
    if (!cleanAnswer) return null;

    // Remove duplicate content that appears in the answer (e.g., when "Gemini a dit" appears twice)
    // Split by markers and take only the last occurrence of unique content
    const markerSplit = cleanAnswer.split(/(?:gemini a dit|gemini said|vous avez dit|you said)\s*[:\-–]?\s*/gi);
    let dedupedAnswer = markerSplit[markerSplit.length - 1]?.trim() || cleanAnswer;
    // If the last part is empty or too short, try the second to last
    if (dedupedAnswer.length < 20 && markerSplit.length > 1) {
      dedupedAnswer = markerSplit[markerSplit.length - 2]?.trim() || cleanAnswer;
    }

    const ov = overlapScore(cleanPrompt, dedupedAnswer);
    if (ov < 0.7) return { prompt: cleanPrompt, answer: dedupedAnswer };

    // If answer duplicates prompt/context block, replace with model-only candidate.
    const altAnswer = findAssistantAnswerCandidate(cleanPrompt);
    if (altAnswer && overlapScore(cleanPrompt, altAnswer) < 0.7) {
      return { prompt: cleanPrompt, answer: altAnswer };
    }
    return null;
  };

  // Gemini UI often includes explicit text markers in `innerText`
  // such as "Gemini a dit" / "Vous avez dit" (FR) or "Gemini said" / "You said" (EN).
  // Use those labels to extract the full latest assistant block, which fixes truncation
  // caused by DOM chunking.
  const extractLatestTurnForGeminiByTextMarkers = () => {
    const pageText = document.body.innerText || "";
    if (!pageText) return null;

    const combinedRe = /(Gemini a dit|Gemini said|Vous avez dit|You said)\s*[:\-–]?\s*/gi;
    const matches = [];
    let m;
    while ((m = combinedRe.exec(pageText)) !== null) {
      const label = (m[1] || "").toLowerCase();
      const type = label.includes("gemini") ? "assistant" : "user";
      matches.push({ type, start: m.index + m[0].length });
    }
    if (matches.length < 2) return null;

    const blocks = matches.map((x, idx) => {
      const end = idx + 1 < matches.length ? matches[idx + 1].start : pageText.length;
      const text = (pageText.slice(x.start, end) || "").trim();
      return { type: x.type, text };
    });

    // Find last assistant block and nearest preceding user block.
    let lastAssistantIdx = -1;
    for (let i = blocks.length - 1; i >= 0; i--) {
      if (blocks[i].type === "assistant" && blocks[i].text) {
        lastAssistantIdx = i;
        break;
      }
    }
    if (lastAssistantIdx === -1) return null;

    let lastUserIdx = -1;
    for (let i = lastAssistantIdx - 1; i >= 0; i--) {
      if (blocks[i].type === "user" && blocks[i].text) {
        lastUserIdx = i;
        break;
      }
    }
    if (lastUserIdx === -1) return null;

    const promptText = blocks[lastUserIdx].text;
    const answerText = blocks[lastAssistantIdx].text;

    if (textLooksLikeUiChrome(promptText) || textLooksLikeUiChrome(answerText)) return null;
    return finalizePair(promptText, answerText);
  };

  const markerPair = extractLatestTurnForGeminiByTextMarkers();
  if (markerPair) return markerPair;

  // Gemini can split one model reply into multiple adjacent chunks.
  // Starting from an assistant chunk index, merge following assistant/neutral
  // chunks until the next explicit user chunk.
  const expandAssistantAnswerFromIndex = (assistantIdx, promptText) => {
    if (assistantIdx < 0 || assistantIdx >= deduped.length) return null;

    // Walk backward to include leading sentence fragments from the same model turn.
    let start = assistantIdx;
    for (let i = assistantIdx - 1; i >= 0; i--) {
      const item = deduped[i];
      if (!item || !item.text) continue;
      if (item.role === "user") break;
      if (textLooksLikeUiChrome(item.text)) break;
      // Short question-like neutral nodes are likely a user prompt; don't include.
      if (item.role == null && /\?\s*$/.test(item.text.trim()) && item.text.length < 220) break;
      start = i;
      if (assistantIdx - start >= 8) break;  // Increased from 4 to capture more leading fragments
    }

    const parts = [];
    for (let i = start; i < deduped.length && parts.length < 15; i++) {  // Increased from 10
      const item = deduped[i];
      if (!item || !item.text) continue;
      if (i > assistantIdx && item.role === "user") break;
      if (textLooksLikeUiChrome(item.text)) continue;

      // Keep assistant or neutral chunks; skip clearly user-like questions
      // that can appear later in the feed.
      if (item.role === "user") continue;
      if (item.role == null && /\?\s*$/.test(item.text.trim()) && item.text.length < 220) {
        continue;
      }

      const last = parts[parts.length - 1];
      if (!last || normalizeForCompare(last) !== normalizeForCompare(item.text)) {
        parts.push(item.text);
      }
    }

    if (parts.length === 0) return null;
    const merged = parts.join("\n\n").trim();
    if (!merged) return null;
    if (overlapScore(merged, promptText) >= 0.7) return null;
    return merged;
  };

  // Preferred path: role-aware extraction (latest assistant + closest preceding user).
  if (deduped.length >= 2) {
    const buildAnswerFromUserIndex = (userIdx) => {
      const promptText = deduped[userIdx]?.text || "";
      if (!promptText) return null;

      const parts = [];
      const maxChunks = 150; // Increased from 80 to capture longer streaming responses
      const maxLen = 80000; // Increased from 50000 to allow longer answers
      const isLikelyNextUserTurn = (text) => {
        const s = (text || "").trim();
        const lower = s.toLowerCase();
        if (!s) return false;
        // If Gemini mislabels assistant chunks as user, they often start like this.
        if (lower.startsWith("based on")) return false;
        if (lower.startsWith("would you like")) return false;
        if (lower.startsWith("would you")) return false;
        // Common LLM closing questions - don't treat as next user turn.
        if (lower.includes("would you like me")) return false;
        if (lower.includes("is there anything else")) return false;
        if (lower.includes("do you have any")) return false;
        if (lower.includes("let me know if")) return false;
        if (lower.includes("feel free to ask")) return false;
        if (lower.includes("anything else i can help")) return false;
        // Only stop for short, likely prompt-like snippets.
        if (s.length > 400) return false;
        // Make the condition stricter: require prompt punctuation patterns.
        if (s.length > 100 && /\?/.test(s)) return false;
        if (!/[?!]/.test(s)) return false;
        return true;
      };
      for (
        let i = userIdx + 1;
        i < deduped.length &&
        parts.join("\n\n").length < maxLen &&
        i < userIdx + 1 + maxChunks;
        i++
      ) {
        const item = deduped[i];
        if (!item || !item.text) continue;
        // Stop only when we likely reached the next user turn.
        // Gemini can mislabel assistant chunks as user, so we avoid stopping
        // on common assistant-start patterns.
        if (item.role === "user" && isLikelyNextUserTurn(item.text)) break;
        if (textLooksLikeUiChrome(item.text)) continue;

        // Avoid leading UI noise fragments.
        // Gemini can split the first sentence into multiple small DOM chunks.
        if (parts.length === 0 && item.text.trim().length < 4) continue;

        const last = parts[parts.length - 1];
        if (!last || normalizeForCompare(last) !== normalizeForCompare(item.text)) {
          parts.push(item.text);
        }
      }
      if (!parts.length) return null;
      const merged = parts.join("\n\n").trim();
      if (!merged) return null;
      return finalizePair(promptText, merged);
    };

    const buildPairFromAssistantAnchor = (assistantIdx) => {
      if (assistantIdx < 0) return null;

      // Expand backward to include leading assistant fragments from the same turn.
      let start = assistantIdx;
      for (let i = assistantIdx - 1; i >= 0; i--) {
        const it = deduped[i];
        if (!it || !it.text) continue;
        if (it.role === "user" && it.text.trim().length <= 800) {
          const s = (it.text || "").trim().toLowerCase();
          if (!s.startsWith("based on") && !s.startsWith("would you like") && !s.startsWith("would you")) break;
        }
        if (textLooksLikeUiChrome(it.text)) break;

        // Stop on likely prompt/question line boundaries.
        if (it.text.trim().length < 220 && /\?\s*$/.test(it.text.trim())) break;
        start = i;
        if (assistantIdx - start >= 10) break;
      }

      // Expand forward until next user turn starts.
      let end = assistantIdx;
      for (let i = assistantIdx + 1; i < deduped.length; i++) {
        const it = deduped[i];
        if (!it || !it.text) continue;
        if (it.role === "user" && it.text.trim().length <= 800) {
          const s = (it.text || "").trim().toLowerCase();
          if (!s.startsWith("based on") && !s.startsWith("would you like") && !s.startsWith("would you")) break;
        }
        if (textLooksLikeUiChrome(it.text)) continue;
        end = i;
        if (end - assistantIdx >= 30) break;
      }

      const answerParts = [];
      for (let i = start; i <= end; i++) {
        const it = deduped[i];
        if (!it || !it.text) continue;
        if (textLooksLikeUiChrome(it.text)) continue;
        // Allow very small leading fragments; Gemini can split sentences.
        if (answerParts.length === 0 && it.text.trim().length < 4) continue;

        const last = answerParts[answerParts.length - 1];
        if (!last || normalizeForCompare(last) !== normalizeForCompare(it.text)) {
          answerParts.push(it.text);
        }
      }

      const answerMerged = answerParts.join("\n\n").trim();
      if (!answerMerged) return null;

      // Pick nearest previous user message as prompt.
      let promptText = "";
      for (let i = start - 1; i >= 0; i--) {
        if (deduped[i].role === "user" && (deduped[i].text || "").trim().length <= 800) {
          promptText = deduped[i].text;
          break;
        }
      }
      if (!promptText) return null;

      return finalizePair(promptText, answerMerged);
    };

    const scorePair = (pair) => {
      if (!pair) return -Infinity;
      const ans = (pair.answer || "");
      const paras = ans.split(/\n{2,}/).filter((p) => p.trim().length > 0).length;
      const base = paras * 3 + Math.min(10, ans.length / 1200);
      const ov = overlapScore(pair.prompt, ans);
      return base - ov * 10;
    };

    // Try strategy 1: user-anchored forward merge.
    let lastUserIdx = -1;
    for (let i = deduped.length - 1; i >= 0; i--) {
      if (deduped[i].role === "user") {
        lastUserIdx = i;
        break;
      }
    }
    const pairs = [];
    if (lastUserIdx !== -1) pairs.push(buildAnswerFromUserIndex(lastUserIdx));

    // Try strategy 2: assistant-anchored expand (back + forward).
    let lastAssistantIdx = -1;
    for (let i = deduped.length - 1; i >= 0; i--) {
      if (deduped[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
    if (lastAssistantIdx !== -1) pairs.push(buildPairFromAssistantAnchor(lastAssistantIdx));

    // Choose best valid pair.
    let best = null;
    let bestScore = -Infinity;
    for (const p of pairs) {
      const s = scorePair(p);
      if (s > bestScore) {
        bestScore = s;
        best = p;
      }
    }
    if (best) return best;
  }

  // Fallback heuristic: choose best adjacent pair near tail.
  if (deduped.length >= 2) {
    const looksLikeQuestion = (s) => /\?\s*$/.test((s || "").trim()) || (s || "").includes("?");
    let best = null;
    let bestScore = -Infinity;
    const start = Math.max(0, deduped.length - 14);
    for (let i = start; i < deduped.length - 1; i++) {
      const prompt = deduped[i].text;
      const answer = deduped[i + 1].text;
      if (!prompt || !answer) continue;
      if (prompt === answer) continue;

      let score = 0;
      if (looksLikeQuestion(prompt)) score += 4;
      if (answer.length >= Math.max(30, Math.floor(prompt.length * 0.75))) score += 2;
      if (answer.length > prompt.length) score += 1;

      // Penalize likely non-chat metadata.
      if (/@/.test(prompt) && prompt.length < 160) score -= 6;
      if (/@/.test(answer) && answer.length < 160) score -= 6;
      if (/^[\w.+-]+@[\w.-]+\.[a-z]{2,}$/i.test(prompt.trim())) score -= 10;
      if (/^[\w.+-]+@[\w.-]+\.[a-z]{2,}$/i.test(answer.trim())) score -= 10;

      if (score > bestScore) {
        bestScore = score;
        best = { prompt, answer };
      }
    }
    if (best && bestScore >= 2) {
      // Try to find the same answer chunk index in deduped and expand it.
      let bestAnswerIdx = -1;
      const bestAnswerNorm = normalizeForCompare(best.answer);
      for (let i = deduped.length - 1; i >= 0; i--) {
        if (normalizeForCompare(deduped[i].text) === bestAnswerNorm) {
          bestAnswerIdx = i;
          break;
        }
      }
      const expandedAnswer =
        bestAnswerIdx !== -1
          ? expandAssistantAnswerFromIndex(bestAnswerIdx, best.prompt) || best.answer
          : best.answer;
      const pair = finalizePair(best.prompt, expandedAnswer);
      if (pair) return pair;
    }
  }

  // Last-resort textual marker fallback.
  const pageText = root.innerText || document.body.innerText || "";
  const marker = /You said:\s*([\s\S]*?)\n(?:Gemini|Assistant) said:\s*([\s\S]*?)(?=\nYou said:|\n(?:Gemini|Assistant) said:|$)/gi;
  let match;
  let latest = null;
  while ((match = marker.exec(pageText)) !== null) {
    const prompt = (match[1] || "").trim();
    const answer = (match[2] || "").trim();
    if (prompt && answer && !textLooksLikeUiChrome(prompt) && !textLooksLikeUiChrome(answer)) {
      latest = { prompt, answer };
    }
  }
  if (latest) {
    const pair = finalizePair(latest.prompt, latest.answer);
    if (pair) return pair;
  }

  // Final Gemini-specific fallback:
  // use the last likely question line and the nearest following longer block.
  const lines = (pageText || "")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length >= 12 && !textLooksLikeUiChrome(l));
  if (lines.length >= 2) {
    let qIdx = -1;
    for (let i = lines.length - 1; i >= 0; i--) {
      if (/\?/.test(lines[i])) {
        qIdx = i;
        break;
      }
    }
    if (qIdx !== -1) {
      for (let j = qIdx + 1; j < lines.length; j++) {
        if (lines[j].length >= Math.max(24, Math.floor(lines[qIdx].length * 0.7))) {
          const pair = finalizePair(lines[qIdx], lines[j]);
          if (pair) return pair;
        }
      }
    }
  }

  return null;
}

function extractLatestTurnForClaude() {
  const messages = getChatMessagesOrdered();
  const turns = buildTurnsFromMessages(messages);
  if (turns.length > 0) {
    const last = turns[turns.length - 1];
    const prompt = sanitizeCapturedMessageText((last.prompt || "").trim());
    const answer = sanitizeCapturedMessageText((last.answer || "").trim());
    if (prompt && answer && hasMeaningfulConversationPair({ prompt, answer })) {
      return { prompt, answer };
    }
  }
  return null;
}

function extractLatestTurnGeneric() {
  // Very generic heuristic: look for the last two text-rich nodes in common chat containers.
  const containerSelectors = [
    '[role="list"]',
    '[aria-label*="Conversation"]',
    '[data-testid*="conversation"]',
    'main',
    'body',
  ];
  let root = document.body;
  for (const sel of containerSelectors) {
    const el = document.querySelector(sel);
    if (el) {
      root = el;
      break;
    }
  }

  const candidateEls = Array.from(
    root.querySelectorAll(
      "div, p, span, article, section, li"
    )
  ).filter(el => {
    const text = el.innerText?.trim() || "";
    const style = window.getComputedStyle(el);
    return text.length > 10 && text.length < 2000 && style.display !== 'none' && style.visibility !== 'hidden';
  });

  const texts = candidateEls
    .map((el) => el.innerText?.trim() || "")
    .filter((t) => t.length > 15);

  console.log("Generic extraction found", texts.length, "candidate texts");

  if (texts.length >= 2) {
    const prompt = texts[texts.length - 2];
    const answer = texts[texts.length - 1];
    console.log("Using generic pair:", { prompt: prompt.slice(0, 50), answer: answer.slice(0, 50) });
    return { prompt, answer };
  }
  return null;
}

function extractLatestTurnByChrono() {
  // Collect visible text nodes from chat-like wrappers and use the last two.
  const wrappers = Array.from(document.querySelectorAll("[role='listitem'], [data-testid*='message'], [class*='group'], [class*='message'], [aria-label*='message'], div, p, span, article"))
    .filter(el => {
      const text = el.innerText?.trim() || "";
      return text.length > 6 && text.length < 1000; // Reasonable message length
    });
  const texts = wrappers
    .map((el) => el.innerText?.trim() || "")
    .filter((t) => t.length > 6);
  if (texts.length >= 2) {
    return { prompt: texts[texts.length - 2], answer: texts[texts.length - 1] };
  }

  // fallback to page text split by newline and last two long lines
  const lines = (document.body.innerText || "").split("\n").map((l) => l.trim()).filter((l) => l.length > 8);
  if (lines.length >= 2) {
    return { prompt: lines[lines.length - 2], answer: lines[lines.length - 1] };
  }
  return null;
}

function extractLatestTurnForDeepseek() {
  // DeepSeek UI variants change often, so we combine role-aware extraction
  // with robust fallbacks scoped to the active chat area (not the sidebar).
  const root = getDeepseekChatRoot();
  const normalize = (s) =>
    (s || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  const overlap = (a, b) => {
    const aa = normalize(a);
    const bb = normalize(b);
    if (!aa || !bb) return 0;
    if (aa === bb) return 1;
    const short = aa.length <= bb.length ? aa : bb;
    const long = aa.length > bb.length ? aa : bb;
    if (short.length < 20) return 0;
    if (long.includes(short)) return short.length / long.length;
    return 0;
  };
  const isUiNoise = (text) => isDeepseekUiNoise(text);
  const classifyRole = (el) => {
    const hints = [
      el.getAttribute("data-role"),
      el.getAttribute("data-testid"),
      el.getAttribute("data-message-author-role"),
      el.getAttribute("aria-label"),
      el.className,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    if (
      hints.includes("assistant") ||
      hints.includes("model") ||
      hints.includes("deepseek") ||
      hints.includes("bot")
    ) {
      return "assistant";
    }
    if (
      hints.includes("user") ||
      hints.includes("human") ||
      hints.includes("query") ||
      hints.includes("prompt")
    ) {
      return "user";
    }
    return null;
  };

  // Strong path for DeepSeek-like chat layouts:
  // pair the latest explicit user message with the nearest following assistant message.
  const explicitUserSelectors = [
    '[data-role*="user" i]',
    '[data-message-author-role*="user" i]',
    '[data-testid*="user" i]',
    '[class*="UserMessage"]',
    '[class*="user-message"]',
  ];
  const explicitAssistantSelectors = [
    '[data-role*="assistant" i]',
    '[data-role*="model" i]',
    '[data-message-author-role*="assistant" i]',
    '[data-testid*="assistant" i]',
    '[data-testid*="model" i]',
    '[class*="AssistantMessage"]',
    '[class*="assistant-message"]',
    ".ds-markdown",
  ];

  const collectDeepseekNodes = (selector, minLen, maxLen) =>
    Array.from(root.querySelectorAll(selector))
      .filter((el) => !isInsideDeepseekSidebar(el) && !isBiasExtensionUiElement(el))
      .map((el) => getDeepseekTextForMessageEl(el))
      .filter((t) => t.length >= minLen && t.length <= maxLen)
      .filter((t) => !isUiNoise(t) && !isNumericOrUiGarbage(t));

  const userNodes = collectDeepseekNodes(explicitUserSelectors.join(", "), 3, 25000);
  const assistantNodes = collectDeepseekNodes(explicitAssistantSelectors.join(", "), 8, 35000);

  if (userNodes.length > 0 && assistantNodes.length > 0) {
    const promptCandidate = userNodes[userNodes.length - 1];
    // DeepSeek can split one answer across multiple assistant nodes.
    // Build merged candidates from tail windows and keep the richest valid one.
    const uniqueAssistant = [];
    for (const t of assistantNodes) {
      if (!uniqueAssistant.length || normalize(uniqueAssistant[uniqueAssistant.length - 1]) !== normalize(t)) {
        uniqueAssistant.push(t);
      }
    }
    let bestAnswer = null;
    let bestScore = -Infinity;
    for (let end = uniqueAssistant.length - 1; end >= 0; end--) {
      for (let chunks = 1; chunks <= 8; chunks++) {
        const start = end - chunks + 1;
        if (start < 0) break;
        const merged = uniqueAssistant.slice(start, end + 1).join("\n\n").trim();
        if (!merged || merged.length < 20) continue;
        if (overlap(promptCandidate, merged) >= 0.75) continue;
        // Favor complete responses (longer and multi-paragraph).
        const paras = merged.split(/\n{2,}/).filter((p) => p.trim().length > 0).length;
        const score = Math.min(12, Math.floor(merged.length / 180)) + Math.min(5, paras);
        if (score > bestScore) {
          bestScore = score;
          bestAnswer = merged;
        }
      }
    }
    if (bestAnswer) {
      return { prompt: promptCandidate, answer: bestAnswer };
    }
  }

  const blocks = Array.from(
    root.querySelectorAll(
      "[data-role], [data-testid*='message'], [data-testid*='chat'], [data-message-author-role], [role='listitem'], [role='article'], [class*='message'], [class*='chat'], .ds-message, .ds-chat-message, article, section, li, pre"
    )
  )
    .filter((el) => !isInsideDeepseekSidebar(el) && !isBiasExtensionUiElement(el))
    .map((el) => {
      const text = getDeepseekTextForMessageEl(el);
      const role = classifyRole(el) || getDeepseekRoleForMessageEl(el);
      return { text, role };
    })
    .filter((item) => {
      const t = item.text || "";
      if (!t) return false;
      if (t.length < 10 || t.length > 25000) return false;
      return !isUiNoise(t);
    });

  const deduped = [];
  for (const item of blocks) {
    if (!deduped.length || normalize(deduped[deduped.length - 1].text) !== normalize(item.text)) {
      deduped.push(item);
    }
  }

  // Preferred: latest assistant + nearest previous user.
  if (deduped.length >= 2) {
    let assistantIdx = -1;
    for (let i = deduped.length - 1; i >= 0; i--) {
      if (deduped[i].role === "assistant") {
        assistantIdx = i;
        break;
      }
    }
    if (assistantIdx !== -1) {
      for (let i = assistantIdx - 1; i >= 0; i--) {
        if (deduped[i].role !== "user") continue;
        const prompt = (deduped[i].text || "").trim();
        const answerParts = [];
        const isLikelyNextUserTurn = (t) => {
          const s = (t || "").trim();
          if (!s) return false;
          if (s.length > 500) return false;
          if (!s.includes("?")) return false;
          const lower = s.toLowerCase();
          if (lower.startsWith("let") || lower.startsWith("therefore") || lower.startsWith("based on")) {
            return false;
          }
          return true;
        };
        for (let j = assistantIdx; j < deduped.length && answerParts.length < 20; j++) {
          const msg = deduped[j];
          if (j > assistantIdx && msg.role === "user" && isLikelyNextUserTurn(msg.text)) break;
          if (isUiNoise(msg.text)) continue;
          if (msg.role === "user" && isLikelyNextUserTurn(msg.text)) continue;
          const last = answerParts[answerParts.length - 1];
          if (!last || normalize(last) !== normalize(msg.text)) answerParts.push(msg.text);
        }
        const answer = answerParts.join("\n\n").trim();
        if (prompt && answer && overlap(prompt, answer) < 0.75) {
          return { prompt, answer };
        }
      }
    }
  }

  const texts = deduped.map((x) => x.text).filter(Boolean);
  if (texts.length < 2) return null;

  const textLooksLikeUserQuestion = (s) => {
    const trimmed = (s || "").trim();
    if (!trimmed.includes("?")) return false;
    return /\?\s*$/.test(trimmed) || trimmed.slice(trimmed.lastIndexOf("?")).length <= 6;
  };

  // DeepSeek fallback strategy:
  // use the latest likely user question, then merge subsequent non-question
  // blocks as one assistant answer to avoid truncation.
  const looksLikeAssistantStarter = (s) => {
    const t = normalize(s);
    if (!t) return false;
    return (
      t.startsWith("let") ||
      t.startsWith("therefore") ||
      t.startsWith("so ") ||
      t.startsWith("answer:") ||
      t.startsWith("based on") ||
      t.startsWith("here")
    );
  };
  const findLatestQuestionIdx = () => {
    for (let i = texts.length - 1; i >= 0; i--) {
      const t = (texts[i] || "").trim();
      if (t.length < 20) continue;
      // Prefer explicit question prompts over fragments.
      if (!t.includes("?")) continue;
      if (isUiNoise(t)) continue;
      return i;
    }
    return -1;
  };
  const questionIdx = findLatestQuestionIdx();
  if (questionIdx !== -1 && questionIdx + 1 < texts.length) {
    const prompt = texts[questionIdx];
    const answerParts = [];
    for (let i = questionIdx + 1; i < texts.length && answerParts.length < 24; i++) {
      const t = (texts[i] || "").trim();
      if (!t) continue;
      if (isUiNoise(t)) continue;
      // Stop when we hit next clear user-style question turn.
      if (i > questionIdx + 1 && t.includes("?") && t.length < 500 && !looksLikeAssistantStarter(t)) {
        break;
      }
      const last = answerParts[answerParts.length - 1];
      if (!last || normalize(last) !== normalize(t)) {
        answerParts.push(t);
      }
    }
    const mergedAnswer = answerParts.join("\n\n").trim();
    if (mergedAnswer && mergedAnswer.length >= 40 && overlap(prompt, mergedAnswer) < 0.75) {
      return { prompt, answer: mergedAnswer };
    }
  }

  const scorePromptCandidate = (prompt, answer) => {
    const p = (prompt || "").trim();
    const a = (answer || "").trim();
    const pLen = p.length;
    const aLen = a.length;
    let score = 0;
    if (p === a || pLen < 10 || aLen < 10) return -Infinity;
    if (textLooksLikeUserQuestion(p)) score += 5;
    if (aLen >= pLen * 0.7) score += 2;
    if (aLen >= 30) score += 1;
    if (overlap(p, a) >= 0.75) score -= 6;
    return score;
  };

  const tailStart = Math.max(0, texts.length - 12);
  const tail = texts.slice(tailStart);
  let best = null;
  let bestScore = -Infinity;
  for (let localI = tail.length - 1; localI >= 0; localI--) {
    const promptIdx = tailStart + localI;
    const answerIdx = promptIdx + 1;
    if (answerIdx >= texts.length) continue;
    const prompt = texts[promptIdx];
    const answer = texts[answerIdx];
    const s = scorePromptCandidate(prompt, answer);
    if (s > bestScore) {
      bestScore = s;
      best = { prompt, answer };
    }
  }
  if (best) return best;

  // Text-marker fallback from full page text.
  const pageText = root.innerText || document.body.innerText || "";
  const patterns = [
    /You:\s*([\s\S]*?)\n(?:DeepSeek|Assistant):\s*([\s\S]*?)(?=\nYou:|\n(?:DeepSeek|Assistant):|$)/gi,
    /User:\s*([\s\S]*?)\n(?:DeepSeek|Assistant):\s*([\s\S]*?)(?=\nUser:|\n(?:DeepSeek|Assistant):|$)/gi,
    /Human:\s*([\s\S]*?)\n(?:DeepSeek|Assistant):\s*([\s\S]*?)(?=\nHuman:|\n(?:DeepSeek|Assistant):|$)/gi,
  ];
  let latest = null;
  for (const re of patterns) {
    let m;
    while ((m = re.exec(pageText)) !== null) {
      const prompt = (m[1] || "").trim();
      const answer = (m[2] || "").trim();
      if (!prompt || !answer) continue;
      if (isUiNoise(prompt) || isUiNoise(answer)) continue;
      if (overlap(prompt, answer) >= 0.75) continue;
      latest = { prompt, answer };
    }
  }
  if (latest) return latest;

  // Final fallback: last valid adjacent pair.
  for (let i = texts.length - 2; i >= 0; i--) {
    const prompt = texts[i];
    const answer = texts[i + 1];
    if (!prompt || !answer) continue;
    if (prompt.length < 12) continue;
    if (prompt.length < 120 && !/[?]/.test(prompt)) continue;
    if (normalize(prompt) === normalize(answer)) continue;
    if (answer.length < Math.max(20, prompt.length * 0.6)) continue;
    if (overlap(prompt, answer) >= 0.8) continue;
    return { prompt, answer };
  }
  return null;
}

function extractLatestTurnForCurrentSite() {
  const hostname = location.hostname || "";

  console.log("Extracting for hostname:", hostname);

  if (hostname.includes("chatgpt.com") || hostname.includes("openai.com")) {
    const pair = extractLatestTurnForChatGPT();
    if (pair) {
      console.log("ChatGPT pair extracted:", pair);
      return pair;
    }
    const chrono = extractLatestTurnByChrono();
    if (chrono) {
      console.log("ChatGPT chrono fallback:", chrono);
      return chrono;
    }
    return extractLatestTurnGeneric();
  }
  if (hostname.includes("gemini.google.com") || hostname.includes("ai.google.com")) {
    const pair = extractLatestTurnForGemini();
    if (pair) return pair;
    const ordered = pairFromOrderedMessages();
    if (ordered) return ordered;
    return extractLatestTurnByChrono();
  }
  if (hostname.includes("claude.ai")) {
    const pair = extractLatestTurnForClaude();
    if (pair) return pair;
    const ordered = pairFromOrderedMessages();
    if (ordered) return ordered;
    return null;
  }
  if (hostname.includes("deepseek") || hostname.includes("deepthink")) {
    const deepseek = extractLatestTurnForDeepseek();
    if (deepseek && passesDeepseekTurnValidation(deepseek.prompt, deepseek.answer)) {
      return deepseek;
    }
    const ordered = pairFromOrderedMessages();
    if (ordered && passesDeepseekTurnValidation(ordered.prompt, ordered.answer)) {
      return ordered;
    }
    return null;
  }

  return null;
}

function splitPromptContextAndAnswer(prompt) {
  if (!prompt || typeof prompt !== "string") {
    return { question: "", context: "", answerFromPrompt: "" };
  }

  const text = prompt.trim();
  if (!text) return { question: "", context: "", answerFromPrompt: "" };

  // First, try explicit labels used by many prompt templates.
  // Example:
  // Context: ...
  // Question: ...
  const lower = text.toLowerCase();
  const qLabelIdx = lower.lastIndexOf("question:");
  if (qLabelIdx !== -1) {
    const contextPart = text.slice(0, qLabelIdx).trim();
    const questionAndMaybeAfter = text.slice(qLabelIdx + "question:".length).trim();
    const qMarkInQuestionPart = questionAndMaybeAfter.indexOf("?");
    const questionPart =
      qMarkInQuestionPart !== -1
        ? questionAndMaybeAfter.slice(0, qMarkInQuestionPart + 1).trim()
        : questionAndMaybeAfter;
    const answerFromPrompt =
      qMarkInQuestionPart !== -1
        ? questionAndMaybeAfter.slice(qMarkInQuestionPart + 1).trim()
        : "";
    if (questionPart) {
      const cleanedContext = contextPart
        .replace(/^\s*context:\s*/i, "")
        .trim();
      return { question: questionPart, context: cleanedContext, answerFromPrompt };
    }
  }

  // Rule requested:
  // - sentence with "?" => question
  // - before that sentence => context
  // - after that sentence => answerFromPrompt
  // - if no "?" => all goes to question
  const questionMatches = [...text.matchAll(/[^.!?\n]*\?+/g)];
  const match = questionMatches.length > 0 ? questionMatches[questionMatches.length - 1] : null;
  if (match && typeof match.index === "number") {
    const start = match.index;
    const end = start + (match[0] || "").length;
    const question = (match[0] || "").trim();
    const context = text.slice(0, start).trim();
    const answerFromPrompt = text.slice(end).trim();
    if (question) {
      return { question, context, answerFromPrompt };
    }
  }

  return { question: text, context: "", answerFromPrompt: "" };
}

function resolveLatestConversationPair() {
  const isDeepseek = isDeepseekHost();
  const direct = extractLatestTurnForCurrentSite();
  if (direct?.prompt?.trim() && direct?.answer?.trim()) {
    if (!isDeepseek || passesDeepseekTurnValidation(direct.prompt, direct.answer)) {
      return direct;
    }
  }

  const turns = getConversationTurnsPayload();
  if (turns.length === 0) {
    return isDeepseek ? null : direct;
  }

  const latest = turns[turns.length - 1];
  const prompt = (latest.prompt || latest.question || "").trim();
  const answer = (latest.answer || "").trim();
  if (prompt && answer) {
    return { prompt, answer };
  }

  return isDeepseek ? null : direct;
}

function sendNewLlmTurn(payload) {
  chrome.runtime.sendMessage(
    {
      type: "NEW_LLM_TURN",
      payload,
    },
    () => {
      if (chrome.runtime.lastError) {
        setTimeout(() => {
          chrome.runtime.sendMessage({ type: "NEW_LLM_TURN", payload }, () => {
            if (chrome.runtime.lastError) {
              console.warn("Failed to send NEW_LLM_TURN:", chrome.runtime.lastError.message);
            }
          });
        }, 1500);
      }
    }
  );
}

function detectAndSendLatestTurn() {
  try {
    if (!isSupportedLLMHost()) return;

    const pair = resolveLatestConversationPair();
    console.debug("LLM capture pair:", pair);
    if (!hasMeaningfulConversationPair(pair)) {
      console.log("No pair extracted");
      const turns = getConversationTurnsPayload();
      if (turns.length === 0) {
        noteEmptyConversation();
      } else {
        emptyConversationSince = null;
      }
      return;
    }
    emptyConversationSince = null;

    const key = `${pair.prompt.slice(0, 100)}|||${pair.answer.slice(0, 100)}`;
    const now = Date.now();

    const hostname = location.hostname || "";
    const isGemini =
      hostname.includes("gemini.google.com") || hostname.includes("ai.google.com");
    const isDeepseek = isDeepseekHost(hostname);
    const isClaude = hostname.includes("claude.ai");
    const graceMs = isGemini || isDeepseek || isClaude ? 30000 : 12000;
    const minGrowthChars = isGemini || isDeepseek || isClaude ? 250 : 150;
    const normalize = (s) =>
      (s || "")
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim();
    const overlap = (a, b) => normalizePairOverlap(a, b);

    const promptText = (pair.prompt || "").trim();
    const dedupeRepeatedBlocks = (text) => {
      const raw = (text || "").trim();
      if (!raw) return "";
      const norm = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();
      const normLoose = (s) =>
        norm(s)
          .replace(/[.,!?;:()\[\]{}"'\-–—]/g, "")
          .replace(/\s+/g, " ")
          .trim();

      // 1) Remove repeated paragraph blocks while preserving order.
      const paras = raw
        .split(/\n{2,}/)
        .map((p) => p.trim())
        .filter(Boolean);
      const paraSeen = new Set();
      const paraOut = [];
      for (const p of paras) {
        const k = norm(p);
        if (paraSeen.has(k)) continue;
        paraSeen.add(k);
        paraOut.push(p);
      }
      let cleaned = paraOut.join("\n\n");

      // 2) If the full response is accidentally duplicated (A + A), keep one copy.
      const n = cleaned.length;
      if (n >= 80) {
        const half = Math.floor(n / 2);
        const a = cleaned.slice(0, half).trim();
        const b = cleaned.slice(half).trim();
        if (a && b && norm(a) === norm(b)) {
          cleaned = a;
        }
      }

      // 3) Remove duplicated/fragment lines caused by nested DOM nodes.
      const lines = cleaned
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
      const seen = new Set();
      const seenLoose = new Set();
      const out = [];
      for (const line of lines) {
        const key = norm(line);
        const keyLoose = normLoose(line);
        if (!key) continue;
        if (seen.has(key)) continue;
        if (keyLoose && seenLoose.has(keyLoose)) continue;

        // Drop short fragment line if it's contained by an already-kept longer line.
        if (key.length >= 8 && key.length <= 110) {
          const containedByLonger = out.some((prev) => {
            const prevKey = norm(prev);
            const prevLoose = normLoose(prev);
            return (
              (prevKey.length > key.length + 12 && prevKey.includes(key)) ||
              (prevLoose.length > keyLoose.length + 12 && prevLoose.includes(keyLoose))
            );
          });
          if (containedByLonger) continue;
        }

        // If this is a fuller line that contains a previous short fragment, replace it.
        let replaced = false;
        for (let i = out.length - 1; i >= 0; i--) {
          const prevKey = norm(out[i]);
          const prevLoose = normLoose(out[i]);
          if (
            prevKey.length >= 8 &&
            prevKey.length <= 110 &&
            (
              (key.length > prevKey.length + 12 && key.includes(prevKey)) ||
              (keyLoose.length > prevLoose.length + 12 && keyLoose.includes(prevLoose))
            )
          ) {
            seen.delete(prevKey);
            seenLoose.delete(prevLoose);
            out.splice(i, 1);
          }
        }

        seen.add(key);
        if (keyLoose) seenLoose.add(keyLoose);
        out.push(line);
      }

      // 4) Reduce accidental duplicated tails: keep first occurrence order.
      let compact = out
        .filter((l, idx, arr) => !(idx > 0 && norm(l) === norm(arr[idx - 1])))
        .join("\n")
        .trim();

      // 5) Trim repeated tail when the second half replays the first with minor variants.
      const paraList = compact
        .split(/\n{2,}/)
        .map((p) => p.trim())
        .filter(Boolean);
      if (paraList.length >= 4) {
        let cutAt = -1;
        for (let i = 1; i < paraList.length; i++) {
          const curr = normLoose(paraList[i]);
          if (!curr || curr.length < 24) continue;
          const seenBefore = paraList.slice(0, i).some((p) => {
            const prev = normLoose(p);
            return prev === curr || prev.includes(curr) || curr.includes(prev);
          });
          if (seenBefore) {
            cutAt = i;
            break;
          }
        }
        if (cutAt !== -1) {
          compact = paraList.slice(0, cutAt).join("\n\n").trim();
        }
      }
      return compact;
    };
    const answerText = dedupeRepeatedBlocks((pair.answer || "").trim());
    if (!promptText || !answerText) {
      console.log("Skipping incomplete pair");
      return;
    }

    if (isDeepseek && !passesDeepseekTurnValidation(promptText, answerText)) {
      console.log("Skipping invalid DeepSeek pair");
      return;
    }

    const tlPrompt = normalize(promptText);
    const tlAnswer = normalize(answerText);
    if (isClaude) {
      if (
        tlPrompt.includes("sonnet") ||
        tlPrompt.includes("haiku") ||
        tlPrompt.includes("opus") ||
        tlPrompt.includes("fable") ||
        tlPrompt === "claude" ||
        tlPrompt === "claude 3" ||
        tlPrompt === "claude 3.5" ||
        tlPrompt === "claude 4" ||
        tlPrompt === "sonnet 4.5"
      ) {
        console.log("Skipping Claude model label as prompt");
        return;
      }
      if (
        tlAnswer.includes("claude is ai and can make mistakes") ||
        tlAnswer.includes("please double-check responses") ||
        tlAnswer.includes("currently unavailable") ||
        tlAnswer.includes("traffic light") ||
        tlAnswer.includes("embedder=") ||
        tlAnswer.includes("sentence_transformers")
      ) {
        console.log("Skipping Claude disclaimer/UI noise as answer");
        return;
      }
    }
    const paOverlap = overlap(promptText, answerText);
    if (paOverlap >= 0.82) {
      console.log("Skipping pair with high prompt-answer overlap");
      return;
    }
    if (
      isClaude &&
      looksQuestionLikeText(answerText) &&
      answerText.length <= promptText.length * 1.05
    ) {
      console.log("Skipping prompt-like answer candidate");
      return;
    }

    // Allow re-sending same turn after a short grace period (for popup reopen / recovery)
    if (key === lastSentTurnKey && now - lastSentTurnTimestamp < graceMs) {
      const newLen = (pair.answer || "").length;
      const grewEnough = newLen > lastSentTurnAnswerLen + minGrowthChars;

      // During Gemini streaming, the first 100 chars often stay the same for a while.
      // If the answer is clearly growing, allow re-send so we capture the full turn.
      if (!grewEnough) {
        console.log("Duplicate turn recently seen, skipping");
        return;
      }
    }

    lastSentTurnKey = key;
    lastSentTurnTimestamp = now;
    lastSentTurnAnswerLen = (pair.answer || "").length;

    const parsed = splitPromptContextAndAnswer(pair.prompt || "");
    const finalAnswer = (pair.answer || "").trim() || parsed.answerFromPrompt || "";
    console.debug("LLM parsed prompt/question/context/answer:", parsed);

    console.log("Sending NEW_LLM_TURN with:", {
      question: parsed.question,
      prompt: pair.prompt,
      context: parsed.context || "",
      answer: finalAnswer,
    });

    sendNewLlmTurn({
      question: parsed.question,
      prompt: pair.prompt,
      context: parsed.context || "",
      answer: finalAnswer,
      mode_speed: "fast",
      mode_depth: "normal",
    });
  } catch (err) {
    console.warn("Failed to auto-capture LLM turn:", err);
  }
}

function startAutoCapture() {
  if (!isSupportedLLMHost()) return;

  installRouteChangeListeners();

  const observer = new MutationObserver(() => {
    scheduleAutoDetect();
  });
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  AUTO_DETECT_LOAD_DELAYS_MS.forEach((delay) => {
    setTimeout(() => detectAndSendLatestTurn(), delay);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      scheduleAutoDetect(300);
    }
  });

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      resetCaptureStateForNavigation();
    }
    scheduleAutoDetect(300);
  });

  window.addEventListener("focus", () => {
    scheduleAutoDetect(500);
  });
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", startAutoCapture);
} else {
  startAutoCapture();
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PING_CONTENT_SCRIPT") {
    sendResponse({ ok: true });
  } else if (message.type === "GET_PAGE_TEXT") {
    const text = extractVisibleText();
    sendResponse({ text });
  } else if (message.type === "GET_SELECTION_TEXT") {
    const payload = extractSelectionPayload();
    sendResponse(payload);
  } else if (message.type === "GET_CONVERSATION_TURNS") {
    sendResponse({ ok: true, turns: getConversationTurnsPayload() });
  } else if (message.type === "HIGHLIGHT_SENTENCES") {
    highlightSentences(message.sentences || []);
  } else if (message.type === "BIAS_RESULT_FOR_LLM_TURN") {
    const payload = message.payload || {};
    if (!payload.result) return;
    if (window.BiasOverlayHub) {
      window.BiasOverlayHub.update(payload);
    }
  } else if (message.type === "LLM_TURN_CLEARED") {
    lastSentTurnKey = null;
    lastSentTurnTimestamp = 0;
    lastSentTurnAnswerLen = 0;
    emptyConversationSince = null;
    if (window.BiasOverlayHub?.clear) {
      window.BiasOverlayHub.clear();
    }
    highlightSentences([]);
    sendResponse({ ok: true });
  } else if (message.type === "UPDATE_OVERLAY_PREFS") {
    if (window.BiasOverlayHub && message.prefs) {
      window.BiasOverlayHub.setPrefs(message.prefs, !!message.persist);
    }
    sendResponse({ ok: true, prefs: window.BiasOverlayHub?.getPrefs?.() });
    return true;
  } else if (message.type === "GET_OVERLAY_PREFS") {
    sendResponse({ ok: true, prefs: window.BiasOverlayHub?.getPrefs?.() });
    return true;
  }
});

if (window.BiasOverlayHub) {
  window.BiasOverlayHub.loadPrefs();
}

