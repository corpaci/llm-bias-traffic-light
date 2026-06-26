const scanButton = document.getElementById("scanButton");
const turnPickerGroup = document.getElementById("turnPickerGroup");
const turnPickerEl = document.getElementById("turnPicker");
const scanTurnButton = document.getElementById("scanTurnButton");
const statusEl = document.getElementById("status");
const biasLevelEl = document.getElementById("riskLevel");
const scorePercentEl = document.getElementById("scorePercent");
const explanationEl = document.getElementById("explanation");
const sentencesListEl = document.getElementById("sentencesList");
const biasRadarCanvas = document.getElementById("biasRadar");
const biasRadarHintEl = document.getElementById("biasRadarHint");
const lastPromptEl = document.getElementById("lastPrompt");
const lastContextEl = document.getElementById("lastContext");
const lastAnswerEl = document.getElementById("lastAnswer");
const modeDepthEl = document.getElementById("modeDepth");
const biasTypesEl = document.getElementById("biasTypes");
const overlayEnabledEl = document.getElementById("overlayEnabled");
const overlayModuleInputs = Array.from(
  document.querySelectorAll("[data-overlay-module]")
);
const appHeaderEl = document.getElementById("appHeader");
const qaCardEl = document.getElementById("qaCard");
const resultCardEl = document.getElementById("resultCard");
const chartCardEl = document.getElementById("chartCard");
const sentencesSectionEl = document.getElementById("sentencesSection");
const interactionPlaceholderEl = document.getElementById("interactionPlaceholder");
const advancedSettingsToggle = document.getElementById("advancedSettingsToggle");
const advancedSettingsContent = document.getElementById("advancedSettingsContent");
const advancedSettingsAccordion = document.querySelector(".settings-accordion");

let conversationTurns = [];

const OVERLAY_STORAGE_KEY = "biasOverlayPrefs";
const DEFAULT_OVERLAY_PREFS = {
  enabled: true,
  modules: {
    score: true,
    explanation: true,
    flagged: false,
    spider: true,
  },
};

const DEFAULT_BIAS_TYPES = ["gender"];

function getSelectedBiasTypes() {
  if (!biasTypesEl) return DEFAULT_BIAS_TYPES;
  const selected = Array.from(biasTypesEl.selectedOptions)
    .map((option) => option.value)
    .filter(Boolean);
  return selected.length > 0 ? selected : DEFAULT_BIAS_TYPES;
}

function saveBiasTypeSelection() {
  if (!chrome?.storage?.local) return;
  chrome.storage.local.set({ bias_types: getSelectedBiasTypes() });
}

function loadBiasTypeSelection() {
  if (!biasTypesEl || !chrome?.storage?.local) return;
  chrome.storage.local.get({ bias_types: DEFAULT_BIAS_TYPES }, (result) => {
    const selected = Array.isArray(result?.bias_types) && result.bias_types.length > 0
      ? result.bias_types
      : DEFAULT_BIAS_TYPES;
    for (const option of biasTypesEl.options) {
      option.selected = selected.includes(option.value);
    }
  });
}

// ============================================================================
// UI STATE MANAGEMENT
// ============================================================================

const uiState = {
  hasInteractionData: false,
  hasRunScan: false,
  hasSupportedSite: false,
};

function updateUIVisibility() {
  // Show/hide placeholder based on whether we have interaction data
  if (interactionPlaceholderEl) {
    interactionPlaceholderEl.classList.toggle("hidden", uiState.hasInteractionData);
  }
  if (qaCardEl) {
    qaCardEl.classList.toggle("hidden", !uiState.hasInteractionData);
  }

  if (resultCardEl) {
    resultCardEl.classList.toggle("hidden", !uiState.hasRunScan);
  }

  if (turnPickerGroup) {
    turnPickerGroup.classList.toggle("hidden", !uiState.hasSupportedSite);
  }
}

function updateHeaderBackground(biasScore) {
  if (!appHeaderEl) return;
  
  // Remove all risk classes
  appHeaderEl.classList.remove("risk-low", "risk-medium", "risk-high");
  
  // If no score yet, keep neutral (no class)
  if (biasScore === null || biasScore === undefined) {
    return;
  }

  const score = typeof biasScore === "number" ? biasScore : 0;
  if (score < 0.25) {
    appHeaderEl.classList.add("risk-low");
  } else if (score < 0.55) {
    appHeaderEl.classList.add("risk-medium");
  } else {
    appHeaderEl.classList.add("risk-high");
  }
}

// ============================================================================
// ACCORDION CONTROL
// ============================================================================

if (advancedSettingsToggle) {
  advancedSettingsToggle.addEventListener("click", () => {
    if (advancedSettingsAccordion) {
      advancedSettingsAccordion.classList.toggle("open");
    }
  });
}

// ============================================================================
// SITE DETECTION & INITIALIZATION
// ============================================================================

// Check if we're on a supported AI website
const isAllowedSite = async () => {
  return new Promise((resolve) => {
    try {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (!tabs || tabs.length === 0) {
          resolve(false);
          return;
        }
        const url = tabs[0].url || "";
        const supportedDomains = [
          "chat.openai.com",
          "chatgpt.com",
          "gemini.google.com",
          "ai.google.com",
          "claude.ai",
          "chat.deepseek.com",
          "deepseek.com",
          "www.deepseek.com"
        ];
        const isAllowed = supportedDomains.some(domain => url.includes(domain));
        resolve(isAllowed);
      });
    } catch (err) {
      console.error("Error checking site:", err);
      resolve(false);
    }
  });
};

const initializeForCurrentSite = async () => {
  const allowed = await isAllowedSite();
  uiState.hasSupportedSite = !!allowed;
  if (!allowed) {
    // Disable all controls on non-AI websites
    [scanButton, scanTurnButton].forEach(btn => {
      if (btn) btn.disabled = true;
    });
    if (modeDepthEl) modeDepthEl.disabled = true;
    if (biasTypesEl) biasTypesEl.disabled = true;
    
    setStatus(
      "⚠️ Bias Detector only works on supported AI chat platforms: " +
      "ChatGPT, Gemini, Claude, and DeepSeek."
    );
    updateUIVisibility();
    return false;
  }
  updateUIVisibility();
  loadConversationTurns();
  return true;
};

// Initialize on load
document.addEventListener("DOMContentLoaded", initializeForCurrentSite);
if (document.readyState === "loading") {
  // Already handled by DOMContentLoaded
} else {
  initializeForCurrentSite();
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function setStatus(text) {
  if (statusEl) {
    statusEl.textContent = text;
  }
}

function toUserErrorMessage(err) {
  const raw = err?.message || String(err || "Unknown error");
  const msg = raw.toLowerCase();
  if (
    msg.includes("receiving end does not exist") ||
    msg.includes("could not establish connection")
  ) {
    return "Could not connect to page script. Refresh the LLM tab once, then retry.";
  }
  return raw;
}

const SUPPORTED_DOMAINS = [
  "chat.openai.com",
  "chatgpt.com",
  "gemini.google.com",
  "ai.google.com",
  "claude.ai",
  "chat.deepseek.com",
  "deepseek.com",
  "www.deepseek.com",
];

async function getActiveTab() {
  return new Promise((resolve, reject) => {
    try {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs?.[0];
        if (!tab || !tab.id) {
          reject(new Error("No active tab found."));
          return;
        }
        resolve(tab);
      });
    } catch (err) {
      reject(err);
    }
  });
}

function isSupportedUrl(url = "") {
  return SUPPORTED_DOMAINS.some((domain) => url.includes(domain));
}

async function ensureContentScriptReady(tab) {
  if (!tab || !tab.id) throw new Error("No active tab found.");

  const ping = () =>
    new Promise((resolve, reject) => {
      chrome.tabs.sendMessage(tab.id, { type: "PING_CONTENT_SCRIPT" }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve(response);
      });
    });

  try {
    const resp = await ping();
    if (resp?.ok) return;
  } catch (err) {
    // Try one-time reinjection for supported sites.
  }

  if (!isSupportedUrl(tab.url || "")) {
    throw new Error("This page is not a supported LLM website.");
  }

  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["contentScript.js"],
  });

  const respAfterInject = await ping();
  if (!respAfterInject?.ok) {
    throw new Error("Could not connect to the page script. Please refresh the tab once.");
  }
}

async function sendMessageToActiveTab(message) {
  const tab = await getActiveTab();
  await ensureContentScriptReady(tab);
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tab.id, message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(response);
    });
  });
}

function setLoading(isLoading) {
  scanButton.disabled = isLoading;
  scanButton.textContent = isLoading ? "Scanning…" : "Scan page";
  if (scanTurnButton) {
    scanTurnButton.disabled = isLoading || conversationTurns.length === 0;
    scanTurnButton.textContent = isLoading ? "Scanning…" : "Scan conversation";
  }
}

function setLastInteraction(question, context, answer) {
  uiState.hasInteractionData = !!(question || context || answer);
  
  if (lastPromptEl) {
    lastPromptEl.textContent = question || "Prompt not detected.";
    lastPromptEl.classList.toggle("muted", !question);
  }
  if (lastContextEl) {
    lastContextEl.textContent = context || "Context not detected.";
    lastContextEl.classList.toggle("muted", !context);
  }
  if (lastAnswerEl) {
    lastAnswerEl.textContent = answer || "Answer not detected.";
    lastAnswerEl.classList.toggle("muted", !answer);
  }
  
  updateUIVisibility();
}

function clearPluginState() {
  uiState.hasRunScan = false;
  uiState.hasInteractionData = false;

  updateHeaderBackground(null);

  if (biasLevelEl) {
    biasLevelEl.textContent = "–";
    biasLevelEl.className = "badge badge-neutral";
  }
  if (scorePercentEl) scorePercentEl.textContent = "–";
  if (explanationEl) {
    explanationEl.textContent = "Run a scan to see analysis.";
    explanationEl.classList.add("muted");
  }
  if (biasRadarHintEl) {
    biasRadarHintEl.textContent = "";
    biasRadarHintEl.classList.add("hidden");
  }
  drawRadarChart({ scores: {} });
  if (sentencesListEl) sentencesListEl.innerHTML = "";

  if (lastPromptEl) {
    lastPromptEl.textContent = "Waiting for a new prompt…";
    lastPromptEl.classList.add("muted");
  }
  if (lastContextEl) {
    lastContextEl.textContent = "Waiting for page context…";
    lastContextEl.classList.add("muted");
  }
  if (lastAnswerEl) {
    lastAnswerEl.textContent = "Waiting for a new answer…";
    lastAnswerEl.classList.add("muted");
  }

  updateUIVisibility();
  highlightSentencesOnPage([]);
}

function buildTurnMatchKey({ prompt, question, answer } = {}) {
  const promptText = prompt || question || "";
  const answerText = answer || "";
  return `${promptText.slice(0, 100)}|||${answerText.slice(0, 100)}`;
}

function turnMatchesStored(turn, stored) {
  if (!turn || !stored) return false;
  return buildTurnMatchKey(turn) === buildTurnMatchKey(stored);
}

function applyStoredTurn(stored) {
  const displayPrompt = stored.question || stored.prompt || "";
  const context = stored.context || "";
  const answer = stored.answer || "";
  setLastInteraction(displayPrompt, context, answer);
  if (stored.result) {
    renderResult(stored.result);
  }
}

function requestClearStoredTurn() {
  if (!chrome?.runtime?.sendMessage) return;
  chrome.runtime.sendMessage({ type: "CLEAR_LLM_TURN" }, () => {
    if (chrome.runtime.lastError) {
      // Popup may reload while the service worker restarts.
    }
  });
}

function restoreLastTurnIfRelevant(turns) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "GET_LAST_LLM_TURN" }, (response) => {
      const stored = response?.data;
      if (!stored?.result) {
        resolve();
        return;
      }

      const turnList = Array.isArray(turns) ? turns : [];

      if (turnList.length === 0) {
        clearPluginState();
        requestClearStoredTurn();
        resolve();
        return;
      }

      const latestTurn = turnList[turnList.length - 1];
      if (turnMatchesStored(latestTurn, stored)) {
        applyStoredTurn(stored);
        setStatus("Restored latest automatic LLM turn.");
      } else {
        clearPluginState();
        previewSelectedTurn();
      }

      resolve();
    });
  });
}

function drawRadarChart({ scores }) {
  if (!biasRadarCanvas) return;
  const ctx = biasRadarCanvas.getContext("2d");
  if (!ctx) return;

  const labels = Object.keys(scores || {});
  const values = labels.map((k) => {
    const v = scores[k];
    const num = typeof v === "number" && Number.isFinite(v) ? v : 0;
    return Math.max(0, Math.min(1, num));
  });

  const w = biasRadarCanvas.width;
  const h = biasRadarCanvas.height;
  ctx.clearRect(0, 0, w, h);

  if (labels.length < 1) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.55)";
    ctx.font = "12px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.fillText("No breakdown available.", 12, 22);
    return;
  }

  const cx = w / 2;
  const cy = h / 2 + 6;
  const radius = Math.min(w, h) * 0.34;

  const gridColor = "rgba(148, 163, 184, 0.22)";
  const axisColor = "rgba(148, 163, 184, 0.26)";
  const labelColor = "rgba(226, 232, 240, 0.92)";
  const polyFill = "rgba(34, 197, 94, 0.18)";
  const polyStroke = "rgba(34, 197, 94, 0.85)";
  const pointFill = "rgba(34, 197, 94, 0.95)";

  const n = labels.length;
  const angle0 = -Math.PI / 2;

  const polar = (i, r) => {
    const a = angle0 + (2 * Math.PI * i) / n;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  };

  for (const t of [0.25, 0.5, 0.75, 1.0]) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = polar(i, radius * t);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.font = "11px system-ui, -apple-system, Segoe UI, sans-serif";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const p = polar(i, radius);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(p.x, p.y);
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    const lp = polar(i, radius + 18);
    const label = labels[i];
    ctx.fillStyle = labelColor;
    const align = lp.x < cx - 6 ? "right" : lp.x > cx + 6 ? "left" : "center";
    ctx.textAlign = align;
    ctx.fillText(label, lp.x, lp.y);
  }

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const p = polar(i, radius * values[i]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
  ctx.fillStyle = polyFill;
  ctx.fill();
  ctx.strokeStyle = polyStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  for (let i = 0; i < n; i++) {
    const p = polar(i, radius * values[i]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = pointFill;
    ctx.fill();
  }
}

function readOverlayPrefsFromUi() {
  const modules = { ...DEFAULT_OVERLAY_PREFS.modules };
  overlayModuleInputs.forEach((input) => {
    const moduleId = input.dataset.overlayModule;
    if (moduleId) modules[moduleId] = !!input.checked;
  });
  return {
    enabled: overlayEnabledEl ? !!overlayEnabledEl.checked : DEFAULT_OVERLAY_PREFS.enabled,
    modules,
  };
}

function applyOverlayPrefsToUi(prefs) {
  const next = {
    enabled:
      typeof prefs?.enabled === "boolean" ? prefs.enabled : DEFAULT_OVERLAY_PREFS.enabled,
    modules: { ...DEFAULT_OVERLAY_PREFS.modules, ...(prefs?.modules || {}) },
  };
  if (overlayEnabledEl) overlayEnabledEl.checked = next.enabled;
  setOverlayModuleInputsDisabled(!next.enabled);
  overlayModuleInputs.forEach((input) => {
    const moduleId = input.dataset.overlayModule;
    if (moduleId && Object.prototype.hasOwnProperty.call(next.modules, moduleId)) {
      input.checked = !!next.modules[moduleId];
    }
  });
  return next;
}

function persistOverlayPrefs(prefs) {
  chrome.storage.local.set({ [OVERLAY_STORAGE_KEY]: prefs });
}

async function syncOverlayPrefsToActiveTab(prefs, persist = true) {
  if (persist) persistOverlayPrefs(prefs);
  try {
    const tab = await getActiveTab();
    await ensureContentScriptReady(tab);
    chrome.tabs.sendMessage(tab.id, {
      type: "UPDATE_OVERLAY_PREFS",
      prefs,
      persist: false,
    });
  } catch (err) {
    // Best-effort when tab is unavailable.
    console.warn("Could not sync overlay prefs to tab:", err);
  }
}

function loadOverlayPrefs() {
  chrome.storage.local.get(OVERLAY_STORAGE_KEY, (data) => {
    const stored = data?.[OVERLAY_STORAGE_KEY];
    const prefs = applyOverlayPrefsToUi(stored || DEFAULT_OVERLAY_PREFS);
    syncOverlayPrefsToActiveTab(prefs, false);
  });
}

function setOverlayModuleInputsDisabled(disabled) {
  overlayModuleInputs.forEach((input) => {
    input.disabled = disabled;
  });
}

function bindOverlayControls() {
  if (overlayEnabledEl) {
    overlayEnabledEl.addEventListener("change", () => {
      setOverlayModuleInputsDisabled(!overlayEnabledEl.checked);
      const prefs = readOverlayPrefsFromUi();
      syncOverlayPrefsToActiveTab(prefs, true);
    });
  }

  overlayModuleInputs.forEach((input) => {
    input.addEventListener("change", () => {
      const prefs = readOverlayPrefsFromUi();
      syncOverlayPrefsToActiveTab(prefs, true);
    });
  });
}

// ============================================================================
// RESULT RENDERING
// ============================================================================

function renderResult(result) {
  const { bias_score, biased_sentences, explanation } = result;
  
  uiState.hasRunScan = true;

  // Numeric score
  const numericScore = typeof bias_score === "number" ? bias_score : null;
  
  // Update header background based on risk
  updateHeaderBackground(numericScore);

  // Qualitative risk level badge and merged display
  let levelText = "–";
  let levelClass = "badge badge-neutral";
  if (numericScore != null) {
    if (numericScore < 0.25) {
      levelText = "Low";
      levelClass = "badge badge-low";
    } else if (numericScore < 0.55) {
      levelText = "Medium";
      levelClass = "badge badge-medium";
    } else {
      levelText = "High";
      levelClass = "badge badge-high";
    }
  }
  
  biasLevelEl.textContent = levelText;
  biasLevelEl.className = levelClass;
  
  if (scorePercentEl) {
    scorePercentEl.textContent = numericScore != null ? `(${(numericScore * 100).toFixed(0)}%)` : "–";
  }

  let explanationText = explanation || "";
  const similarityParts = [];
  if (typeof result.similarity_question_answer === "number") {
    similarityParts.push(
      `Question↔Answer sim: ${result.similarity_question_answer.toFixed(3)}`
    );
  } else if (typeof result.similarity_prompt_answer === "number") {
    similarityParts.push(
      `Prompt↔Answer sim: ${result.similarity_prompt_answer.toFixed(3)}`
    );
  }
  if (typeof result.similarity_context_answer === "number") {
    similarityParts.push(
      `Context↔Answer sim: ${result.similarity_context_answer.toFixed(3)}`
    );
  }
  if (typeof result.similarity_question_context === "number") {
    similarityParts.push(
      `Question↔Context sim: ${result.similarity_question_context.toFixed(3)}`
    );
  } else if (typeof result.similarity_prompt_context === "number") {
    similarityParts.push(
      `Prompt↔Context sim: ${result.similarity_prompt_context.toFixed(3)}`
    );
  }
  if (similarityParts.length > 0) {
    explanationText += `\n${similarityParts.join(" · ")}`;
  }
  explanationEl.textContent = explanationText;

  // Radar chart (per-bias breakdown)
  const biasTypeScores = result?.bias_type_scores;
  if (biasRadarHintEl) {
    const hasBreakdown = biasTypeScores && typeof biasTypeScores === "object";
    biasRadarHintEl.textContent = hasBreakdown ? "" : "No breakdown returned yet.";
    biasRadarHintEl.classList.toggle("hidden", hasBreakdown);
  }
  if (biasTypeScores && typeof biasTypeScores === "object") {
    drawRadarChart({ scores: biasTypeScores });
  } else {
    drawRadarChart({ scores: {} });
  }

  sentencesListEl.innerHTML = "";
  if (!biased_sentences || biased_sentences.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No potentially biased sentences detected by prototype.";
    li.classList.add("muted");
    sentencesListEl.appendChild(li);
  } else {
    biased_sentences.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      sentencesListEl.appendChild(li);
    });
  }

  // Update UI visibility
  updateUIVisibility();

  // Ask the content script to visually highlight biased sentences on the page
  if (Array.isArray(biased_sentences) && biased_sentences.length > 0) {
    highlightSentencesOnPage(biased_sentences);
  } else {
    // Clear any previous highlights
    highlightSentencesOnPage([]);
  }

  // Also update modular on-page overlay boxes (bottom-right).
  try {
    getActiveTab()
      .then((tab) => ensureContentScriptReady(tab).then(() => tab))
      .then((tab) => {
        chrome.tabs.sendMessage(tab.id, {
          type: "BIAS_RESULT_FOR_LLM_TURN",
          payload: { result, bias_type_scores: result.bias_type_scores },
        });
      })
      .catch(() => {
        // Best-effort visual step, ignore errors.
      });
  } catch (err) {
    console.error("Failed to send overlay update:", err);
  }
}

function highlightSentencesOnPage(sentences) {
  try {
    getActiveTab()
      .then((tab) => ensureContentScriptReady(tab).then(() => tab))
      .then((tab) => {
        chrome.tabs.sendMessage(tab.id, {
          type: "HIGHLIGHT_SENTENCES",
          sentences,
        });
      })
      .catch(() => {
        // Best-effort visual step, ignore errors.
      });
  } catch (err) {
    console.error("Failed to send highlight request:", err);
  }
}

async function getActiveTabText() {
  const response = await sendMessageToActiveTab({ type: "GET_PAGE_TEXT" });
  return response?.text || "";
}

async function getActiveSelectionText() {
  const response = await sendMessageToActiveTab({ type: "GET_SELECTION_TEXT" });
  return (
    response?.text ||
    response?.expandedText ||
    response?.selectionText ||
    ""
  );
}

async function getConversationTurns() {
  const response = await sendMessageToActiveTab({ type: "GET_CONVERSATION_TURNS" });
  return response?.turns || [];
}

function populateTurnPicker(turns) {
  conversationTurns = Array.isArray(turns) ? turns : [];
  if (!turnPickerEl) return;

  turnPickerEl.innerHTML = "";
  if (conversationTurns.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No conversations found on this page";
    turnPickerEl.appendChild(opt);
    turnPickerEl.disabled = true;
    if (scanTurnButton) scanTurnButton.disabled = true;
    return;
  }

  conversationTurns.forEach((turn, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    const label = turn.label || turn.question || turn.prompt || "Conversation";
    opt.textContent = `#${idx + 1} · ${label}`;
    turnPickerEl.appendChild(opt);
  });

  turnPickerEl.disabled = false;
  turnPickerEl.value = String(conversationTurns.length - 1);
  if (scanTurnButton) scanTurnButton.disabled = false;
  previewSelectedTurn();
}

async function loadConversationTurns() {
  if (!uiState.hasSupportedSite) return;
  try {
    const turns = await getConversationTurns();
    populateTurnPicker(turns);
    await restoreLastTurnIfRelevant(turns);
  } catch (err) {
    populateTurnPicker([]);
    clearPluginState();
    setStatus(`Could not load conversations: ${toUserErrorMessage(err)}`);
  }
}

// ============================================================================
// TEXT PROCESSING UTILITIES
// ============================================================================

function splitPromptContextAndAnswer(prompt) {
  if (!prompt || typeof prompt !== "string") {
    return { question: "", context: "", answerFromPrompt: "" };
  }

  const text = prompt.trim();
  if (!text) return { question: "", context: "", answerFromPrompt: "" };

  // Try explicit labels first: "Context: ... Question: ..."
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
      const cleanedContext = contextPart.replace(/^\s*context:\s*/i, "").trim();
      return { question: questionPart, context: cleanedContext, answerFromPrompt };
    }
  }

  // Rule: last sentence containing "?" is the question; before it is context.
  // If no "?" exists, treat all as question.
  const questionMatches = [...text.matchAll(/[^.!?\n]*\?+/g)];
  const match = questionMatches.length > 0 ? questionMatches[questionMatches.length - 1] : null;
  if (match && typeof match.index === "number") {
    const start = match.index;
    const end = start + (match[0] || "").length;
    const question = (match[0] || "").trim();
    const context = text.slice(0, start).trim();
    const answerFromPrompt = text.slice(end).trim();
    if (question) return { question, context, answerFromPrompt };
  }

  return { question: text, context: "", answerFromPrompt: "" };
}

async function analyzeText({ text, question, prompt, context, answer, mode_depth }) {
  return new Promise((resolve, reject) => {
    const bias_types = getSelectedBiasTypes();
    chrome.runtime.sendMessage(
      {
        type: "ANALYZE_TEXT",
        payload: { text, question, prompt, context, answer, mode_depth, bias_types },
      },
      (response) => {
        if (!response) {
          reject(new Error("No response from background script."));
          return;
        }
        if (!response.ok) {
          reject(new Error(response.error || "Unknown error from API."));
          return;
        }
        resolve(response.data);
      }
    );
  });
}

// ============================================================================
// EVENT LISTENERS - SCAN BUTTON
// ============================================================================

scanButton.addEventListener("click", async () => {
  setLoading(true);
  setStatus("Collecting text from page…");

  try {
    const text = await getActiveTabText();
    if (!text || text.trim().length === 0) {
      setStatus("No readable text found on this page.");
      setLoading(false);
      return;
    }

    const mode_depth = modeDepthEl?.value || "normal";
    setStatus("Sending page text to bias API…");
    const result = await analyzeText({
      text,
      context: text,
      mode_depth,
    });

    renderResult(result);
    setStatus("Scan complete.");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${toUserErrorMessage(err)}`);
  } finally {
    setLoading(false);
  }
});

// ============================================================================
// EVENT LISTENERS - CONVERSATION PICKER
// ============================================================================

function previewSelectedTurn() {
  const idx = Number.parseInt(turnPickerEl?.value ?? "", 10);
  const turn = Number.isFinite(idx) ? conversationTurns[idx] : null;
  if (!turn) return;
  setLastInteraction(
    turn.question || turn.prompt || "Prompt not detected.",
    turn.context || "Context not detected.",
    turn.answer || "Answer not detected."
  );
}

turnPickerEl?.addEventListener("change", () => {
  previewSelectedTurn();
});

scanTurnButton?.addEventListener("click", async () => {
  const idx = Number.parseInt(turnPickerEl?.value ?? "", 10);
  const turn = Number.isFinite(idx) ? conversationTurns[idx] : null;
  if (!turn) {
    setStatus("Choose a conversation from the list first.");
    return;
  }

  setLoading(true);
  setStatus("Scanning selected conversation…");

  try {
    previewSelectedTurn();

    const mode_depth = modeDepthEl?.value || "normal";
    const transcriptForBackend = [
      turn.prompt ? `User: ${turn.prompt}` : "",
      turn.answer ? `Assistant: ${turn.answer}` : "",
    ]
      .filter(Boolean)
      .join("\n\n");

    const result = await analyzeText({
      text: transcriptForBackend,
      question: turn.question || "",
      prompt: turn.prompt || "",
      context: turn.context || "",
      answer: turn.answer || "",
      mode_depth,
    });

    renderResult(result);
    setStatus("Conversation scan complete.");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${toUserErrorMessage(err)}`);
  } finally {
    setLoading(false);
  }
});

// ============================================================================
// MESSAGE LISTENERS
// ============================================================================

// Listen for automatic LLM turn results broadcast from the background.
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "BIAS_RESULT_FOR_LLM_TURN") {
    const { question, prompt, context, answer, result } = message.payload || {};
    const displayPrompt = question || prompt || "";
    const hasInteraction = !!(displayPrompt || context || answer);
    if (hasInteraction) {
      setLastInteraction(displayPrompt, context || "", answer || "");
    }
    if (result && hasInteraction) {
      renderResult(result);
      setStatus("Automatic scan of latest LLM reply.");
    }
    return;
  }

  if (message.type === "LLM_TURN_CLEARED") {
    clearPluginState();
    setStatus("Waiting for a new LLM reply…");
  }
});

// ============================================================================
// INITIALIZATION
// ============================================================================

bindOverlayControls();
loadOverlayPrefs();
loadBiasTypeSelection();
if (biasTypesEl) {
  biasTypesEl.addEventListener("change", saveBiasTypeSelection);
}

// Initialize UI visibility
updateUIVisibility();
