/**
 * Modular on-page overlay hub — bottom-right floating info boxes.
 * Each block is registered independently and can be toggled on/off.
 */
(function () {
  const STORAGE_KEY = "biasOverlayPrefs";
  const DEFAULT_PREFS = {
    enabled: true,
    modules: {
      score: true,
      explanation: true,
      flagged: false,
      spider: true,
    },
  };

  let rootEl = null;
  let showFabEl = null;
  let prefs = { ...DEFAULT_PREFS };
  let lastPayload = null;
  const moduleEls = new Map();

  function riskMeta(score) {
    const numeric = typeof score === "number" && Number.isFinite(score) ? score : 0;
    if (numeric >= 0.55) return { level: "High", color: "rgba(220, 38, 38, 0.92)", text: "#fef2f2" };
    if (numeric >= 0.25) return { level: "Medium", color: "rgba(234, 179, 8, 0.95)", text: "#1c1917" };
    return { level: "Low", color: "rgba(22, 163, 74, 0.92)", text: "#ecfdf5" };
  }

  function boxStyle(el) {
    el.style.pointerEvents = "auto";
    el.style.background = "rgba(11, 16, 32, 0.94)";
    el.style.border = "1px solid rgba(55, 65, 81, 0.9)";
    el.style.borderRadius = "12px";
    el.style.boxShadow = "0 12px 28px rgba(0, 0, 0, 0.38)";
    el.style.color = "#f9fafb";
    el.style.fontFamily =
      'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    el.style.fontSize = "12px";
    el.style.lineHeight = "1.4";
    el.style.maxWidth = "300px";
    el.style.overflow = "hidden";
    el.style.backdropFilter = "blur(8px)";
  }

  function makeHeader(title, moduleId) {
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "8px";
    header.style.padding = "8px 10px";
    header.style.borderBottom = "1px solid rgba(55, 65, 81, 0.65)";
    header.style.background = "rgba(2, 6, 23, 0.55)";
    header.style.cursor = "pointer";
    header.style.userSelect = "none";

    const titleEl = document.createElement("span");
    titleEl.textContent = title;
    titleEl.style.fontWeight = "600";
    titleEl.style.fontSize = "11px";
    titleEl.style.letterSpacing = "0.02em";
    titleEl.style.textTransform = "uppercase";
    titleEl.style.color = "#9ca3af";

    const collapseBtn = document.createElement("button");
    collapseBtn.type = "button";
    collapseBtn.textContent = "−";
    collapseBtn.title = "Collapse";
    collapseBtn.style.border = "none";
    collapseBtn.style.background = "transparent";
    collapseBtn.style.color = "#9ca3af";
    collapseBtn.style.cursor = "pointer";
    collapseBtn.style.fontSize = "14px";
    collapseBtn.style.padding = "0 2px";
    collapseBtn.style.lineHeight = "1";

    const body = document.createElement("div");
    body.dataset.moduleBody = moduleId;
    body.style.padding = "8px 10px 10px";

    let collapsed = false;
    const toggleCollapse = () => {
      collapsed = !collapsed;
      body.style.display = collapsed ? "none" : "block";
      collapseBtn.textContent = collapsed ? "+" : "−";
      collapseBtn.title = collapsed ? "Expand" : "Collapse";
    };

    header.addEventListener("click", (e) => {
      if (e.target === collapseBtn) return;
      toggleCollapse();
    });
    collapseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleCollapse();
    });

    header.appendChild(titleEl);
    header.appendChild(collapseBtn);
    return { header, body, collapseBtn };
  }

  function drawRadarChart(canvas, scores) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const labels = Object.keys(scores || {});
    const values = labels.map((k) => {
      const v = scores[k];
      const num = typeof v === "number" && Number.isFinite(v) ? v : 0;
      return Math.max(0, Math.min(1, num));
    });

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (labels.length < 1) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.55)";
      ctx.font = "11px system-ui, -apple-system, Segoe UI, sans-serif";
      ctx.fillText("No breakdown available.", 8, 18);
      return;
    }

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.32;

    const gridColor = "rgba(148, 163, 184, 0.22)";
    const axisColor = "rgba(148, 163, 184, 0.26)";
    const labelColor = "rgba(226, 232, 240, 0.85)";
    const polyFill = "rgba(34, 197, 94, 0.18)";
    const polyStroke = "rgba(34, 197, 94, 0.85)";
    const pointFill = "rgba(34, 197, 94, 0.95)";

    const n = labels.length;
    const angle0 = -Math.PI / 2;

    const polar = (i, r) => {
      const a = angle0 + (2 * Math.PI * i) / n;
      return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
    };

    // Draw concentric grid circles
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

    // Draw axes and labels
    ctx.font = "10px system-ui, -apple-system, Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      const p = polar(i, radius);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      const lp = polar(i, radius + 16);
      const label = labels[i];
      ctx.fillStyle = labelColor;
      const align = lp.x < cx - 6 ? "right" : lp.x > cx + 6 ? "left" : "center";
      ctx.textAlign = align;
      ctx.fillText(label, lp.x, lp.y);
    }

    // Draw data polygon
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

    // Draw data points
    for (let i = 0; i < n; i++) {
      const p = polar(i, radius * values[i]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = pointFill;
      ctx.fill();
    }
  }

  const MODULE_DEFS = {
    score: {
      title: "Bias score",
      render(result) {
        const score = typeof result?.bias_score === "number" ? result.bias_score : 0;
        const meta = riskMeta(score);
        const wrap = document.createElement("div");
        wrap.style.display = "flex";
        wrap.style.alignItems = "center";
        wrap.style.justifyContent = "space-between";
        wrap.style.gap = "10px";

        const badge = document.createElement("div");
        badge.style.padding = "6px 12px";
        badge.style.borderRadius = "999px";
        badge.style.background = meta.color;
        badge.style.color = meta.text;
        badge.style.fontWeight = "700";
        badge.style.fontSize = "13px";
        badge.textContent = `${score.toFixed(2)} · ${meta.level}`;

        const hint = document.createElement("span");
        hint.style.color = "#9ca3af";
        hint.style.fontSize = "11px";
        hint.textContent = "Traffic light";

        wrap.appendChild(badge);
        wrap.appendChild(hint);
        return wrap;
      },
    },
    explanation: {
      title: "Explanation",
      render(result) {
        const text = document.createElement("p");
        text.style.margin = "0";
        text.style.color = "#e5e7eb";
        text.style.whiteSpace = "pre-wrap";
        text.style.maxHeight = "96px";
        text.style.overflowY = "auto";
        text.textContent = result?.explanation || "No explanation available.";
        return text;
      },
    },
    flagged: {
      title: "Flagged sentences",
      render(result) {
        const list = document.createElement("ul");
        list.style.margin = "0";
        list.style.padding = "0";
        list.style.listStyle = "none";
        list.style.maxHeight = "110px";
        list.style.overflowY = "auto";

        const sentences = Array.isArray(result?.biased_sentences)
          ? result.biased_sentences
          : [];

        if (sentences.length === 0) {
          const li = document.createElement("li");
          li.style.color = "#9ca3af";
          li.textContent = "No flagged sentences.";
          list.appendChild(li);
          return list;
        }

        sentences.forEach((s) => {
          const li = document.createElement("li");
          li.style.padding = "5px 7px";
          li.style.marginBottom = "4px";
          li.style.borderRadius = "8px";
          li.style.background = "rgba(3, 7, 18, 0.85)";
          li.style.border = "1px solid rgba(31, 41, 55, 0.9)";
          li.textContent = s;
          list.appendChild(li);
        });
        return list;
      },
    },
    spider: {
      title: "Bias breakdown",
      render(result) {
        const biasTypeScores = result?.bias_type_scores || {};
        const hasScores = Object.keys(biasTypeScores).length > 0;
        
        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "6px";

        if (!hasScores) {
          const msg = document.createElement("p");
          msg.style.margin = "0";
          msg.style.color = "#9ca3af";
          msg.style.fontSize = "11px";
          msg.textContent = "No bias breakdown available.";
          container.appendChild(msg);
          return container;
        }

        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = 140;
        canvas.style.display = "block";
        canvas.style.margin = "0 auto";
        
        container.appendChild(canvas);
        
        // Draw the radar chart
        drawRadarChart(canvas, biasTypeScores);
        
        return container;
      },
    },
  };

  function ensureRoot() {
    if (rootEl) return rootEl;

    rootEl = document.createElement("div");
    rootEl.id = "bias-overlay-hub";
    rootEl.setAttribute("data-bias-extension-ui", "true");
    rootEl.setAttribute("aria-hidden", "true");
    rootEl.style.position = "fixed";
    rootEl.style.bottom = "16px";
    rootEl.style.right = "16px";
    rootEl.style.zIndex = "2147483646";
    rootEl.style.display = "flex";
    rootEl.style.flexDirection = "column-reverse";
    rootEl.style.alignItems = "flex-end";
    rootEl.style.gap = "8px";
    rootEl.style.maxWidth = "300px";
    rootEl.style.pointerEvents = "none";

    const toolbar = document.createElement("div");
    toolbar.style.pointerEvents = "auto";
    toolbar.style.display = "flex";
    toolbar.style.alignItems = "center";
    toolbar.style.gap = "6px";
    toolbar.style.padding = "5px 8px";
    toolbar.style.borderRadius = "999px";
    toolbar.style.background = "rgba(11, 16, 32, 0.94)";
    toolbar.style.border = "1px solid rgba(55, 65, 81, 0.9)";
    toolbar.style.boxShadow = "0 8px 20px rgba(0, 0, 0, 0.35)";
    toolbar.style.fontSize = "11px";
    toolbar.style.color = "#9ca3af";

    const label = document.createElement("span");
    label.textContent = "Bias info";

    const hideBtn = document.createElement("button");
    hideBtn.type = "button";
    hideBtn.textContent = "Hide";
    hideBtn.title = "Hide on-page info boxes";
    hideBtn.style.border = "1px solid #374151";
    hideBtn.style.background = "#020617";
    hideBtn.style.color = "#e5e7eb";
    hideBtn.style.borderRadius = "999px";
    hideBtn.style.padding = "3px 8px";
    hideBtn.style.fontSize = "11px";
    hideBtn.style.cursor = "pointer";
    hideBtn.addEventListener("click", () => {
      setEnabled(false, true);
    });

    toolbar.appendChild(label);
    toolbar.appendChild(hideBtn);
    rootEl.appendChild(toolbar);
    rootEl.dataset.toolbar = "true";

    document.body.appendChild(rootEl);
    return rootEl;
  }

  function ensureShowFab() {
    if (showFabEl) return showFabEl;

    showFabEl = document.createElement("button");
    showFabEl.type = "button";
    showFabEl.id = "bias-overlay-show-fab";
    showFabEl.setAttribute("data-bias-extension-ui", "true");
    showFabEl.setAttribute("aria-hidden", "true");
    showFabEl.textContent = "Show bias info";
    showFabEl.title = "Show on-page info boxes";
    showFabEl.style.position = "fixed";
    showFabEl.style.bottom = "16px";
    showFabEl.style.right = "16px";
    showFabEl.style.zIndex = "2147483646";
    showFabEl.style.padding = "8px 12px";
    showFabEl.style.borderRadius = "999px";
    showFabEl.style.border = "1px solid rgba(55, 65, 81, 0.9)";
    showFabEl.style.background = "rgba(11, 16, 32, 0.94)";
    showFabEl.style.color = "#e5e7eb";
    showFabEl.style.fontSize = "12px";
    showFabEl.style.cursor = "pointer";
    showFabEl.style.boxShadow = "0 10px 24px rgba(0, 0, 0, 0.35)";
    showFabEl.style.fontFamily =
      'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

    showFabEl.addEventListener("click", () => {
      setEnabled(true, true);
    });

    document.body.appendChild(showFabEl);
    return showFabEl;
  }

  function createModuleBox(moduleId) {
    const def = MODULE_DEFS[moduleId];
    if (!def) return null;

    const box = document.createElement("div");
    box.dataset.moduleId = moduleId;
    boxStyle(box);

    const { header, body } = makeHeader(def.title, moduleId);
    box.appendChild(header);
    box.appendChild(body);
    return { box, body };
  }

  function renderModule(moduleId, result) {
    const def = MODULE_DEFS[moduleId];
    if (!def) return;

    let entry = moduleEls.get(moduleId);
    if (!entry) {
      const created = createModuleBox(moduleId);
      if (!created) return;
      entry = created;
      moduleEls.set(moduleId, entry);
      ensureRoot().appendChild(entry.box);
    }

    entry.body.innerHTML = "";
    entry.body.appendChild(def.render(result));
  }

  function applyVisibility() {
    if (prefs.enabled) {
      if (showFabEl) showFabEl.style.display = "none";
      if (rootEl) rootEl.style.display = "flex";
    } else {
      if (rootEl) rootEl.style.display = "none";
      ensureShowFab().style.display = "block";
    }
  }

  function syncModuleVisibility() {
    const enabledModules = prefs.modules || {};
    moduleEls.forEach((entry, moduleId) => {
      entry.box.style.display = enabledModules[moduleId] ? "block" : "none";
    });
  }

  function updateOverlay(payload) {
    lastPayload = payload;
    const result = payload?.result;
    if (!result) return;

    if (!prefs.enabled) {
      applyVisibility();
      return;
    }

    ensureRoot();
    applyVisibility();

    Object.keys(MODULE_DEFS).forEach((moduleId) => {
      if (prefs.modules?.[moduleId]) {
        renderModule(moduleId, result);
      }
    });
    syncModuleVisibility();
  }

  function setPrefs(nextPrefs, persist = false) {
    prefs = {
      enabled:
        typeof nextPrefs?.enabled === "boolean" ? nextPrefs.enabled : prefs.enabled,
      modules: {
        ...DEFAULT_PREFS.modules,
        ...(prefs.modules || {}),
        ...(nextPrefs?.modules || {}),
      },
    };

    applyVisibility();

    if (lastPayload?.result) {
      Object.keys(MODULE_DEFS).forEach((moduleId) => {
        if (prefs.modules?.[moduleId]) {
          renderModule(moduleId, lastPayload.result);
        }
      });
    }
    syncModuleVisibility();

    if (persist && chrome?.storage?.local) {
      chrome.storage.local.set({ [STORAGE_KEY]: prefs });
    }
  }

  function setEnabled(enabled, persist = false) {
    setPrefs({ enabled: !!enabled }, persist);
  }

  function setModuleEnabled(moduleId, enabled, persist = false) {
    setPrefs({ modules: { [moduleId]: !!enabled } }, persist);
  }

  function loadPrefs(callback) {
    if (!chrome?.storage?.local) {
      callback?.(prefs);
      return;
    }
    chrome.storage.local.get(STORAGE_KEY, (data) => {
      const stored = data?.[STORAGE_KEY];
      if (stored && typeof stored === "object") {
        prefs = {
          enabled:
            typeof stored.enabled === "boolean" ? stored.enabled : DEFAULT_PREFS.enabled,
          modules: { ...DEFAULT_PREFS.modules, ...(stored.modules || {}) },
        };
      }
      applyVisibility();
      if (lastPayload?.result) updateOverlay(lastPayload);
      callback?.(prefs);
    });
  }

  window.BiasOverlayHub = {
    MODULE_IDS: Object.keys(MODULE_DEFS),
    DEFAULT_PREFS,
    STORAGE_KEY,
    update: updateOverlay,
    setPrefs,
    setEnabled,
    setModuleEnabled,
    getPrefs: () => ({ ...prefs, modules: { ...prefs.modules } }),
    loadPrefs,
  };
})();
