const $ = (sel) => document.querySelector(sel);

const state = {
  models: [],
  messages: [],
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.headers.get("content-type")?.includes("text/event-stream")) {
    return res;
  }
  return res.json();
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    const active = panel.id === `panel-${name}`;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

function fillSelect(el, models, selected) {
  el.innerHTML = models
    .map(
      (m) =>
        `<option value="${m.id}" ${m.id === selected ? "selected" : ""}>${m.name} (${m.params_b}B)</option>`
    )
    .join("");
}

async function loadPlatform() {
  const p = await api("/api/platform");
  $("#platform-chip").textContent = p.label;
  const stats = [
    ["OS", p.os],
    ["Arch", p.arch],
    ["Accel", p.accel],
    ["GPU", p.gpu_name || "—"],
    ["Memory", p.vram_gb != null ? `${p.vram_gb.toFixed(1)} GB` : "—"],
    ["Ollama", p.ollama ? "yes" : "no"],
    ["Python", p.python],
  ];
  $("#platform-stats").innerHTML = stats
    .map(([k, v]) => `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`)
    .join("");
  $("#platform-notes").innerHTML = (p.notes || []).map((n) => `<li>${n}</li>`).join("");
}

async function loadModels() {
  const method = $("#filter-method").value;
  const backend = $("#filter-backend").value;
  const data = await api(`/api/models?method=${method}&backend=${backend}&show_all=true`);
  state.models = data.models;
  $("#model-list").innerHTML = data.models
    .map(
      (m) => `
      <article class="model">
        <div>
          <h3>${m.name}</h3>
          <div class="meta">${m.id} · ${m.hf_id}${m.notes ? ` · ${m.notes}` : ""}</div>
          <div class="meta">${m.runnable ? m.reason : m.reason}</div>
        </div>
        <span class="badge ${m.runnable ? "yes" : "no"}">${m.runnable ? "trainable" : "blocked"}</span>
      </article>`
    )
    .join("");

  const selected = $("#cfg-model").value || data.models[0]?.id;
  fillSelect($("#cfg-model"), data.models, selected);
  fillSelect($("#chat-model"), data.models, $("#chat-model").value || selected);
}

function formConfig() {
  return {
    model: $("#cfg-model").value,
    method: $("#cfg-method").value,
    backend: $("#cfg-backend").value,
    dataset: $("#cfg-dataset").value,
    output_dir: $("#cfg-output").value,
    train: {
      epochs: Number($("#cfg-epochs").value || 1),
      lora_r: Number($("#cfg-lora-r").value || 16),
    },
  };
}

async function loadConfig() {
  const data = await api("/api/config");
  const c = data.config;
  $("#cfg-method").value = c.method;
  $("#cfg-backend").value = c.backend;
  $("#cfg-dataset").value = c.dataset;
  $("#cfg-output").value = c.output_dir;
  $("#cfg-epochs").value = c.train.epochs;
  $("#cfg-lora-r").value = c.train.lora_r;
  if (state.models.length) {
    fillSelect($("#cfg-model"), state.models, c.model);
    fillSelect($("#chat-model"), state.models, c.model);
  } else {
    $("#cfg-model").innerHTML = `<option value="${c.model}">${c.model}</option>`;
    $("#chat-model").innerHTML = `<option value="${c.model}">${c.model}</option>`;
  }
}

async function saveConfig() {
  await api("/api/config", { method: "PUT", body: JSON.stringify(formConfig()) });
  $("#train-log").textContent = "Config saved to config.yaml";
}

async function runTrain(dryRun) {
  $("#train-log").textContent = dryRun ? "Planning…" : "Training… this may take a while.";
  try {
    const data = await api("/api/train", {
      method: "POST",
      body: JSON.stringify({ dry_run: dryRun, config: formConfig() }),
    });
    $("#train-log").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $("#train-log").textContent = String(err.message || err);
  }
}

function appendBubble(role, text) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.textContent = text;
  $("#chat-thread").appendChild(el);
  $("#chat-thread").scrollTop = $("#chat-thread").scrollHeight;
  return el;
}

async function sendChat(event) {
  event.preventDefault();
  const input = $("#chat-input");
  const content = input.value.trim();
  if (!content) return;
  input.value = "";
  state.messages.push({ role: "user", content });
  appendBubble("user", content);
  const assistantEl = appendBubble("assistant", "");

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: $("#chat-model").value,
      messages: state.messages,
    }),
  });

  if (!res.ok) {
    assistantEl.textContent = `Error: ${res.statusText}`;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice(5).trim());
      if (payload.error) {
        assistantEl.textContent = payload.error;
        return;
      }
      if (payload.token) {
        full += payload.token;
        assistantEl.textContent = full;
        $("#chat-thread").scrollTop = $("#chat-thread").scrollHeight;
      }
    }
  }

  state.messages.push({ role: "assistant", content: full });
}

function wire() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
  $("#refresh-models").addEventListener("click", loadModels);
  $("#filter-method").addEventListener("change", loadModels);
  $("#filter-backend").addEventListener("change", loadModels);
  $("#save-config").addEventListener("click", saveConfig);
  $("#dry-run").addEventListener("click", () => runTrain(true));
  $("#start-train").addEventListener("click", () => {
    if (confirm("Start training on this machine? This uses local GPU/memory.")) {
      runTrain(false);
    }
  });
  $("#chat-form").addEventListener("submit", sendChat);
  $("#clear-chat").addEventListener("click", () => {
    state.messages = [];
    $("#chat-thread").innerHTML = "";
  });
}

async function boot() {
  wire();
  try {
    const health = await api("/api/health");
    $("#version").textContent = `LLMBench v${health.version}`;
    await loadPlatform();
    await loadModels();
    await loadConfig();
  } catch (err) {
    $("#platform-chip").textContent = `boot failed: ${err.message || err}`;
  }
}

boot();
