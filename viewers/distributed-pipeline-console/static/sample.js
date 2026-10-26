const params = new URLSearchParams(window.location.search);

const state = {
  date: params.get("date") || "",
  name: params.get("name") || "",
  file: params.get("file") || "",
  rowIndex: params.get("row_index") || "0",
};

const els = {
  subtitle: document.querySelector("#detailSubtitle"),
  content: document.querySelector("#detailContent"),
  backButton: document.querySelector("#backButton"),
};

async function api(path) {
  const response = await fetch(path, {headers: {"Content-Type": "application/json"}});
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function runUrl(suffix) {
  return `/api/runs/${encodeURIComponent(state.date)}/${encodeURIComponent(state.name)}${suffix}`;
}

function audioUrl(sample, key) {
  const query = new URLSearchParams();
  query.set("file", sample.file);
  query.set("row_index", String(sample.row_index));
  query.set("key", key);
  return runUrl(`/parquet-audio?${query.toString()}`);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function render(sample) {
  els.subtitle.textContent = `${state.date}/${state.name} | ${sample.file} | row ${sample.row_index}`;
  els.content.innerHTML = `
    <article class="sample-card sample-card-full">
      <div class="sample-head">
        <div>
          <h2>${escapeHtml(sample.audio_id || `row ${sample.row_index}`)}</h2>
          <div class="subtle">${escapeHtml(sample.from_dataset || "-")}</div>
        </div>
      </div>
      ${sample.text ? `<div class="sample-text">${escapeHtml(sample.text)}</div>` : ""}
      <div class="audio-list">
        ${(sample.audio_keys || []).map((key) => `
          <label>
            <span>${escapeHtml(key)}</span>
            <audio controls preload="none" src="${escapeHtml(audioUrl(sample, key))}"></audio>
          </label>
        `).join("") || '<span class="subtle">No audio column.</span>'}
      </div>
      <div class="section-row">
        <h3>Metadata</h3>
        <pre class="metadata-block">${escapeHtml(JSON.stringify(sample.metadata || {}, null, 2))}</pre>
      </div>
    </article>
  `;
}

async function load() {
  if (!state.date || !state.name || !state.file) {
    els.content.innerHTML = '<div class="warning">Missing date, name, or file query parameter.</div>';
    return;
  }
  const query = new URLSearchParams();
  query.set("file", state.file);
  query.set("row_index", state.rowIndex);
  try {
    render(await api(runUrl(`/parquet-row?${query.toString()}`)));
  } catch (error) {
    els.content.innerHTML = `<div class="warning">${escapeHtml(error.message)}</div>`;
  }
}

els.backButton.addEventListener("click", () => window.history.back());
load();
