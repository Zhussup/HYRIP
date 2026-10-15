/* hyrip SPA — vanilla ES modules, no build step. */

const $ = (id) => document.getElementById(id);
const state = { article: null, sessionId: null, session: null, view: null,
                idx: 0, timer: null, startedAt: null, budget: null };

/* ---- api helpers ---------------------------------------------------------- */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* keep */ }
    throw new Error(detail);
  }
  return response.json();
}

/* SSE over POST or GET: read data: frames from a plain fetch stream. */
async function sse(url, options, onToken, onError) {
  const response = await fetch(url, options);
  if (!response.ok || !response.body) {
    let detail = `${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* keep */ }
    onError(detail);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, index); buffer = buffer.slice(index + 2);
      if (!frame.startsWith("data:")) continue;
      const payload = JSON.parse(frame.slice(5).trim());
      if (payload.token !== undefined) onToken(payload.token);
      if (payload.error) onError(payload.error);
      if (payload.done) return;
    }
  }
}

/* ---- screens -------------------------------------------------------------- */

const SCREENS = ["input", "prepare", "article", "questions", "report", "history"];
function show(name) {
  for (const id of SCREENS) $(`screen-${id}`).classList.toggle("hidden", id !== name);
}

/* ---- input ---------------------------------------------------------------- */

async function loadModels() {
  try {
    const models = await api("/api/models");
    const select = $("model");
    select.innerHTML = '<option value="">default</option>';
    for (const m of models) {
      const option = document.createElement("option");
      option.value = m.name; option.textContent = m.name;
      select.appendChild(option);
    }
  } catch { /* catalog not reachable — the server default still applies */ }
}

async function prepare() {
  const text = $("article-text").value.trim();
  const url = $("article-url").value.trim();
  const errorBox = $("input-error");
  errorBox.classList.add("hidden");
  let payload;
  if (text) payload = { source: "text", text };
  else if (url) payload = { source: "url", url };
  else {
    errorBox.textContent = "Paste an article or give a URL first.";
    errorBox.classList.remove("hidden");
    return;
  }
  let articleResponse;
  try {
    articleResponse = await api("/api/articles", { method: "POST", body: JSON.stringify(payload) });
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
    return;
  }
  state.article = articleResponse;
  state.sessionId = null;
  const session = {
    article_id: articleResponse.id,
    mode: $("mode").value,
    model: $("model").value || null,
    n_questions: parseInt($("n-questions").value, 10) || null,
    time_budget_min: parseInt($("time-budget").value, 10) || null,
  };
  try {
    const created = await api("/api/sessions", {
      method: "POST", body: JSON.stringify(session),
    });
    state.sessionId = created.session_id;
    state.session = null;
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
    return;
  }
  $("prepare-detail").textContent = `“${articleResponse.title}” — ${articleResponse.word_count} words`;
  show("prepare");
  pollGeneration();
}

let polling = false;
async function pollGeneration() {
  if (polling) return;
  polling = true;
  try {
    while (state.sessionId) {
      state.session = await api(`/api/sessions/${state.sessionId}`);
      if (state.session.status === "generating") { await sleep(1500); continue; }
      if (state.session.status === "failed") {
        show("prepare");
        $("prepare-error").textContent =
          "Question generation failed — try again, or paste the article text directly.";
        $("prepare-error").classList.remove("hidden");
        return;
      }
      showArticle();
      return;
    }
  } catch (err) {
    $("prepare-error").textContent = err.message;
    $("prepare-error").classList.remove("hidden");
  } finally { polling = false; }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---- article reading ------------------------------------------------------ */

function showArticle() {
  const article = state.article;
  $("article-title").textContent = article.title;
  $("article-meta").textContent =
    `${article.word_count} words · source: ${article.source}` +
    (article.source_url ? ` · ${article.source_url}` : "");
  const body = $("article-body");
  body.innerHTML = "";
  article.paragraphs.forEach((para, i) => {
    const p = document.createElement("p");
    const label = document.createElement("span");
    label.className = "plabel"; label.textContent = article.paragraph_labels[i];
    p.appendChild(label); p.appendChild(document.createTextNode(para));
    body.appendChild(p);
  });
  show("article");
}

function startAnswering() {
  api(`/api/sessions/${state.sessionId}/start`, { method: "POST" })
    .then(() => {
      state.startedAt = Date.now();
      state.budget = state.session.time_budget_min;
      launchTimer($("q-timer"));
      state.view = state.session.view;
      state.idx = state.session.view.answered.length
        ? Math.min(state.session.view.answered.length, state.view.n_questions - 1) : 0;
      show("questions");
      renderQuestion();
    })
    .catch((err) => alert(err.message));
}

function launchTimer(el) {
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
    const remaining = state.budget * 60 - elapsed;
    const sign = remaining >= 0 ? "" : "−";
    const abs = Math.abs(remaining);
    el.textContent =
      `${sign}${String(Math.floor(abs / 60)).padStart(2, "0")}:${String(abs % 60).padStart(2, "0")}`;
    el.classList.toggle("over", remaining < 300);
  }, 1000);
}

/* ---- questions ------------------------------------------------------------ */

function answersByNumber() {
  return new Set(state.view.answered);
}

function renderQuestion() {
  const view = state.view;
  const q = view.questions[state.idx];
  const group = view.groups.find((g) => g.group_no === q.group_no);
  $("q-group").textContent = `Questions ${group.group_no}`;
  $("q-instructions").textContent = group.instructions;
  const bank = $("q-headings");
  if (group.headings) {
    bank.textContent = group.headings.map((h) => `  ${h}`).join("\n");
    bank.classList.remove("hidden");
  } else bank.classList.add("hidden");

  const card = $("question-card");
  card.innerHTML = "";
  const heading = document.createElement("p");
  heading.innerHTML = `<span class="q-num">Q${q.number}.</span>`;
  heading.appendChild(document.createTextNode(q.prompt));
  card.appendChild(heading);

  const saved = locallySaved()[q.number] || "";
  const renderOption = (value, label, name) => {
    const row = document.createElement("div");
    row.className = "option-row";
    const input = document.createElement("input");
    input.type = "radio"; input.name = name; input.value = value;
    input.id = `${name}-${value}`;
    if (saved && saved.toUpperCase() === value.toUpperCase()) input.checked = true;
    row.appendChild(input);
    const lab = document.createElement("label");
    lab.htmlFor = input.id;
    lab.textContent = label === value ? value : `${value}) ${label}`;
    row.appendChild(lab);
    return row;
  };

  const build = {
    multiple_choice: () => q.options.forEach((opt, i) =>
      card.appendChild(renderOption(String.fromCharCode(65 + i), opt, `q${q.number}`))),
    summary_choice: () => q.options.forEach((opt, i) =>
      card.appendChild(renderOption(String.fromCharCode(65 + i), opt, `q${q.number}`))),
    true_false_not_given: () =>
      ["TRUE", "FALSE", "NOT GIVEN"].forEach((v) =>
        card.appendChild(renderOption(v, v, `q${q.number}`))),
    yes_no_not_given: () =>
      ["YES", "NO", "NOT GIVEN"].forEach((v) =>
        card.appendChild(renderOption(v, v, `q${q.number}`))),
    matching_headings: () => {
      const note = document.createElement("p");
      note.className = "limit-hint";
      note.textContent = `paragraph ${q.paragraph_label} — pick a heading (roman numeral)`;
      card.appendChild(note);
      const select = document.createElement("select");
      select.className = "q-input"; select.id = `input-${q.number}`;
      const blank = document.createElement("option"); blank.value = ""; blank.textContent = "— choose —";
      select.appendChild(blank);
      const bankGroup = view.groups.find((g) => g.group_no === q.group_no);
      (bankGroup?.headings || []).forEach((headingText, i) => {
        const opt = document.createElement("option");
        const roman = ((bankGroup?.headings || []).length > i)
          ? ROMANS[i + 1] : String(i + 1);
        opt.value = roman;
        opt.textContent = `${roman} — ${headingText.replace(/^\s*[ivx]+\.?\s*/, "")}`;
        select.appendChild(opt);
      });
      if (saved) select.value = saved;
      card.appendChild(select);
    },
    match_paragraphs: () => {
      const select = document.createElement("select");
      select.className = "q-input"; select.id = `input-${q.number}`;
      const blank = document.createElement("option"); blank.value = ""; blank.textContent = "— choose —";
      select.appendChild(blank);
      for (const label of state.article.paragraph_labels) {
        const opt = document.createElement("option");
        opt.value = label; opt.textContent = label;
        select.appendChild(opt);
      }
      if (saved) select.value = saved;
      card.appendChild(select);
    },
    completion: () => textInput(card, q, saved),
    short_answer: () => textInput(card, q, saved),
  };
  build[q.question_type]?.();

  const nav = $("btn-prev");
  nav.disabled = state.idx === 0;
  $("tutor-box").classList.add("hidden");
  renderDots();
}

const ROMANS = [null, "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"];

function textInput(card, q, saved) {
  const input = document.createElement("input");
  input.type = "text"; input.className = "q-input"; input.id = `input-${q.number}`;
  input.value = saved;
  if (q.word_limit) {
    const hint = document.createElement("p");
    hint.className = "limit-hint";
    hint.textContent = `no more than ${q.word_limit} words`;
    card.appendChild(hint);
  }
  card.appendChild(input);
}

function currentAnswer() {
  const q = state.view.questions[state.idx];
  const radios = document.getElementsByName(`q${q.number}`);
  for (const radio of radios) if (radio.checked) return radio.value;
  const input = $(`input-${q.number}`);
  return input ? input.value.trim() : "";
}

function saveLocally(number, raw) {
  const all = locallySaved();
  all[number] = raw;
  sessionStorage.setItem(`hyrip-${state.sessionId}`, JSON.stringify(all));
}
function locallySaved() {
  try { return JSON.parse(sessionStorage.getItem(`hyrip-${state.sessionId}`) || "{}"); }
  catch { return {}; }
}

async function commitAnswer(toIndex, { forceSkip = false } = {}) {
  const q = state.view.questions[state.idx];
  const raw = currentAnswer();
  if (!raw && !forceSkip) {
    throw new Error("Answer first, or press Skip to leave it blank.");
  }
  if (!raw) saveLocally(q.number, "");
  else saveLocally(q.number, raw);
  const response = await api(`/api/sessions/${state.sessionId}/answers`, {
    method: "POST",
    body: JSON.stringify({ number: q.number, raw: raw || null, skipped: !raw }),
  });
  markAnswered(q.number);
  if (state.session.mode === "tutor" && response.verdict) {
    await tutorRound(q, response.verdict);
  }
  state.idx = toIndex;
  renderQuestion();
}

function markAnswered(number) {
  const view = state.view;
  if (!view.answered.includes(number)) view.answered.push(number);
}

function renderDots() {
  const grid = $("dot-grid");
  grid.innerHTML = "";
  const answered = answersByNumber();
  state.view.questions.forEach((q, i) => {
    const dot = document.createElement("button");
    dot.className = "dot";
    dot.textContent = q.number;
    if (answered.has(q.number)) dot.classList.add("answered");
    if (i === state.idx) dot.classList.add("current");
    dot.addEventListener("click", () => { state.idx = i; renderQuestion(); });
    grid.appendChild(dot);
  });
}

/* ---- tutor box ------------------------------------------------------------ */

async function tutorRound(q, verdict) {
  const box = $("tutor-box");
  box.classList.remove("hidden");
  $("tutor-verdict").innerHTML =
    `<span class="verdict-${verdict.verdict}">${verdict.verdict}</span>` +
    ` <span class="dim">· marked by ${verdict.graded_by}</span>`;
  $("followup-thread").innerHTML = "";
  $("followup-input").value = "";
  await streamFeedback(q.number);
}

async function streamFeedback(number) {
  const feedback = $("tutor-feedback");
  feedback.textContent = "";
  feedback.classList.add("streaming");
  const error = (message) => {
    feedback.classList.remove("streaming");
    feedback.insertAdjacentText("beforeend", ` (tutor unavailable: ${message})`);
  };
  await sse(`/api/sessions/${state.sessionId}/tutor-feedback/${number}`, {},
    (token) => { feedback.insertBefore(document.createTextNode(token), null); },
    error);
  feedback.classList.remove("streaming");
}

async function sendFollowup() {
  const q = state.view.questions[state.idx];
  const input = $("followup-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const thread = $("followup-thread");
  const you = document.createElement("p");
  you.className = "you"; you.textContent = `you: ${text}`;
  thread.appendChild(you);
  const line = document.createElement("p");
  line.className = "tutor-line streaming";
  thread.appendChild(line);
  try {
    await sse(
      `/api/sessions/${state.sessionId}/tutor-followup`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ number: q.number, text }) },
      (token) => line.insertBefore(document.createTextNode(token), null),
      (message) => {
        line.classList.remove("streaming");
        line.textContent = ` (tutor unavailable: ${message})`;
      });
  } finally {
    line.classList.remove("streaming");
  }
}

/* ---- finish + report ------------------------------------------------------ */

function confirmFinish() {
  const unanswered =
    state.view.n_questions - answersByNumber().size;
  const message = unanswered
    ? `Finish with ${unanswered} unanswered? Unseen questions are marked skipped.`
    : "Finish and mark the paper?";
  if (confirm(message)) finish();
}

async function finish() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  try {
    const report = await api(`/api/sessions/${state.sessionId}/finish`, { method: "POST" });
    renderReport(report, state.session?.elapsed_seconds);
  } catch (err) {
    alert(`couldn't finish: ${err.message}`);
  }
}

function renderReport(report, elapsedSeconds) {
  $("score-raw").textContent = `${report.score.raw_correct}/${report.score.total}`;
  $("score-band").textContent = `band ≈ ${report.score.band}`;
  const minutes = Math.floor((elapsedSeconds ?? report.elapsed_seconds ?? 0) / 60);
  $("score-meta").textContent =
    `${minutes} min · ${report.score.pct}% raw${state.session?.mode === "tutor" ? " · tutor" : ""}`;
  $("score-disclaimer").textContent = report.score.disclaimer;

  const obs = $("report-observations");
  obs.innerHTML = report.observations.length ? "<h3>What your reading did</h3>" : "";
  const list = document.createElement("ul");
  list.className = "obs-list";
  for (const o of report.observations) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="obs-title"></span> <span class="obs-nums"></span><br>`;
    li.querySelector(".obs-title").textContent = o.title;
    li.querySelector(".obs-nums").textContent =
      `Q${o.question_numbers.join(", Q")} — ${o.detail}`;
    list.appendChild(li);
  }
  obs.appendChild(list);

  const focus = $("report-focus");
  focus.innerHTML = report.next_focus.length ? "<h3>Next focus</h3>" : "";
  const focusList = document.createElement("ul");
  focusList.className = "obs-list";
  for (const item of report.next_focus) {
    const li = document.createElement("li"); li.textContent = item;
    focusList.appendChild(li);
  }
  focus.appendChild(focusList);

  const reviews = $("report-reviews");
  reviews.innerHTML = "";
  for (const review of report.reviews) {
    const details = document.createElement("details");
    details.className = "review";
    const summary = document.createElement("summary");
    summary.innerHTML =
      `<span class="verdict ${review.verdict}">${review.verdict}</span>` +
      ` <span>Q${review.number}</span> <span class="dim"></span>`;
    summary.querySelector(".dim").textContent = review.prompt.slice(0, 90);
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "review-body";
    const row = (k, v) => {
      const p = document.createElement("p");
      p.className = "review-row";
      p.innerHTML = `<span class="k"></span>`;
      p.querySelector(".k").textContent = k;
      p.appendChild(document.createTextNode(v || "—"));
      return p;
    };
    body.appendChild(row("type", review.question_type));
    body.appendChild(row("you said", review.user_raw));
    body.appendChild(row("answer", review.answer_key));
    if (review.why) body.appendChild(row("why", review.why));
    if (review.evidence?.length) {
      const ev = review.evidence[0];
      const p = document.createElement("p");
      p.className = "review-row";
      p.innerHTML = `<span class="k">evidence</span>`;
      const quote = document.createElement("span");
      quote.className = "evidence-quote";
      quote.textContent =
        `“${ev.quote}” (${ev.paragraph})` + (ev.quote_verified ? "" : " ⚠ not verbatim");
      p.appendChild(quote);
      body.appendChild(p);
    }
    details.appendChild(body);
    reviews.appendChild(details);
  }
  show("report");
}

/* ---- history -------------------------------------------------------------- */

async function showHistory() {
  show("history");
  try {
    const sessions = await api("/api/history");
    const stats = await api("/api/stats");
    const tbody = $("history-table").querySelector("tbody");
    tbody.innerHTML = "";
    $("history-stats").textContent = stats.total_sessions
      ? `sessions: ${stats.total_sessions} · answers: ${stats.total_questions_answered} · ` +
        `correct: ${stats.total_correct}`
      : "No sessions yet — run one first.";
  for (const row of sessions) {
    const tr = document.createElement("tr");
    for (let cellIndex = 0; cellIndex < 8; cellIndex++) {
      tr.appendChild(document.createElement("td"));
    }
    const tds = [...tr.children];
    tds[0].textContent = row.id;
    tds[1].textContent = row.mode;
    tds[2].textContent = row.status;
    tds[3].textContent = row.passage_title || "—";
    tds[4].textContent = row.score_correct != null ? `${row.score_correct}/${row.score_total}` : "—";
    tds[5].textContent = row.band_estimate != null ? row.band_estimate.toFixed(1) : "—";
    tds[6].textContent = row.elapsed_seconds != null ? fmtElapsed(row.elapsed_seconds) : "—";
    if (row.status === "report") {
      const open = document.createElement("button");
      open.className = "secondary"; open.textContent = "replay";
      open.addEventListener("click", () => replay(row.id));
      tds[7].appendChild(open);
    }
    tbody.appendChild(tr);
  }
    drawSparkline(sessions.filter((s) => s.band_estimate != null));
  } catch (err) {
    $("history-stats").textContent = err.message;
  }
}

function fmtElapsed(seconds) {
  const m = Math.floor(seconds / 60), s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

async function replay(sessionId) {
  try {
    const report = await api(`/api/sessions/${sessionId}/report`);
    renderReport(report);
  } catch (err) { alert(err.message); }
}

function drawSparkline(rows) {
  const canvas = $("band-sparkline");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!rows.length) { return; }
  const step = canvas.width / Math.max(rows.length, 8);
  const toY = (band) => canvas.height - ((band - 3.0) / 6.0) * (canvas.height - 20) - 10;
  ctx.strokeStyle = "#0b7a55"; ctx.lineWidth = 2; ctx.beginPath();
  rows.forEach((row, i) => {
    const x = 15 + i * step, y = toY(row.band_estimate);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#0b7a55";
  rows.forEach((row, i) => {
    ctx.beginPath();
    ctx.arc(15 + i * step, toY(row.band_estimate), 3, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = "#5c6f66"; ctx.font = "11px sans-serif";
  ctx.fillText("band trend (3.5 → 9.0)", 12, 14);
}

/* ---- wiring ---------------------------------------------------------------- */

async function abandon() {
  if (!confirm("Quit and save as abandoned?")) return;
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  try { await api(`/api/sessions/${state.sessionId}/abandon`, { method: "POST" }); }
  catch { /* already settled */ }
  show("input");
}

function wire() {
  $("btn-prepare").addEventListener("click", prepare);
  $("btn-start-answering").addEventListener("click", startAnswering);
  $("btn-next").addEventListener("click", () => {
    const last = state.idx === state.view.n_questions - 1;
    if (last) return confirmFinish();
    return commitAnswer(state.idx + 1).catch((e) => alert(e.message));
  });
  $("btn-prev").addEventListener("click", () => {
    if (state.idx > 0) { state.idx -= 1; renderQuestion(); }
  });
  $("btn-skip").addEventListener("click", () => {
    const last = state.idx === state.view.n_questions - 1;
    if (last) return confirmFinish();
    return commitAnswer(state.idx + 1, { forceSkip: true }).catch((e) => alert(e.message));
  });
  $("btn-finish").addEventListener("click", confirmFinish);
  $("btn-abandon").addEventListener("click", abandon);
  $("btn-followup").addEventListener("click", sendFollowup);
  $("btn-again").addEventListener("click", () => {
    state.sessionId = null; state.article = null; state.view = null;
    show("input");
  });
  document.querySelectorAll("[data-nav]").forEach((btn) =>
    btn.addEventListener("click", () => {
      if (btn.dataset.nav === "history") return void showHistory();
      return void show("input");
    }));
  loadModels();
}

wire();