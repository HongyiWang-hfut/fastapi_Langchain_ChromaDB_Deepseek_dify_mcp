/* ==========================================================================
   合肥工业大学 · 校园智能问答工作台 v0.7 SPA
   多模块架构：hash 路由 + 视图切换 + 各模块交互
   ========================================================================== */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* ---- 全局状态 ---- */
const state = {
  theme: 'light',
  studentId: 'S001',
  apiKey: 'campus-qa-dev-key',
  endpoint: '/ask',
  view: 'dashboard',
  msgCount: 0,
  inited: new Set(),
};

/* ---- 路由表 ---- */
const ROUTES = {
  dashboard: { title: '概览', crumb: 'Dashboard' },
  chat:      { title: '智能问答', crumb: 'Chat' },
  tools:     { title: '校园工具', crumb: 'Tools' },
  knowledge: { title: '知识库', crumb: 'Knowledge' },
  history:   { title: '对话历史', crumb: 'History' },
  settings:  { title: '设置', crumb: 'Settings' },
};

/* ============================================================
   helpers
   ============================================================ */
const esc = (t) => String(t).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) {
    try {
      marked.setOptions({ breaks: true, gfm: true });
      const raw = marked.parse(String(text));
      return window.DOMPurify.sanitize(raw, {
        ALLOWED_TAGS: ['p','br','strong','em','del','code','pre','blockquote',
          'ul','ol','li','h1','h2','h3','h4','h5','h6','a','table','thead',
          'tbody','tr','th','td','hr','span'],
        ALLOWED_ATTR: ['href','title'],
      });
    } catch (_) { /* 降级 */ }
  }
  return esc(text).replaceAll('\n', '<br>');
}

function headers() {
  const keyEl = $('#api-key');
  return {
    'Content-Type': 'application/json',
    'X-API-Key': (keyEl ? keyEl.value.trim() : '') || state.apiKey,
  };
}

/* ============================================================
   主题管理
   ============================================================ */
function initTheme() {
  const saved = localStorage.getItem('campus-qa-theme');
  const mode = localStorage.getItem('campus-qa-theme-mode') || 'manual';
  const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = mode === 'auto' ? (prefersDark ? 'dark' : 'light') : (saved || (prefersDark ? 'dark' : 'light'));
  applyTheme(theme);
  syncSettingsTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (localStorage.getItem('campus-qa-theme-mode') === 'auto') {
      applyTheme(e.matches ? 'dark' : 'light');
    }
  });
}

function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  const btn = $('#theme-btn');
  if (btn) {
    const icon = btn.querySelector('.theme-icon');
    if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', theme === 'dark' ? '切换浅色模式' : '切换深色模式');
  }
  const now = $('#set-theme-now');
  if (now) now.textContent = theme === 'dark' ? '深色' : '浅色';
}

function toggleTheme() {
  const next = state.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  localStorage.setItem('campus-qa-theme', next);
  localStorage.setItem('campus-qa-theme-mode', 'manual');
  syncSettingsTheme();
}

function setThemeMode(mode) {
  localStorage.setItem('campus-qa-theme-mode', mode);
  if (mode === 'auto') {
    const d = matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(d ? 'dark' : 'light');
    localStorage.setItem('campus-qa-theme', d ? 'dark' : 'light');
  } else {
    applyTheme(mode);
    localStorage.setItem('campus-qa-theme', mode);
  }
  syncSettingsTheme();
}

function syncSettingsTheme() {
  const mode = localStorage.getItem('campus-qa-theme-mode') || 'manual';
  $$('.seg-btn').forEach(b => {
    const m = b.dataset.themeSet;
    b.classList.toggle('active', (mode === 'auto' && m === 'auto') || (mode === 'manual' && m === state.theme));
  });
}

/* ============================================================
   路由 / 视图切换
   ============================================================ */
function handleRoute() {
  const hash = location.hash.replace(/^#\/?/, '') || 'dashboard';
  const name = ROUTES[hash] ? hash : 'dashboard';
  switchView(name);
}

function switchView(name) {
  state.view = name;
  $$('.view').forEach(v => v.classList.remove('active'));
  const target = $('#view-' + name);
  if (target) target.classList.add('active');

  $$('.nav-link').forEach(a => a.classList.toggle('active', a.dataset.route === name));
  const meta = ROUTES[name];
  $('#view-title').textContent = meta.title;
  $('#view-crumb').textContent = meta.crumb;

  // 关闭移动端菜单
  $('.side-nav')?.classList.remove('open');
  $('#menu-toggle')?.setAttribute('aria-expanded', 'false');

  // 滚动到顶
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // 懒初始化
  if (!state.inited.has(name)) {
    state.inited.add(name);
    const fn = INITERS[name];
    if (fn) try { fn(); } catch (e) { console.error('init ' + name, e); }
  }
  // 每次进入也刷新
  const ref = REFRESHERS[name];
  if (ref) try { ref(); } catch (e) { console.error('refresh ' + name, e); }
}

const INITERS = {};
const REFRESHERS = {};

/* ============================================================
   Dashboard
   ============================================================ */
INITERS.dashboard = function initDashboard() {
  // 数字滚动
  $$('.count').forEach(el => {
    const to = parseInt(el.dataset.to, 10) || 0;
    animateCount(el, to);
  });
  // 状态检测
  checkStatus();
};

function animateCount(el, to) {
  if (to <= 0) { el.textContent = '0'; return; }
  const dur = 1100, start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(to * eased);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

async function checkStatus() {
  const set = (id, txt, cls) => { const e = $('#' + id); if (e) { e.textContent = txt; e.className = 'badge ' + cls; } };
  const pulse = $('#nav-status .status-pulse');
  const pulseTxt = $('#nav-status-text');
  set('d-fastapi', '检测中', '');
  // FastAPI
  try {
    const r = await fetch('/health');
    if (r.ok) { set('d-fastapi', '在线', 'on'); pulse.classList.add('on'); pulse.classList.remove('err'); pulseTxt.textContent = '服务在线'; }
    else throw 0;
  } catch { set('d-fastapi', '离线', 'err'); pulse.classList.add('err'); pulse.classList.remove('on'); pulseTxt.textContent = '服务离线'; return; }

  // RAG（轻量检索，不调 LLM）
  set('d-rag', '检测中', '');
  try {
    const r = await fetch('/knowledge/search?q=图书馆&k=1', { headers: headers() });
    const d = await r.json();
    set('d-rag', d.count > 0 ? '就绪' : '空库', d.count > 0 ? 'on' : '');
  } catch { set('d-rag', '异常', 'err'); }

  // MCP / LLM：预热完成后标记就绪（避免昂贵的 LLM 调用）
  set('d-mcp', '就绪', 'on');
  set('d-llm', '就绪', 'on');

  // 累计问答数（取当前学生历史数）
  try {
    const sid = state.studentId;
    const r = await fetch('/history/' + encodeURIComponent(sid) + '?limit=200', { headers: headers() });
    const d = await r.json();
    const qaEl = $('#qa-count');
    if (qaEl) animateCount(qaEl, d.count || 0);
  } catch { /* ignore */ }
};

REFRESHERS.dashboard = function () {
  // 重新进入 dashboard 时刷新问答计数
  const qaEl = $('#qa-count');
  if (qaEl) {
    fetch('/history/' + encodeURIComponent(state.studentId) + '?limit=200', { headers: headers() })
      .then(r => r.json()).then(d => animateCount(qaEl, d.count || 0)).catch(() => {});
  }
};

/* ============================================================
   Chat（迁移并增强现有对话逻辑）
   ============================================================ */
INITERS.chat = function initChat() {
  const sendBtn = $('#send-btn');
  const resetBtn = $('#reset-btn');
  const questionEl = $('#question');
  const studentIdEl = $('#student-id');
  const apiKeyEl = $('#api-key');
  const endpointEl = $('#endpoint');

  // 恢复保存的偏好
  const savedSid = localStorage.getItem('campus-qa-sid');
  const savedEp = localStorage.getItem('campus-qa-endpoint');
  if (savedSid) { studentIdEl.value = savedSid; state.studentId = savedSid; }
  if (savedEp) { endpointEl.value = savedEp; state.endpoint = savedEp; }

  // 同步顶栏 SID
  const syncSid = () => { const t = $('#topbar-sid'); if (t) t.textContent = studentIdEl.value.trim() || 'S001'; state.studentId = studentIdEl.value.trim() || 'S001'; };
  syncSid();
  studentIdEl.addEventListener('input', syncSid);

  sendBtn.addEventListener('click', ask);
  resetBtn.addEventListener('click', resetConv);
  questionEl.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') ask(); });

  // 快捷预设
  $$('.quick-row .pill[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => { questionEl.value = btn.dataset.preset; questionEl.focus(); });
  });
};

let streamCtrl = null;

function setBusy(busy) { const m = $('#messages'); if (m) m.setAttribute('aria-busy', busy ? 'true' : 'false'); }

function addMsg(role, text, meta, isError) {
  const messagesEl = $('#messages');
  const el = document.createElement('div');
  el.className = `msg ${role}${isError ? ' error' : ''}`;
  const avatar = role === 'user' ? '你' : 'AI';
  const body = role === 'user'
    ? esc(text).replaceAll('\n', '<br>')
    : (isError ? esc(text).replaceAll('\n', '<br>') : renderMarkdown(text));
  el.innerHTML =
    `<div class="msg-avatar" aria-hidden="true">${avatar}</div>` +
    `<div class="msg-bubble${role === 'assistant' && !isError ? ' md' : ''}">${body}</div>` +
    (meta ? `<div class="typing-hint" style="padding:2px 4px">${esc(meta)}</div>` : '');
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function showTyping() {
  const messagesEl = $('#messages');
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `<div class="msg-avatar" aria-hidden="true">AI</div><div class="msg-bubble"><div class="typing-dots" aria-label="正在思考"><span></span><span></span><span></span></div></div>`;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function typingToStream(el) { const b = el.querySelector('.msg-bubble'); if (b) b.innerHTML = '<span class="s-text"></span>'; return el; }
function appendStream(el, text) { const s = el.querySelector('.s-text'); if (s) { s.textContent += text; $('#messages').scrollTop = $('#messages').scrollHeight; } }
function finishStream(el, meta) {
  const s = el.querySelector('.s-text'); const b = el.querySelector('.msg-bubble');
  if (s && b) { b.className = 'msg-bubble md'; b.innerHTML = renderMarkdown(s.textContent); }
  if (meta) { const h = document.createElement('div'); h.className = 'typing-hint'; h.style.padding = '2px 4px'; h.textContent = meta; el.appendChild(h); }
}

function setStatus({ mode, autoGenerated, toolsUsed, toolResults: tr }) {
  const set = (id, v) => { const e = $('#' + id); if (e) e.textContent = v; };
  set('mode-tag', mode || '-');
  set('auto-tag', autoGenerated ? '是' : '否');
  set('tools-tag', toolsUsed?.length ? toolsUsed.join('、') : '-');
  const trEl = $('#tool-results');
  if (trEl) trEl.textContent = (tr && Object.keys(tr).length) ? JSON.stringify(tr, null, 2) : '未命中或未返回。';
}

async function askNormal(question, endpoint, sid) {
  const res = await fetch(endpoint, { method: 'POST', headers: headers(), body: JSON.stringify({ question, student_id: sid }) });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
  return data;
}

async function askStream(question, sid, el) {
  streamCtrl = new AbortController();
  const res = await fetch('/ask/stream', { method: 'POST', headers: headers(), body: JSON.stringify({ question, student_id: sid }), signal: streamCtrl.signal });
  if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d?.detail || `HTTP ${res.status}`); }
  const reader = res.body.getReader(); const dec = new TextDecoder();
  let buf = '', meta = null, firstToken = true;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n'); buf = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const p = JSON.parse(line.slice(6));
      if (p.event === 'meta') { meta = p; setStatus(meta); }
      else if (p.event === 'token') { if (firstToken) { typingToStream(el); firstToken = false; } appendStream(el, p.token); }
      else if (p.event === 'done') { finishStream(el, meta ? `模式：${meta.mode}${meta.auto_generated ? ' · 自动生成' : ''} · 完成` : '完成'); }
    }
  }
  if (firstToken) { el.querySelector('.msg-bubble').innerHTML = '<span class="typing-hint">未收到内容。</span>'; }
  streamCtrl = null;
}

async function ask() {
  const questionEl = $('#question'); const endpointEl = $('#endpoint'); const studentIdEl = $('#student-id');
  const sendBtn = $('#send-btn');
  const q = questionEl.value.trim();
  const endpoint = endpointEl.value;
  const sid = studentIdEl.value.trim() || 'S001';
  if (!q) return questionEl.focus();
  if (streamCtrl) { streamCtrl.abort(); streamCtrl = null; }

  sendBtn.disabled = true; sendBtn.textContent = '…';
  setBusy(true);
  addMsg('user', q, `接口：${endpoint} · ${sid}`);
  state.msgCount++;

  try {
    if (endpoint === '/ask/stream') {
      const el = showTyping();
      await askStream(q, sid, el);
    } else {
      const el = showTyping();
      try {
        const data = await askNormal(q, endpoint, sid);
        el.remove();
        addMsg('assistant', data.answer, `模式：${data.mode || '-'}${data.auto_generated ? ' · 自动生成' : ''}`);
        setStatus(data);
      } catch (e) { el.remove(); throw e; }
    }
  } catch (e) {
    if (e.name === 'AbortError') { setBusy(false); return; }
    addMsg('assistant', `请求失败：${e.message}`, '错误', true);
    setStatus({ mode: 'error', autoGenerated: false, toolsUsed: [], toolResults: {} });
  } finally {
    sendBtn.disabled = false; sendBtn.textContent = '发 送';
    setBusy(false);
    questionEl.focus();
  }
}

async function resetConv() {
  const sid = $('#student-id').value.trim() || 'S001';
  if (streamCtrl) { streamCtrl.abort(); streamCtrl = null; }
  try {
    const res = await fetch('/reset', { method: 'POST', headers: headers(), body: JSON.stringify({ student_id: sid }) });
    const data = await res.json();
    if (data.status === 'ok') {
      state.msgCount = 0;
      setStatus({ mode: '已重置', autoGenerated: false, toolsUsed: [], toolResults: {} });
      $('#tool-results').textContent = '对话历史已清除。';
      addMsg('assistant', `已清除 ${sid} 的对话历史。`, '系统');
    }
  } catch (e) { addMsg('assistant', `重置失败：${e.message}`, '错误', true); }
}

/* ============================================================
   Tools
   ============================================================ */
INITERS.tools = function initTools() {
  $$('.tool-card').forEach(card => {
    const go = () => {
      const preset = card.dataset.preset;
      // 跳转到 chat 并填充
      location.hash = '#/chat';
      // 切换后填充（view 已激活）
      setTimeout(() => {
        const q = $('#question');
        if (q) { q.value = preset; q.focus(); }
      }, 80);
    };
    card.addEventListener('click', go);
  });
};

/* ============================================================
   Knowledge
   ============================================================ */
INITERS.knowledge = function initKnowledge() {
  $('#know-form').addEventListener('submit', e => {
    e.preventDefault();
    searchKnowledge();
  });
};

async function searchKnowledge() {
  const q = $('#know-q').value.trim();
  const box = $('#know-results');
  if (!q) return;
  // 骨架屏
  box.innerHTML = '';
  for (let i = 0; i < 3; i++) {
    const sk = document.createElement('div');
    sk.className = 'skeleton';
    sk.style.cssText = 'height:78px;border-radius:10px';
    box.appendChild(sk);
  }
  try {
    const r = await fetch('/knowledge/search?q=' + encodeURIComponent(q) + '&k=6', { headers: headers() });
    const d = await r.json();
    if (!r.ok) throw new Error(d?.detail || `HTTP ${r.status}`);
    renderKnowResults(d);
  } catch (e) {
    box.innerHTML = `<div class="know-empty">检索失败：${esc(e.message)}</div>`;
  }
}

function renderKnowResults(d) {
  const box = $('#know-results');
  if (!d.count) { box.innerHTML = '<div class="know-empty">未命中相关文档，换个关键词试试。</div>'; return; }
  box.innerHTML = d.hits.map((h, i) => `
    <div class="know-hit">
      <div class="know-hit-head">
        <strong>#${i + 1} 命中文档</strong>
        <span class="know-hit-score">距离 ${Number(h.distance ?? 0).toFixed(3)}</span>
      </div>
      <div class="know-hit-content">${esc(h.content || '').slice(0, 240)}${(h.content || '').length > 240 ? '…' : ''}</div>
      <div class="know-hit-meta">来源：${esc(h.source || 'hybrid')} · 模式：向量 + BM25 RRF 融合</div>
    </div>`).join('');
}

/* ============================================================
   History
   ============================================================ */
INITERS.history = function initHistory() {
  $('#hist-load').addEventListener('click', loadHistory);
  $('#hist-clear').addEventListener('click', clearHistory);
  $('#hist-sid').addEventListener('keydown', e => { if (e.key === 'Enter') loadHistory(); });
};

async function loadHistory() {
  const sid = $('#hist-sid').value.trim() || 'S001';
  const list = $('#hist-list');
  list.innerHTML = '<div class="skeleton" style="height:60px;border-radius:10px"></div><div class="skeleton" style="height:60px;border-radius:10px"></div>';
  try {
    const r = await fetch('/history/' + encodeURIComponent(sid) + '?limit=50', { headers: headers() });
    const d = await r.json();
    if (!r.ok) throw new Error(d?.detail || `HTTP ${r.status}`);
    renderHistory(d, sid);
  } catch (e) {
    list.innerHTML = `<div class="know-empty">加载失败：${esc(e.message)}</div>`;
  }
}

function renderHistory(d, sid) {
  const list = $('#hist-list');
  if (!d.count) { list.innerHTML = `<div class="know-empty">学生 ${esc(sid)} 暂无问答记录。去「智能问答」聊几句吧。</div>`; return; }
  list.innerHTML = d.logs.map(log => {
    const time = log.created_at ? new Date(log.created_at).toLocaleString('zh-CN') : '-';
    const modeTag = log.mode === 'mcp' ? 'mcp' : (log.auto_generated ? 'auto' : '');
    const modeTxt = log.mode === 'mcp' ? 'MCP 工具' : (log.auto_generated ? '自动生成' : 'RAG 检索');
    const toolTags = (log.tools_used || []).map(t => `<span class="hist-tag tool">${esc(t)}</span>`).join('');
    return `<div class="hist-item ${modeTag}">
      <div class="hist-meta">
        <div class="hist-tags"><span class="hist-tag ${modeTag}">${modeTxt}</span>${toolTags}</div>
        <span class="hist-time">${time}</span>
      </div>
      <div class="hist-q">Q：${esc(log.question)}</div>
      <div class="hist-a">${esc(log.answer)}</div>
    </div>`;
  }).join('');
}

async function clearHistory() {
  const sid = $('#hist-sid').value.trim() || 'S001';
  if (!confirm(`确定清空学生 ${sid} 的全部对话历史？此操作不可恢复。`)) return;
  try {
    await fetch('/reset', { method: 'POST', headers: headers(), body: JSON.stringify({ student_id: sid }) });
    loadHistory();
  } catch (e) {
    $('#hist-list').innerHTML = `<div class="know-empty">清空失败：${esc(e.message)}</div>`;
  }
}

REFRESHERS.history = function () { /* 进入时不自动加载，避免误触发 */ };

/* ============================================================
   Settings
   ============================================================ */
INITERS.settings = function initSettings() {
  $$('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => setThemeMode(btn.dataset.themeSet));
  });
  // 恢复保存的偏好
  const savedSid = localStorage.getItem('campus-qa-sid');
  const savedEp = localStorage.getItem('campus-qa-endpoint');
  if (savedSid) $('#set-sid').value = savedSid;
  if (savedEp) $('#set-endpoint').value = savedEp;
  $('#set-save').addEventListener('click', () => {
    const sid = $('#set-sid').value.trim() || 'S001';
    const ep = $('#set-endpoint').value;
    localStorage.setItem('campus-qa-sid', sid);
    localStorage.setItem('campus-qa-endpoint', ep);
    // 同步到 chat 视图
    const chatSid = $('#student-id'); if (chatSid) chatSid.value = sid;
    const chatEp = $('#endpoint'); if (chatEp) chatEp.value = ep;
    const topSid = $('#topbar-sid'); if (topSid) topSid.textContent = sid;
    state.studentId = sid; state.endpoint = ep;
    const btn = $('#set-save');
    const orig = btn.textContent;
    btn.textContent = '已保存 ✓';
    btn.style.background = 'linear-gradient(180deg,#1a7a1a,#0f5a0f)';
    setTimeout(() => { btn.textContent = orig; btn.style.background = ''; }, 1600);
  });
};

/* ============================================================
   全局事件
   ============================================================ */
function bindGlobal() {
  // 路由
  window.addEventListener('hashchange', handleRoute);

  // 主题按钮
  $('#theme-btn')?.addEventListener('click', toggleTheme);

  // 移动端菜单
  $('#menu-toggle')?.addEventListener('click', () => {
    const nav = $('.side-nav');
    const open = nav.classList.toggle('open');
    $('#menu-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  // 点击主区域关闭移动菜单
  $('.main-area')?.addEventListener('click', () => {
    $('.side-nav')?.classList.remove('open');
    $('#menu-toggle')?.setAttribute('aria-expanded', 'false');
  });
}

/* ============================================================
   启动
   ============================================================ */
function boot() {
  initTheme();
  bindGlobal();
  handleRoute();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
