/**
 * TruthLens — Frontend JavaScript
 * Communicates with the FastAPI backend at API_BASE_URL.
 * Falls back to simulated predictions if the API is unreachable (demo mode).
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API_BASE_URL = 'http://localhost:8000';
const MAX_HISTORY  = 8;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let history   = JSON.parse(localStorage.getItem('tl_history') || '[]');
let isLoading = false;
let lastResult = null;

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
const $  = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

// ---------------------------------------------------------------------------
// API health check
// ---------------------------------------------------------------------------
async function checkApiHealth() {
  try {
    const r = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error('not ok');
    const data = await r.json();
    setStatus(data.model_loaded ? 'online' : 'loading', data.model_loaded ? 'API Online' : 'Loading model…');
  } catch {
    setStatus('offline', 'API Offline (Demo Mode)');
  }
}

function setStatus(state, text) {
  const dot  = $('statusDot');
  const label = $('statusText');
  dot.className = `status-dot ${state}`;
  label.textContent = text;
}

// ---------------------------------------------------------------------------
// Example loaders
// ---------------------------------------------------------------------------
const EXAMPLES = {
  fake: {
    title: 'SHOCKING: Scientists Discover Time Travel is REAL — Government Hiding Truth!',
    text:  'Anonymous whistleblowers from inside the Pentagon have revealed that the US government has secretly developed a working time machine since 1955. The mainstream media refuses to cover this story. Multiple insiders claim that world leaders travel to the future to manipulate stock markets. Share this before they delete it! The deep state does not want you to know. Wake up, sheeple — this affects every one of us. Anonymous sources confirm everything.',
  },
  real: {
    title: 'Federal Reserve raises interest rates by 25 basis points amid cooling inflation',
    text:  'The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday, its tenth increase since March 2022, bringing it to a target range of 5% to 5.25%. Federal Reserve Chair Jerome Powell said the central bank would take a data-driven approach to future decisions, acknowledging that inflation had declined from its peak of 9.1% last year but remained above the central bank's 2% target. The decision was unanimous among the twelve voting members of the Federal Open Market Committee.',
  },
};

function loadExample(type) {
  const ex = EXAMPLES[type];
  $('articleTitle').value = ex.title;
  $('articleText').value  = ex.text;
  updateCharCounter();
}

// ---------------------------------------------------------------------------
// Character counter
// ---------------------------------------------------------------------------
function updateCharCounter() {
  const len = ($('articleText').value || '').length;
  $('charCounter').textContent = `${len} / 2000`;
  $('charCounter').style.color = len > 1800 ? '#f59e0b' : '';
}

$('articleText').addEventListener('input', updateCharCounter);

// ---------------------------------------------------------------------------
// Main analyze function
// ---------------------------------------------------------------------------
async function analyzeArticle() {
  if (isLoading) return;

  const text  = $('articleText').value.trim();
  const title = $('articleTitle').value.trim();

  if (text.length < 10) {
    showToast('⚠️ Please enter at least 10 characters of article text.');
    return;
  }

  setLoading(true);

  try {
    const result = await fetchPrediction(title, text);
    lastResult = result;
    renderResult(result, title, text);
    addToHistory(result, title, text);
  } catch (err) {
    showToast(`❌ ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// API call with demo fallback
// ---------------------------------------------------------------------------
async function fetchPrediction(title, text) {
  try {
    const body = { text };
    if (title) body.title = title;

    const r = await fetch(`${API_BASE_URL}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(30000),
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    return await r.json();
  } catch (networkErr) {
    // Demo mode — deterministic simulation based on heuristics
    console.warn('API unreachable, using demo mode:', networkErr.message);
    return simulatePrediction(title, text);
  }
}

// ---------------------------------------------------------------------------
// Demo simulator (runs client-side when API is offline)
// ---------------------------------------------------------------------------
function simulatePrediction(title, text) {
  const combined = `${title} ${text}`.toLowerCase();

  // Fake news signal words
  const fakeSignals = [
    'shocking', 'bombshell', 'conspiracy', 'secret', 'suppressed', 'hidden', 'wake up',
    'anonymous sources', 'they don\'t want you', 'sheeple', 'deep state', 'whistleblower',
    'government hiding', 'media refuses', 'miracle', 'cure all', 'you won\'t believe',
    'share before', 'deleted', 'real truth',
  ];
  const realSignals = [
    'according to', 'study', 'researchers', 'university', 'published', 'percent',
    'announced', 'report', 'federal', 'officials', 'confirmed', 'statistics',
    'data', 'committee', 'spokesperson', 'analysis',
  ];

  const fakeScore = fakeSignals.filter(w => combined.includes(w)).length;
  const realScore = realSignals.filter(w => combined.includes(w)).length;
  const total = Math.max(fakeScore + realScore, 1);

  let fakeProb = fakeScore / total;
  let realProb = realScore / total;

  // Add small noise
  fakeProb = Math.min(0.97, Math.max(0.03, fakeProb + (Math.random() - 0.5) * 0.1));
  realProb = 1 - fakeProb;

  const label      = fakeProb > 0.5 ? 'FAKE' : 'REAL';
  const confidence = label === 'FAKE' ? fakeProb : realProb;
  const snippet    = text.slice(0, 110) + (text.length > 110 ? '…' : '');
  const tier       = confidence > 0.90 ? 'high' : confidence > 0.75 ? 'moderate' : 'low';

  return {
    label,
    confidence: round4(confidence),
    probabilities: { REAL: round4(realProb), FAKE: round4(fakeProb) },
    explanation: label === 'FAKE'
      ? `[Demo Mode] The model classified this article as **FAKE** with ${tier} confidence (${pct(confidence)}). Sensationalist language and unverified claims detected. Analysed: "${snippet}"`
      : `[Demo Mode] The model classified this article as **REAL** with ${tier} confidence (${pct(confidence)}). Language is consistent with credible reporting. Analysed: "${snippet}"`,
    latency_ms: +(Math.random() * 30 + 15).toFixed(1),
  };
}

const round4 = n => Math.round(n * 10000) / 10000;
const pct    = n => `${(n * 100).toFixed(1)}%`;

// ---------------------------------------------------------------------------
// Render result
// ---------------------------------------------------------------------------
function renderResult(result, title, text) {
  const card   = $('resultCard');
  const isReal = result.label === 'REAL';
  const realP  = result.probabilities.REAL ?? 0;
  const fakeP  = result.probabilities.FAKE ?? 0;
  const conf   = result.confidence;

  // Verdict badge
  $('resultVerdict').innerHTML = `
    <div class="verdict-badge ${isReal ? 'real' : 'fake'}">
      ${isReal ? '✅' : '🚨'} &nbsp;${result.label}
    </div>
    <span class="verdict-confidence">${pct(conf)}</span>
  `;

  // Latency chip
  $('latencyVal').textContent = `${result.latency_ms} ms`;

  // Confidence bar
  setTimeout(() => {
    $('confBarReal').style.width = `${realP * 100}%`;
    $('confBarFake').style.width = `${fakeP * 100}%`;
  }, 50);
  $('confValReal').textContent = pct(realP);
  $('confValFake').textContent = pct(fakeP);

  // Rings
  animateRing('ringRealCircle', realP);
  animateRing('ringFakeCircle', fakeP);
  $('ringRealPct').textContent = pct(realP);
  $('ringFakePct').textContent = pct(fakeP);

  // Explanation — render **bold** markdown
  $('explanationText').innerHTML = result.explanation
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Raw JSON
  $('apiJson').textContent = JSON.stringify(result, null, 2);

  // Show card
  card.classList.remove('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function animateRing(circleId, probability) {
  const circumference = 213.6;
  const offset = circumference - probability * circumference;
  setTimeout(() => {
    $(circleId).style.transition = 'stroke-dashoffset 0.9s cubic-bezier(0.16,1,0.3,1)';
    $(circleId).style.strokeDashoffset = offset;
  }, 100);
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------
function addToHistory(result, title, text) {
  const item = {
    id:         Date.now(),
    label:      result.label,
    confidence: result.confidence,
    title:      title || text.slice(0, 60),
    excerpt:    text.slice(0, 80) + (text.length > 80 ? '…' : ''),
    result,
    title_full: title,
    text_full:  text,
    ts:         new Date().toLocaleTimeString(),
  };
  history.unshift(item);
  if (history.length > MAX_HISTORY) history.pop();
  localStorage.setItem('tl_history', JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const list  = $('historyList');
  const empty = $('historyEmpty');

  if (history.length === 0) {
    empty.style.display = '';
    return;
  }
  empty.style.display = 'none';

  // Rebuild list (keep empty sentinel)
  const existing = list.querySelectorAll('.history-item');
  existing.forEach(el => el.remove());

  history.forEach(item => {
    const isReal = item.label === 'REAL';
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
      <div class="history-badge ${isReal ? 'real' : 'fake'}">${isReal ? '✅' : '🚨'}</div>
      <div class="history-info">
        <div class="history-title">${escHtml(item.title)}</div>
        <div class="history-excerpt">${escHtml(item.excerpt)}</div>
      </div>
      <div class="history-conf">${pct(item.confidence)} · ${item.ts}</div>
    `;
    div.addEventListener('click', () => {
      $('articleTitle').value = item.title_full || '';
      $('articleText').value  = item.text_full  || '';
      updateCharCounter();
      renderResult(item.result, item.title_full, item.text_full);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    list.appendChild(div);
  });
}

function clearHistory() {
  history = [];
  localStorage.removeItem('tl_history');
  renderHistory();
  $('historyEmpty').style.display = '';
}

// ---------------------------------------------------------------------------
// Loading state
// ---------------------------------------------------------------------------
function setLoading(state) {
  isLoading = state;
  const btn     = $('analyzeBtn');
  const spinner = $('btnSpinner');
  const icon    = btn.querySelector('.btn-icon');
  const text    = btn.querySelector('.btn-text');

  btn.disabled = state;
  spinner.classList.toggle('active', state);
  icon.style.display  = state ? 'none' : '';
  text.textContent    = state ? 'Analyzing…' : 'Analyze Article';
}

// ---------------------------------------------------------------------------
// Toast notification
// ---------------------------------------------------------------------------
function showToast(msg) {
  const existing = qs('.toast');
  if (existing) existing.remove();

  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 30000); // Re-check every 30s
  renderHistory();

  // Smooth nav scroll
  document.querySelectorAll('.nav-pill').forEach(a => {
    a.addEventListener('click', e => {
      document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
      a.classList.add('active');
    });
  });
});
