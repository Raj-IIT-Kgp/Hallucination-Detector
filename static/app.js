// VeriFact frontend: live verification stream, benchmark dashboard, method overview.

const STAGES = [
    { key: 'extract', name: 'Claim extraction' },
    { key: 'queries', name: 'Query generation' },
    { key: 'retrieve', name: 'Evidence retrieval' },
    { key: 'nli', name: 'NLI scoring' },
    { key: 'conformal', name: 'Conformal set' },
    { key: 'rewrite', name: 'Query rewrite', optional: true },
    { key: 'kg', name: 'Knowledge graph', optional: true },
    { key: 'arbiter', name: 'LLM arbiter', optional: true },
    { key: 'report', name: 'Report' },
];
// Stages each claim goes through (claims are verified concurrently); the others run once per input
const PER_CLAIM = ['queries', 'retrieve', 'nli', 'conformal', 'rewrite', 'kg', 'arbiter'];
const isOptional = key => Boolean((STAGES.find(s => s.key === key) || {}).optional);

const VERDICT = {
    supported: { text: 'Supported', icon: '✓', cls: 's-good' },
    contradicted: { text: 'Hallucinated', icon: '✕', cls: 's-critical' },
    unknown: { text: 'Uncertain', icon: '?', cls: 's-warning' },
};

const POLICY = {
    conformal_singleton: {
        tag: 'Settled by NLI',
        text: set => `Decisive evidence: the conformal prediction set is the single label <strong>${set}</strong>, so no LLM call was needed.`,
    },
    contradiction_arbitration: {
        tag: 'LLM review',
        text: () => 'A lone or contested contradiction is unreliable on its own, so the <strong>LLM arbiter</strong> reviewed it against the retrieved evidence (no knowledge graph).',
        needed: 'A lone or contested contradiction is unreliable on its own, so this claim needed review by the <strong>LLM arbiter</strong>.',
    },
    multi_agent_debate: {
        tag: 'Escalated · KG + arbiter',
        text: () => 'The conformal set was ambiguous, so the claim was <strong>escalated</strong>: knowledge-graph evidence was added and the LLM arbiter weighed both sides.',
        needed: 'The conformal set was ambiguous, so this claim needed <strong>escalation</strong> to the knowledge graph and the LLM arbiter.',
    },
    absence_heuristic: {
        tag: 'Absence check',
        text: () => 'No evidence was found. The LLM judged whether the <strong>absence</strong> of such a claim from Wikipedia implies it is false.',
        needed: 'No evidence was found, so the LLM had to judge whether the <strong>absence</strong> of such a claim implies it is false.',
    },
    escalation_required: {
        tag: 'Escalation',
        text: () => 'The conformal set contained more than one label, so the claim required escalation.',
    },
};

const LABEL_ORDER = ['contradicted', 'unknown', 'supported'];   // order of the NLI probability vector

const $ = id => document.getElementById(id);

function escapeHtml(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// ---------------------------------------------------------------- theme

function storedTheme() {
    try { return localStorage.getItem('verifact-theme'); } catch { return null; }
}

function applyTheme(theme) {
    const root = document.documentElement;
    if (theme) root.dataset.theme = theme; else delete root.dataset.theme;
    const dark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.themeEffective = dark ? 'dark' : 'light';
}

applyTheme(storedTheme());
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => applyTheme(storedTheme()));

// ---------------------------------------------------------------- tooltip

const tip = $('vizTip');

function showTip(html, event) {
    tip.innerHTML = html;
    tip.classList.add('visible');
    moveTip(event);
}

function moveTip(event) {
    const pad = 14;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = event.clientX + pad, y = event.clientY + pad;
    if (x + w > window.innerWidth - 8) x = event.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = event.clientY - h - pad;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
}

function hideTip() { tip.classList.remove('visible'); }

// ---------------------------------------------------------------- tabs

let benchLoaded = false;

function showView(name) {
    document.querySelectorAll('.tab').forEach(t => {
        const on = t.dataset.view === name;
        t.classList.toggle('active', on);
        t.setAttribute('aria-selected', on);
    });
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
    if (name === 'benchmarks' && !benchLoaded) loadBenchmarks();
    if (name === 'docs' && !docsLoaded && !docsPromise) docsPromise = loadDocs();
    if (name === 'method') loadCascade();
    try { history.replaceState(null, '', `#${name}`); } catch { /* sandboxed */ }
}

// ================================================================ VERIFY

const run = {
    claims: [],
    results: [],
    claimStages: [],      // per claim: { state: {stage: pending|active|done|skipped}, visited: Set }
    claimStart: [],       // per claim: time of its first event (ms since the request started)
    segments: [],         // per claim: [{stage, start}] for the timeline
    claimEnd: [],         // per claim: time of its result
    claimsAt: null,       // time the claims were extracted
    lastMs: 0,            // time of the latest event
    replay: null,         // {recorded_at, duration_sec} when a recorded run is replayed
    text: '',             // the input text
    decompRoute: null,    // how the text was decomposed: fast_path | llm | fallback (null: not sent)
    sentOwner: [],        // per claim: index of the input sentence it came from
    globalState: {},      // extract and report, which run once per input
    start: 0,
    timer: null,
    logCount: 0,
};

function buildStepper() {
    $('stepper').innerHTML = STAGES.map((s, i) => `
        <li class="step${s.optional ? ' optional' : ''}" data-stage="${s.key}">
            <span class="step-dot">${i + 1}</span>
            <span class="step-name">${s.name}</span>
            <span class="step-count"></span>
        </li>`).join('');
    run.globalState = { extract: 'pending', report: 'pending' };
    run.claimStages = [];
    paintStepper();
}

// Combined state of a stage across concurrently verified claims
function stageState(key) {
    if (!PER_CLAIM.includes(key)) return run.globalState[key] || 'pending';
    const states = run.claimStages.map(c => c.state[key]);
    if (!states.length) return 'pending';
    if (states.includes('active')) return 'active';
    if (states.every(s => s === 'done' || s === 'skipped')) return states.includes('done') ? 'done' : 'skipped';
    return 'pending';
}

function paintStepper() {
    const n = run.claimStages.length;
    document.querySelectorAll('.step').forEach(el => {
        const key = el.dataset.stage;
        const st = stageState(key);
        el.classList.remove('pending', 'active', 'done', 'skipped');
        el.classList.add(st);
        el.querySelector('.step-dot').textContent = st === 'done' ? '✓' : STAGES.findIndex(s => s.key === key) + 1;
        // with several claims, show how many have reached this stage
        const reached = PER_CLAIM.includes(key) ? run.claimStages.filter(c => c.visited.has(key)).length : 0;
        el.querySelector('.step-count').textContent = n > 1 && reached ? `${reached}/${n}` : '';
    });
}

function newClaimStages() {
    return { state: Object.fromEntries(PER_CLAIM.map(k => [k, 'pending'])), visited: new Set() };
}

function advanceClaim(cs, key) {
    const idx = PER_CLAIM.indexOf(key);
    if (idx < 0) return;
    PER_CLAIM.forEach((k, i) => {
        const st = cs.state[k];
        if (k === key) cs.state[k] = 'active';
        else if (st === 'active') cs.state[k] = 'done';
        else if (i < idx && !isOptional(k) && st === 'pending') cs.state[k] = 'done';
        // a query rewrite loops back to retrieval: later stages run again
        else if (i > idx && !isOptional(k) && st === 'done') cs.state[k] = 'pending';
    });
    cs.visited.add(key);
}

function finishClaim(cs) {
    PER_CLAIM.forEach(k => { cs.state[k] = cs.visited.has(k) || !isOptional(k) ? 'done' : 'skipped'; });
}

// Event time in ms since the request started: the server's own timestamp when present, so a
// replayed run shows its original timing
function eventMs(data) {
    return data && data.t != null ? data.t * 1000 : performance.now() - run.start;
}

function logEvent(message, ms = performance.now() - run.start) {
    const t = (ms / 1000).toFixed(1).padStart(5, ' ');
    const row = document.createElement('div');
    row.className = 'log-row';
    row.innerHTML = `<span class="log-time">${t}s</span><span>${escapeHtml(message)}</span>`;
    $('logBody').appendChild(row);
    $('logBody').scrollTop = $('logBody').scrollHeight;
    run.logCount += 1;
    $('logCount').textContent = `(${run.logCount})`;
}

function setRunMessage(msg) { $('runMessage').textContent = msg; }

function resetRun() {
    run.claims = [];
    run.results = [];
    run.claimStart = [];
    run.segments = [];
    run.claimEnd = [];
    run.claimsAt = null;
    run.lastMs = 0;
    run.replay = null;
    run.decompRoute = null;
    run.sentOwner = [];
    run.logCount = 0;
    $('replayNote').classList.add('hidden');
    $('timelineSection').classList.add('hidden');
    $('hoodSection').classList.add('hidden');
    run.traces = null;
    $('logBody').innerHTML = '';
    $('logCount').textContent = '(0)';
    $('runClaim').textContent = '';
    $('runTitle').textContent = 'Running pipeline';
    $('liveDot').className = 'live-dot';
    $('errorBanner').classList.add('hidden');
    $('results').classList.add('hidden');
    $('claimsSection').classList.add('hidden');
    $('kgSection').classList.add('hidden');
    $('claimList').innerHTML = '';
    $('decomp').innerHTML = '';
    buildStepper();
    setRunMessage('Connecting to the verification pipeline…');
}

function startTimer() {
    run.start = performance.now();
    clearInterval(run.timer);
    run.timer = setInterval(() => {
        $('runTimer').textContent = `${((performance.now() - run.start) / 1000).toFixed(1)} s`;
    }, 100);
}

function stopTimer() {
    clearInterval(run.timer);
    $('runTimer').textContent = `${((performance.now() - run.start) / 1000).toFixed(1)} s`;
}

// ---- claim list ----------------------------------------------------------

function initClaims(claims) {
    run.claims = claims;
    run.results = new Array(claims.length).fill(null);
    run.claimStages = claims.map(newClaimStages);
    run.claimStart = new Array(claims.length).fill(null);
    run.segments = claims.map(() => []);
    run.claimEnd = new Array(claims.length).fill(null);
    run.globalState.extract = 'done';
    paintStepper();
    $('claimList').innerHTML = claims.map((c, i) => `
        <article class="claim-card" id="claim-${i}">
            <div class="claim-top" role="button" tabindex="0" aria-expanded="false">
                <span class="claim-num">${i + 1}</span>
                <div class="claim-main">
                    <p class="claim-text">${escapeHtml(c)}</p>
                    <div class="claim-tags"><span class="skeleton" style="width:140px"></span></div>
                </div>
                <span class="verdict-badge pending">Queued</span>
                <svg class="chev" viewBox="0 0 24 24" width="18" height="18"><path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            </div>
            <div class="claim-details"></div>
        </article>`).join('');
    $('claimList').querySelectorAll('.claim-top').forEach(top => {
        const toggle = () => {
            const card = top.parentElement;
            const open = card.classList.toggle('open');
            top.setAttribute('aria-expanded', open);
            if (open) animateProbBars(card);
        };
        top.addEventListener('click', toggle);
        top.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    });

    renderDecomp();

    $('statTotal').textContent = claims.length;
    $('results').classList.remove('hidden');
    $('claimsSection').classList.remove('hidden');
    updateSummary();
}

function focusClaim(i) {
    const card = $(`claim-${i}`);
    if (!card) return;
    if (!card.classList.contains('open') && run.results[i]) {
        card.classList.add('open');
        card.querySelector('.claim-top').setAttribute('aria-expanded', true);
        animateProbBars(card);
    }
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1200);
}

function markClaimActive(i) {
    const card = $(`claim-${i}`);
    if (!card || run.results[i]) return;
    const badge = card.querySelector('.verdict-badge');
    badge.className = 'verdict-badge active';
    badge.innerHTML = '<span class="mini-spinner"></span> Verifying';
    renderDecomp();
}

function setClaimStage(i, stageKey) {
    const card = $(`claim-${i}`);
    const cs = run.claimStages[i];
    if (!card || run.results[i] || !cs) return;
    const stage = STAGES.find(s => s.key === stageKey);
    const dots = PER_CLAIM.map(k => `<span class="ms ${cs.state[k]}${isOptional(k) ? ' opt' : ''}" title="${STAGES.find(s => s.key === k).name}"></span>`).join('');
    card.querySelector('.claim-tags').innerHTML =
        `<span class="mini-steps" aria-hidden="true">${dots}</span><span class="tag">${escapeHtml(stage ? stage.name : stageKey)}…</span>`;
}

function argmaxLabel(probs) {
    if (!probs || probs.length !== 3) return null;
    return LABEL_ORDER[probs.indexOf(Math.max(...probs))];
}

function parseEvidence(text) {
    const notes = [];
    if (!text) return { items: [], notes };
    let t = text;
    if (t.startsWith('EPISTEMIC CONTRADICTION')) {
        notes.push('Sources conflict: high-confidence snippets disagree, so the claim was escalated.');
        t = t.replace(/^EPISTEMIC CONTRADICTION:[^\n]*\n*/, '');
    }
    t = t.replace(/\[SYSTEM TIMESTAMP: ([^\]]+)\]\n?/, (_, d) => {
        notes.push(`Time-sensitive claim: evidence was judged as of ${d}.`);
        return '';
    });
    if (t.startsWith('No evidence found')) {
        notes.push(t.trim());
        return { items: [], notes };
    }
    const items = t.split(/\n\n(?=(?:Evidence \d+|KG Evidence \[[^\]]*\]):)/).map(p => {
        const m = p.match(/^(?:Evidence (\d+)|KG Evidence \[([^\]]*)\]):\s*([\s\S]*)$/);
        if (!m) return { kind: 'web', label: 'Evidence', text: p.trim() };
        return m[2] !== undefined
            ? { kind: 'kg', label: `Knowledge graph · ${m[2]}`, text: m[3].trim() }
            : { kind: 'web', label: `Evidence ${m[1]}`, text: m[3].trim(), index: Number(m[1]) - 1 };
    }).filter(x => x.text);
    return { items, notes };
}

function sourceLinks(sourceUrl) {
    if (!sourceUrl) return '';
    const urls = [...new Set(sourceUrl.split(',').map(s => s.trim()).filter(u => /^https?:\/\//.test(u)))];
    return urls.map(u => {
        let host = u;
        try {
            const url = new URL(u);
            host = url.hostname.replace(/^www\./, '');
            if (host.endsWith('wikipedia.org')) host = `Wikipedia · ${decodeURIComponent(url.pathname.split('/').pop()).replace(/_/g, ' ')}`;
        } catch { /* keep raw */ }
        return `<a class="source" href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer">${escapeHtml(host)} ↗</a>`;
    }).join('');
}

function fillClaim(i, result, { quiet = false } = {}) {
    run.results[i] = result;
    const card = $(`claim-${i}`);
    if (!card) return;
    const v = result.verification || {};
    const label = VERDICT[v.label] ? v.label : 'unknown';
    const verdict = VERDICT[label];
    const policy = POLICY[v.final_decision_policy] || { tag: v.final_decision_policy || 'NLI' };
    const set = v.conformal_set || [];
    const llmDown = Boolean(v.llm_unavailable);   // the LLM step failed: Uncertain is not a judgement
    const elapsed = result._elapsed != null ? `<span class="tag">${result._elapsed.toFixed(1)} s</span>` : '';

    card.className = `claim-card ${label}${quiet ? '' : ' reveal'}${card.classList.contains('open') ? ' open' : ''}`;
    card.querySelector('.verdict-badge').className = `verdict-badge ${label}`;
    card.querySelector('.verdict-badge').innerHTML = `<span class="status-icon ${verdict.cls}">${verdict.icon}</span>${verdict.text}${llmDown ? ' · LLM unavailable' : ''}`;
    card.querySelector('.claim-tags').innerHTML = `
        <span class="tag${v.final_decision_policy === 'conformal_singleton' ? ' route-nli' : ''}">${escapeHtml(policy.tag)}</span>
        ${set.length ? `<span class="tag">Conformal set {${set.map(s => VERDICT[s] ? VERDICT[s].text : s).join(', ')}}</span>` : ''}
        ${elapsed}`;

    const details = card.querySelector('.claim-details');
    details.innerHTML = claimDetailsHtml(result, { hoodIndex: i });
    bindClaimDetails(details);
    const hoodLink = card.querySelector('.hood-link');
    if (hoodLink) hoodLink.addEventListener('click', e => {
        e.stopPropagation();
        selectHoodClaim(i);
        $('hoodSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    if (card.classList.contains('open')) animateProbBars(card);

    renderDecomp();
    updateSummary();
}

// The path a claim took through the cascade: NLI reading → conformal set → route → final verdict
function verdictPathHtml(v) {
    const probs = v.probabilities || [];
    const nliLabel = argmaxLabel(probs);
    const set = v.conformal_set || [];
    const label = VERDICT[v.label] ? v.label : 'unknown';
    const policy = POLICY[v.final_decision_policy] || { tag: v.final_decision_policy || 'NLI' };
    const changed = v.debate_resolved && !v.llm_unavailable && nliLabel && nliLabel !== label;
    const setText = set.map(s => VERDICT[s] ? VERDICT[s].text : s).join(', ');
    const steps = [
        ['NLI model', nliLabel ? `${VERDICT[nliLabel].text} ${pct(Math.max(...probs), 0)}` : 'no evidence to score', nliLabel || ''],
        ...(set.length ? [['Conformal set', `{${setText}}`, set.length === 1 ? set[0] : 'multi']] : []),
        ['Route', policy.tag, 'route'],
        ['Final verdict', `${VERDICT[label].icon} ${VERDICT[label].text}`, `final ${label}`],
    ];
    return `<div class="vpath" aria-label="Decision path">${steps.map(([k, val, cls], n) => `
        ${n ? '<span class="vpath-arrow" aria-hidden="true">→</span>' : ''}
        <span class="vpath-step ${cls}"><span class="vpath-k">${k}</span><span class="vpath-v">${escapeHtml(val)}</span></span>`).join('')}
        ${changed ? `<span class="vpath-note">The LLM changed the NLI reading (${VERDICT[nliLabel].text} → ${VERDICT[label].text})</span>` : ''}
    </div>`;
}

// Decision, probabilities and evidence of one verified claim. hoodIndex links to its Under-the-hood view.
function claimDetailsHtml(result, { hoodIndex = null } = {}) {
    const v = result.verification || {};
    const label = VERDICT[v.label] ? v.label : 'unknown';
    const policy = POLICY[v.final_decision_policy] || { tag: v.final_decision_policy || 'NLI', text: () => '' };
    const set = v.conformal_set || [];
    const nliLabel = argmaxLabel(v.probabilities);
    const llmDown = Boolean(v.llm_unavailable);
    const overridden = v.debate_resolved && !llmDown && nliLabel && nliLabel !== label;

    const probs = v.probabilities || [];
    const probRows = probs.length === 3 ? LABEL_ORDER.map((l, k) => `
        <div class="prob-row">
            <span>${VERDICT[l].text}</span>
            <div class="prob-track"><div class="prob-fill ${l}" data-w="${(probs[k] * 100).toFixed(1)}"></div></div>
            <span class="prob-val">${pct(probs[k])}</span>
        </div>`).join('') : '<p class="muted">No NLI scores (decided without evidence).</p>';

    const pv = v.conformal_p_values || {};
    // Conformal p-values are separate per-label tests, not a distribution: they need not sum to 1
    const pvText = LABEL_ORDER.filter(l => pv[l] != null).map(l => `${VERDICT[l].text} ${Number(pv[l]).toFixed(3)}`).join(' · ');
    const { items, notes } = parseEvidence(result.evidence);
    const scores = v.evidence_scores || null;
    const decidedBy = scores ? scores.decided_by : null;
    const decidingPos = items.findIndex(e => e.index != null && e.index === decidedBy);
    const shown = Math.max(3, decidingPos + 1);
    if (decidedBy === 'joint') notes.push('No single snippet was decisive (≥ 80%), so the NLI model read all snippets together.');
    if (scores && v.debate_resolved && !llmDown) notes.push('The bars show the NLI model\'s reading of each snippet; the final verdict came from the LLM arbiter.');
    const evHtml = items.map((e, k) => {
        const sc = scores && e.index != null ? scores.snippets[e.index] : null;
        const deciding = e.index != null && e.index === decidedBy;
        return `
        <div class="ev ${e.kind}${deciding ? ' deciding' : ''}${k >= shown ? ' hidden extra' : ''}">
            <div class="ev-head">
                <span class="ev-label">${escapeHtml(e.label)}${sc ? ` · ${SOURCE_NAME[sc.source] || escapeHtml(sc.source)}` : ''}</span>
                ${deciding ? '<span class="ev-badge">Deciding snippet</span>' : sc && sc.decisive ? '<span class="ev-badge subtle">Decisive</span>' : ''}
                ${sc && /^https?:\/\//.test(sc.url) ? `<a class="ev-src" href="${escapeHtml(sc.url)}" target="_blank" rel="noopener noreferrer">source ↗</a>` : ''}
            </div>
            ${sc ? snippetScoreHtml(sc) : ''}
            <div class="ev-text">${escapeHtml(e.text.length > 600 ? e.text.slice(0, 600) + '…' : e.text)}</div>
        </div>`;
    }).join('');
    const graph = result.graph || {};
    const nNodes = (graph.nodes || []).length, nEdges = (graph.edges || []).length;

    return `
        ${verdictPathHtml(v)}
        <div class="detail-grid">
            <div class="panel">
                <div class="panel-title"><span>How it was decided</span>${hoodIndex != null ? `<button type="button" class="link-btn hood-link" data-i="${hoodIndex}">See the computation ↓</button>` : ''}</div>
                ${llmDown ? `<div class="ev-note" style="margin-bottom:8px"><strong>The LLM step could not run</strong> (${escapeHtml(llmErrorReason(v.llm_unavailable))}), so the verdict was left Uncertain instead of guessed.${nliLabel && probs.length === 3 ? ` Before review, the NLI model leaned <strong>${VERDICT[nliLabel].text}</strong> (${pct(Math.max(...probs), 0)}).` : ''} Try again in a minute.</div>` : ''}
                <p class="decision">${llmDown && policy.needed ? policy.needed : policy.text(set.map(s => VERDICT[s] ? VERDICT[s].text : s).join(', '))}</p>
                ${overridden ? `<p class="decision" style="margin-top:8px">The arbiter <strong>overrode</strong> the NLI model, which leaned <strong>${VERDICT[nliLabel].text}</strong>.</p>` : ''}
                ${nNodes ? `<p class="decision" style="margin-top:8px">Knowledge graph: ${nNodes} entit${nNodes === 1 ? 'y' : 'ies'}, ${nEdges} relation${nEdges === 1 ? '' : 's'} (shown below).</p>` : ''}
            </div>
            <div class="panel">
                <div class="panel-title"><span>NLI probabilities${overridden ? ' (before arbitration)' : ''}</span><span>sum to 1</span></div>
                ${probRows}
                ${pvText ? `
                <div class="panel-title" style="margin-top:14px"><span>Conformal p-values</span><span class="no-case">α = 0.10</span></div>
                <p class="pv">${escapeHtml(pvText)}</p>
                <p class="pv-note">Not probabilities: each label is tested separately against the calibration claims, so they need not sum to 1. A label enters the prediction set when its p-value exceeds α${set.length ? ` — here {${escapeHtml(set.map(s => VERDICT[s] ? VERDICT[s].text : s).join(', '))}}` : ''}${v.conformal_q_hat != null ? ` (q̂ = ${Number(v.conformal_q_hat).toFixed(4)})` : ''}.</p>` : ''}
            </div>
        </div>
        <div class="panel">
            <div class="panel-title"><span>Retrieved evidence${scores ? ' · NLI reading of each snippet' : ''}</span><span>${items.length} snippet${items.length === 1 ? '' : 's'}</span></div>
            ${notes.map(n => `<div class="ev-note" style="margin-bottom:8px">${escapeHtml(n)}</div>`).join('')}
            <div class="evidence-list">${evHtml || '<p class="muted">No evidence text.</p>'}</div>
            ${items.length > shown ? `<button class="more-btn" data-extra="${items.length - shown}">Show ${items.length - shown} more</button>` : ''}
            <div class="sources">${sourceLinks(result.source_url)}</div>
        </div>`;
}

function bindClaimDetails(root) {
    const more = root.querySelector('.more-btn');
    if (more) more.addEventListener('click', e => {
        e.stopPropagation();
        root.querySelectorAll('.ev.extra').forEach(x => x.classList.toggle('hidden'));
        more.textContent = more.textContent.startsWith('Show') ? 'Show fewer' : `Show ${more.dataset.extra} more`;
    });
    root.querySelectorAll('.snip-bar').forEach(bar => {
        bar.addEventListener('mouseenter', e => showTip(bar.dataset.tip, e));
        bar.addEventListener('mousemove', moveTip);
        bar.addEventListener('mouseleave', hideTip);
    });
}

function llmErrorReason(err) {
    const e = String(err || '');
    if (/503|UNAVAILABLE|high demand/i.test(e)) return 'Gemini is overloaded: 503, high demand';
    if (/429|RESOURCE_EXHAUSTED|quota/i.test(e)) return 'Gemini quota or rate limit reached: 429';
    if (/504|DEADLINE|timeout/i.test(e)) return 'Gemini timed out';
    return 'the LLM call failed';
}

const SOURCE_NAME = { wikipedia: 'Wikipedia', web: 'Web', arxiv: 'arXiv' };
const SNIPPET_READING = { supported: 'supports', unknown: 'neutral', contradicted: 'contradicts' };

// One snippet's NLI probabilities as a stacked bar (contradicts / neutral / supports) plus its reading
function snippetScoreHtml(sc) {
    const p = sc.probabilities || [0, 0, 0];
    const top = LABEL_ORDER.indexOf(sc.label);
    const tip = `<div class="tt-title">NLI reading of this snippet</div>` + LABEL_ORDER.map((l, k) =>
        `<div class="tt-row"><span>${SNIPPET_READING[l]}</span><span>${pct(p[k])}</span></div>`).join('');
    return `
        <div class="snip-score">
            <div class="snip-bar" data-tip="${escapeHtml(tip)}">
                ${LABEL_ORDER.map((l, k) => `<span class="snip-seg ${l}" style="flex-grow:${Math.max(p[k], 0.0001)}"></span>`).join('')}
            </div>
            <span class="snip-reading ${sc.label}">${SNIPPET_READING[sc.label] || sc.label} ${top >= 0 ? pct(p[top], 0) : ''}</span>
        </div>`;
}

function animateProbBars(card) {
    requestAnimationFrame(() => card.querySelectorAll('.prob-fill').forEach(f => { f.style.width = `${f.dataset.w}%`; }));
}

function updateSummary() {
    const done = run.results.filter(Boolean);
    const count = l => done.filter(r => (VERDICT[r.verification.label] ? r.verification.label : 'unknown') === l).length;
    const c = { supported: count('supported'), contradicted: count('contradicted'), unknown: count('unknown') };
    const pending = run.claims.length - done.length;
    $('statSupported').textContent = c.supported;
    $('statContradicted').textContent = c.contradicted;
    $('statUnknown').textContent = c.unknown;
    $('verdictCaption').textContent = pending
        ? `${done.length} of ${run.claims.length} claims verified`
        : `Hallucination rate ${pct(run.claims.length ? c.contradicted / run.claims.length : 0, 0)} (contradicted claims / all claims)`;
    renderOverall(c, pending);

    const segs = [...['supported', 'contradicted', 'unknown'].map(l => [l, c[l]]), ['pending', pending]].filter(([, n]) => n > 0);
    const bar = $('verdictBar');
    bar.innerHTML = segs.map(([l, n]) => {
        const name = l === 'pending' ? 'Pending' : VERDICT[l].text;
        return `<div class="verdict-seg ${l}" style="flex-grow:${n}" data-name="${name}" data-n="${n}">${n}</div>`;
    }).join('');
    bar.querySelectorAll('.verdict-seg').forEach(seg => {
        seg.addEventListener('mouseenter', e => showTip(`<div class="tt-title">${seg.dataset.name}</div>
            <div class="tt-row"><span>Claims</span><span>${seg.dataset.n} of ${run.claims.length}</span></div>`, e));
        seg.addEventListener('mousemove', moveTip);
        seg.addEventListener('mouseleave', hideTip);
    });
}

// The answer-level verdict: one contradicted claim makes the whole answer a hallucination
function renderOverall(c, pending) {
    const n = run.claims.length;
    const claimsWord = k => `claim${k === 1 ? '' : 's'}`;
    let state, icon, title, sub;
    if (c.contradicted) {
        state = 'contradicted'; icon = '✕';
        title = `Hallucination detected: ${c.contradicted} of ${n} ${claimsWord(n)} ${c.contradicted === 1 ? 'is' : 'are'} contradicted by evidence`;
        sub = (pending ? `${pending} ${claimsWord(pending)} still being checked. ` : '') + 'One contradicted claim is enough to flag the whole answer.';
    } else if (pending) {
        state = 'pending'; icon = '<span class="mini-spinner"></span>';
        title = `Checking ${n} ${claimsWord(n)}…`;
        sub = `${n - pending} of ${n} verified so far, nothing contradicted yet.`;
    } else if (c.unknown) {
        state = 'unknown'; icon = '?';
        title = `No hallucination found, but ${c.unknown} of ${n} ${claimsWord(n)} could not be verified`;
        sub = 'The evidence was inconclusive, so the detector abstained instead of guessing.';
    } else {
        state = 'supported'; icon = '✓';
        title = n === 1 ? 'No hallucination found: the claim is supported by evidence' : `No hallucination found: all ${n} claims are supported by evidence`;
        sub = 'Every atomic claim was confirmed by the retrieved evidence.';
    }
    $('overallCard').className = `card overall ${state}`;
    $('overallIcon').innerHTML = icon;
    $('overallTitle').textContent = title;
    $('overallSub').textContent = sub;
}

// ---- from text to claims --------------------------------------------------

const STOPWORDS = new Set(('a an the of in on at to is was were are be been being by for and or but with as from that this ' +
    'these those it its he she they him her them his their has have had which who whom not no').split(' '));
const words = t => (String(t).toLowerCase().match(/\w+/g) || []);

function splitSentences(text) {
    if (typeof Intl !== 'undefined' && Intl.Segmenter) {
        return [...new Intl.Segmenter('en', { granularity: 'sentence' }).segment(text)].map(x => x.segment.trim()).filter(Boolean);
    }
    return text.split(/(?<=[.!?])\s+/).map(x => x.trim()).filter(Boolean);
}

// Each claim is assigned to the input sentence that contains most of its content words
function mapClaimsToSentences(sentences, claims) {
    const sentWords = sentences.map(x => new Set(words(x)));
    return claims.map(c => {
        const cw = words(c).filter(w => !STOPWORDS.has(w));
        let best = 0, bestScore = -1;
        sentWords.forEach((sw, k) => {
            const score = cw.length ? cw.filter(w => sw.has(w)).length / cw.length : 0;
            if (score > bestScore) { best = k; bestScore = score; }
        });
        return best;
    });
}

// The claim with the words that are not in its source sentence marked (resolved pronouns, added subjects)
function markAdded(claim, sentence) {
    const have = new Set(words(sentence));
    return String(claim).split(/(\w+)/).map((part, k) => k % 2 && !have.has(part.toLowerCase()) && !STOPWORDS.has(part.toLowerCase())
        ? `<mark class="resolved" title="Not in the original sentence: added by the decomposition">${escapeHtml(part)}</mark>`
        : escapeHtml(part)).join('').replace(/<\/mark>(\s+)<mark[^>]*>/g, '$1');
}

const claimLabel = i => {
    const r = run.results[i];
    if (r) return VERDICT[r.verification.label] ? r.verification.label : 'unknown';
    return run.claimStart[i] != null ? 'active' : 'pending';
};

function decompRoute() {
    if (run.decompRoute) return run.decompRoute;
    // recorded runs from before the route was sent: an unchanged single sentence took the fast path
    return run.claims.length === 1 && run.claims[0].trim() === run.text.trim() ? 'fast_path' : 'llm';
}

function decompSentences() {
    const sentences = splitSentences(run.text || run.claims.join(' '));
    return sentences.length ? sentences : [run.text];
}

function renderDecomp() {
    const box = $('decomp');
    if (!run.claims.length) { box.innerHTML = ''; return; }
    const sentences = decompSentences();
    run.sentOwner = mapClaimsToSentences(sentences, run.claims);
    const rank = ['contradicted', 'unknown', 'active', 'pending', 'supported'];
    const icon = l => VERDICT[l] ? `<span class="status-icon ${VERDICT[l].cls}">${VERDICT[l].icon}</span>`
        : l === 'active' ? '<span class="mini-spinner"></span>' : '<span class="dec-wait"></span>';

    box.innerHTML = sentences.map((sent, k) => {
        const idx = run.claims.map((_, i) => i).filter(i => run.sentOwner[i] === k);
        const worst = rank.find(l => idx.some(i => claimLabel(i) === l)) || 'none';
        return `<div class="dec-row">
            <div class="dec-sent ${idx.length ? worst : 'none'}"><span class="dec-snum">S${k + 1}</span>${escapeHtml(sent)}</div>
            <div class="dec-arrow" aria-hidden="true">${idx.length ? '→' : ''}</div>
            <div class="dec-claims">${idx.length ? idx.map(i => `
                <button type="button" class="dec-claim ${claimLabel(i)}" data-i="${i}">
                    <span class="dec-num">${i + 1}</span>
                    <span class="dec-text">${markAdded(run.claims[i], sent)}</span>
                    ${icon(claimLabel(i))}
                </button>`).join('') : '<span class="muted">No checkable factual claim</span>'}</div>
        </div>`;
    }).join('');
    box.querySelectorAll('.dec-claim').forEach(b => {
        b.addEventListener('click', () => focusClaim(+b.dataset.i));
        b.addEventListener('mouseenter', e => {
            const r = run.results[+b.dataset.i];
            const v = r ? VERDICT[r.verification.label] || VERDICT.unknown : null;
            showTip(`<div class="tt-title">Claim ${+b.dataset.i + 1}</div>
                <div class="tt-row"><span>${v ? `${v.icon} ${v.text}` : 'Not verified yet'}</span>
                ${r ? `<span>${escapeHtml((POLICY[r.verification.final_decision_policy] || {}).tag || '')}</span>` : ''}</div>`, e);
        });
        b.addEventListener('mousemove', moveTip);
        b.addEventListener('mouseleave', hideTip);
    });

    const ns = sentences.length, nc = run.claims.length;
    const counts = `${ns} sentence${ns === 1 ? '' : 's'} → ${nc} atomic claim${nc === 1 ? '' : 's'}`;
    $('decompCaption').textContent = {
        fast_path: `${counts} · fast path: a short sentence with no clauses is already atomic, so no LLM call was made`,
        llm: `${counts} · decomposed by the LLM, with pronouns resolved and negations kept as written`,
        fallback: `${counts} · the LLM was unavailable, so the text was split into sentences`,
    }[decompRoute()] || counts;
}

// ---- streaming ------------------------------------------------------------

function verifiedCount() { return run.results.filter(Boolean).length; }

function handleEvent(data, text) {
    const ms = eventMs(data);
    run.lastMs = Math.max(run.lastMs, ms);
    if (data.status === 'replay') {
        run.replay = data;
        showReplayNote(data);
        return;
    }
    if (data.status === 'progress') {
        const i = data.claim_index;
        const prefix = i != null && run.claims.length > 1 ? `Claim ${i + 1}: ` : '';
        setRunMessage(prefix + data.message);
        logEvent(prefix + data.message, ms);
        if (i != null && run.claimStages[i]) {
            if (run.claimStart[i] == null) {
                run.claimStart[i] = ms;
                markClaimActive(i);
                renderHoodChips();
            }
            if (run.hoodMsg) run.hoodMsg[i] = data.message;
            if (data.data) hoodEvent(i, data.data);
            else if (i === run.hoodClaim) renderHood();   // no step data (older server): still show progress
            if (data.stage) {
                advanceClaim(run.claimStages[i], data.stage);
                setClaimStage(i, data.stage);
                const segs = run.segments[i];
                if (!segs.length || segs[segs.length - 1].stage !== data.stage) segs.push({ stage: data.stage, start: ms });
                renderTimeline();
            }
        } else if (data.stage === 'extract') {
            run.globalState.extract = 'active';
        } else if (data.stage === 'report') {
            run.globalState.extract = 'done';
            run.globalState.report = 'active';
        }
        paintStepper();
    } else if (data.status === 'claims') {
        run.decompRoute = data.decomposition ? data.decomposition.route : null;
        initClaims(data.claims);
        initHood(data.claims);
        run.claimsAt = ms;
        $('runClaim').textContent = `· 0 of ${data.claims.length} verified`;
        logEvent(`Extracted ${data.claims.length} atomic claim${data.claims.length === 1 ? '' : 's'}`, ms);
    } else if (data.status === 'claim_result') {
        const i = data.index;
        const started = run.claimStart[i] ?? 0;
        run.claimEnd[i] = ms;
        fillClaim(i, { ...data.result, _elapsed: (ms - started) / 1000 });
        hoodResult(i, data.result);
        renderTimeline();
        if (run.claimStages[i]) finishClaim(run.claimStages[i]);
        paintStepper();
        $('runClaim').textContent = `· ${verifiedCount()} of ${run.claims.length} verified`;
        const v = VERDICT[data.result.verification.label] || VERDICT.unknown;
        logEvent(`Claim ${i + 1}: ${v.text}`, ms);
    } else if (data.status === 'claim_graph') {
        // display-only graph enrichment that arrives after the verdict
        const r = run.results[data.index];
        if (r) fillClaim(data.index, { ...r, graph: data.graph }, { quiet: true });
    } else if (data.status === 'done') {
        const claims = data.report.claims || [];
        if (!run.claims.length && claims.length) { initClaims(claims.map(c => c.claim)); initHood(run.claims); }   // older server without per-claim events
        claims.forEach((c, i) => { if (!run.results[i]) { fillClaim(i, c); hoodResult(i, c); } });
        if (!claims.length) {
            $('results').classList.add('hidden');
            showError('No verifiable factual claims were found in this text.');
        }
        const llmDown = claims.filter(c => c.verification && c.verification.llm_unavailable);
        if (llmDown.length) {
            showError(`The LLM was unavailable for ${llmDown.length} claim${llmDown.length === 1 ? '' : 's'} (${llmErrorReason(llmDown[0].verification.llm_unavailable)}), ` +
                `so ${llmDown.length === 1 ? 'its verdict is' : 'their verdicts are'} Uncertain rather than judged. This run was not recorded; try again in a minute.`, 'warning');
        }
        run.claimStages.forEach(finishClaim);
        run.globalState.extract = 'done';
        run.globalState.report = 'done';
        paintStepper();
        renderMainGraph();
        renderTimeline();
    } else if (data.status === 'error') {
        throw new Error(data.message);
    }
}

function showError(msg, kind = 'error') {
    const b = $('errorBanner');
    b.textContent = msg;
    b.className = `banner banner-${kind}`;
}

function showReplayNote(data) {
    const when = new Date(data.recorded_at * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    $('replayText').textContent = `Replay of a recorded run · recorded ${when} · the live run took ${Number(data.duration_sec).toFixed(1)} s`;
    $('replayNote').classList.remove('hidden');
    $('runTitle').textContent = 'Replaying recorded run';
    $('liveDot').className = 'live-dot replay';
}

async function errorMessage(response) {
    try {
        const body = await response.json();
        if (body && body.error) return body.error;
    } catch { /* not JSON */ }
    return `Server returned ${response.status} ${response.statusText}`;
}

async function verify(live = false) {
    const text = $('inputText').value.trim();
    if (!text) {
        showError('Enter some text or pick an example first.');
        $('inputText').focus();
        return;
    }
    const btn = $('analyzeBtn');
    btn.disabled = true;
    btn.classList.add('loading');
    btn.querySelector('.btn-label').textContent = 'Verifying…';
    resetRun();
    run.text = text;
    $('runPanel').classList.remove('hidden');
    startTimer();

    try {
        const response = await fetch('/api/analyze_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, forceKG: $('forceKgToggle').checked, live }),
        });
        if (!response.ok) throw new Error(await errorMessage(response));

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.trim()) handleEvent(JSON.parse(line), text);
            }
        }
        if (buffer.trim()) handleEvent(JSON.parse(buffer), text);
        $('runTitle').textContent = run.replay ? 'Replay complete' : 'Verification complete';
        $('runClaim').textContent = '';
        $('liveDot').className = run.replay ? 'live-dot replay done' : 'live-dot done';
        setRunMessage(`Verified ${run.claims.length} claim${run.claims.length === 1 ? '' : 's'}.`);
    } catch (err) {
        $('runTitle').textContent = 'Verification failed';
        $('liveDot').className = 'live-dot failed';
        showError(err.message.startsWith('Rate limit') || err.message.startsWith('The server is busy') || err.message.startsWith('Text is too long')
            ? err.message : `Verification failed: ${err.message}`);
        logEvent(`Error: ${err.message}`);
    } finally {
        stopTimer();
        if (run.replay) $('runTimer').textContent += ` · live ${Number(run.replay.duration_sec).toFixed(1)} s`;
        loadSamples();   // a completed live run of an example becomes replayable
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.querySelector('.btn-label').textContent = 'Verify';
    }
}

// ---- under the hood ----------------------------------------------------------
// Renders the per-claim computation trace (verification.trace, also streamed live as progress
// events with data) as step cards with the formulas of pipeline.md filled in with real numbers.

const LBL = { contradicted: '\\text{con}', unknown: '\\text{unk}', supported: '\\text{sup}' };
const READING = { contradicted: 'contradicts', unknown: 'neutral', supported: 'supports' };
const CASES = [
    ['single', 'Only one snippet', 'it is scored on its own; a decisive contradiction from one snippet goes to LLM review'],
    ['conflict', 'Sources conflict', 'decisive snippets disagree (S and K both ≥ 0.85, within 0.12): UNKNOWN, the LLM arbiter decides'],
    ['corroborated', 'Corroborated contradiction', `at least m decisive contradictions and no decisive support: CONTRADICTED, no LLM call`],
    ['review', 'Lone or contested contradiction', 'unreliable on its own: the NLI posterior is kept and the LLM reviews it'],
    ['decisive', 'Decisive snippet', 'a snippet at ≥ τ settles the claim; the strongest one gives the verdict'],
    ['joint', 'No decisive snippet', 'the snippets are concatenated and scored together'],
];
const NEXT = {
    accept: 'NLI verdict stands (no LLM call)', review: 'LLM review of the contradiction',
    escalate: 'Knowledge graph + LLM arbiter', rewrite: 'Rewrite the queries and search again',
};

function tex(src, display = true) {
    if (typeof katex === 'undefined') return `<code>${escapeHtml(src)}</code>`;
    try { return katex.renderToString(src, { displayMode: display, throwOnError: false }); } catch { return `<code>${escapeHtml(src)}</code>`; }
}
const fx = (x, d = 3) => Number(x).toFixed(d);
const tf = b => (b ? '\\textcolor{#0ca30c}{\\text{true}}' : '\\textcolor{#898781}{\\text{false}}');

function hoodStep(num, title, tag, body) {
    return `<li class="hood-step enter"><div class="hood-num">${num}</div><div class="hood-body">
        <div class="hood-head"><h3>${title}</h3>${tag ? `<span class="hood-tag">${tag}</span>` : ''}</div>${body}</div></li>`;
}

function probBar(p) {
    return `<span class="snip-bar inline">${LABEL_ORDER.map((l, k) =>
        `<span class="snip-seg ${l}" style="flex-grow:${Math.max(p[k], 0.0001)}"></span>`).join('')}</span>`;
}

const HOOD_RENDER = {
    queries: st => hoodStep('Q', 'Query generation', st.route === 'fast_path' ? 'rule-based fast path · no LLM call' : st.route === 'llm' ? 'LLM · light tier' : 'fallback: the claim itself', `
        <div class="hood-chipline">${st.queries.map(q => `<span class="q-chip">${escapeHtml(q)}</span>`).join('')}</div>
        <p class="hood-note">Domain <strong>${escapeHtml(st.domain)}</strong> · time-sensitive: <strong>${st.temporal ? 'yes, evidence is read with today\'s date' : 'no'}</strong>. Each query is searched on Wikipedia${st.domain === 'scientific' ? ' and arXiv' : ''}; the web is searched once with the full claim.</p>`),

    retrieval: st => {
        const top = st.top || [];
        const first = top.find(t => t.selected) || top[0];
        const sources = Object.entries(st.by_source || {}).map(([k, v]) => `${v} ${SOURCE_NAME[k] || k}`).join(' · ');
        const rows = top.map((t, i) => `
            <tr class="${t.selected ? 'sel' : ''}">
                <td>${i + 1}</td><td>${SOURCE_NAME[t.source] || t.source}</td>
                <td class="hood-txt" title="${escapeHtml(t.text)}">${escapeHtml(t.text.slice(0, 70))}…</td>
                <td><span class="cos-bar"><span style="width:${Math.max(0, Math.min(1, t.dense)) * 100}%"></span></span>${fx(t.dense)}</td>
                <td>${fx(t.bm25, 2)}</td><td>${t.dense_rank}</td><td>${t.bm25_rank}</td><td>${fx(t.rrf, 5)}</td>
                <td>${t.selected ? '✓' : ''}</td>
            </tr>`).join('');
        const math = first ? tex(`\\text{RRF}(d_1) = \\frac{1}{60 + r_{\\text{dense}}} + \\frac{1}{60 + r_{\\text{bm25}}} = \\frac{1}{60 + ${first.dense_rank}} + \\frac{1}{60 + ${first.bm25_rank}} = ${fx(1 / (60 + first.dense_rank), 5)} + ${fx(1 / (60 + first.bm25_rank), 5)} = ${fx(1 / (60 + first.dense_rank) + 1 / (60 + first.bm25_rank), 5)}`)
            + tex(`\\text{keep}(d_1) \\iff \\underbrace{${fx(first.dense)}}_{\\cos(\\mathbf c, \\mathbf e)} \\ge 0.20 \\;\\lor\\; \\underbrace{${fx(first.bm25, 2)}}_{\\text{BM25}} > 1.0 \\;\\Rightarrow\\; ${tf(first.dense >= 0.2 || first.bm25 > 1)}`) : '';
        return hoodStep('R', `Retrieval and hybrid re-ranking${st.pass > 1 ? ` · pass ${st.pass}` : ''}`, `${st.candidates} candidates`, `
            <p class="hood-note">${sources}; average length ${st.avgdl} tokens. Each paragraph gets a dense score (cosine similarity of all-MiniLM-L6-v2 embeddings) and a sparse score (BM25), and the two <em>ranks</em> are fused:</p>
            <div class="table-wrap"><table class="data hood-table">
                <thead><tr><th>#</th><th>Source</th><th>Paragraph</th><th>Cosine</th><th>BM25</th><th>r<sub>dense</sub></th><th>r<sub>bm25</sub></th><th>RRF</th><th>Kept</th></tr></thead>
                <tbody>${rows}</tbody></table></div>
            <div class="hood-math">${math}</div>
            <p class="hood-note">Kept ${st.selected} of ${st.candidates}${st.echo_filtered ? `; ${st.echo_filtered} web snippet${st.echo_filtered === 1 ? '' : 's'} removed for merely quoting the claim` : ''}; ${st.kept} snippet${st.kept === 1 ? '' : 's'} go to the NLI model.</p>`);
    },

    nli: st => {
        const sn = st.snippets || [];
        const focus = Number.isInteger(st.decided_by) ? st.decided_by : 0;
        const fz = sn[focus] && sn[focus].logits;
        const rows = sn.map((s, i) => `
            <tr class="${i === st.decided_by ? 'sel' : ''}">
                <td>${i + 1}</td><td>${SOURCE_NAME[s.source] || s.source}</td>
                <td class="mono">${s.logits ? `(${s.logits.map(z => fx(z, 2)).join(', ')})` : '—'}</td>
                <td>${probBar(s.probabilities)}</td>
                <td><span class="snip-reading ${s.label}">${READING[s.label]} ${pct(Math.max(...s.probabilities), 1)}</span></td>
                <td>${s.decisive ? (i === st.decided_by ? '<span class="ev-badge">deciding</span>' : '<span class="ev-badge subtle">decisive</span>') : ''}</td>
            </tr>`).join('');
        let softmax = '';
        if (fz) {
            const k = LABEL_ORDER.indexOf(sn[focus].label);
            const e = fz.map(z => Math.exp(z)), sum = e.reduce((a, b) => a + b, 0);
            softmax = tex(`\\hat\\pi_{${LBL[sn[focus].label]}}(e_${focus + 1}) = \\frac{e^{z_{${LBL[sn[focus].label]}}}}{\\sum_y e^{z_y}} = \\frac{e^{${fx(fz[k], 2)}}}{e^{${fx(fz[0], 2)}} + e^{${fx(fz[1], 2)}} + e^{${fx(fz[2], 2)}}} = ${fx(e[k] / sum, 4)}`);
        }
        const conflict = st.S >= 0.85 && st.K >= 0.85 && Math.abs(st.S - st.K) < 0.12;
        const cases = CASES.map(([key, name, desc]) => `
            <li class="${key === st.case ? 'fired' : ''}"><strong>${name}</strong> — ${desc}${key === st.case ? ' <span class="ev-badge">this one</span>' : ''}</li>`).join('');
        return hoodStep('N', 'NLI scoring and the decision rule', 'BART-large-MNLI · one forward pass per snippet', `
            <p class="hood-note">Each snippet is the premise and the claim the hypothesis; the model outputs three logits <em>z</em> = (contradiction, neutral, entailment), turned into probabilities by softmax:</p>
            <div class="table-wrap"><table class="data hood-table">
                <thead><tr><th>Snippet</th><th>Source</th><th>Logits z</th><th>Probabilities</th><th>Reading</th><th></th></tr></thead>
                <tbody>${rows}</tbody></table></div>
            <div class="hood-math">${softmax}
                ${tex(`S = ${fx(st.S)},\\quad K = ${fx(st.K)} \\;\\Rightarrow\\; \\text{conflict} = (S \\ge 0.85) \\land (K \\ge 0.85) \\land (|S-K| < 0.12) = ${tf(conflict)}`)}
                ${tex(`|D_{\\text{sup}}| = ${st.n_sup},\\quad |D_{\\text{con}}| = ${st.n_con} \\qquad (\\text{decisive} \\iff \\hat\\pi \\ge \\tau = ${st.tau}${st.m != null ? `,\\; m = ${st.m}` : ''})`)}
            </div>
            <ul class="hood-cases">${cases}</ul>`);
    },

    conformal: st => {
        const p = st.probabilities || [0, 1, 0], q = st.q_hat, n = st.n || 100;
        const k = Math.ceil((n + 1) * 0.9);
        const rows = LABEL_ORDER.map((l, i) => {
            const s = 1 - p[i], inSet = (st.set || []).includes(l), pv = (st.p_values || {})[l];
            return `<tr class="${inSet ? 'sel' : ''}"><td>${VERDICT[l].text}</td><td>${fx(p[i], 4)}</td><td>${fx(s, 4)}</td>
                <td>${s <= q ? '≤' : '>'} ${fx(q, 4)}</td><td>${pv != null ? fx(pv, 4) : '—'}</td><td>${inSet ? '✓ in set' : ''}</td></tr>`;
        }).join('');
        const top = LABEL_ORDER[p.indexOf(Math.max(...p))];
        const pv = (st.p_values || {})[top];
        const count = pv != null ? Math.round(pv * (n + 1)) - 1 : null;
        return hoodStep('C', 'Conformal prediction set', `calibrated on n = ${n} claims · α = 0.10`, `
            <p class="hood-note">Non-conformity score s<sub>y</sub> = 1 − π<sub>y</sub>; a label enters the set when its score is no larger than the calibrated quantile q̂:</p>
            <div class="hood-math">${tex(`\\hat q = s_{(\\lceil (n+1)(1-\\alpha) \\rceil)} = s_{(${k})} = ${fx(q, 4)} \\qquad \\mathcal C(X) = \\{\\, y : 1 - \\hat\\pi_y \\le \\hat q \\,\\}`)}</div>
            <div class="table-wrap"><table class="data hood-table">
                <thead><tr><th>Label y</th><th>π<sub>y</sub></th><th>s<sub>y</sub> = 1 − π<sub>y</sub></th><th>vs q̂</th><th>p-value</th><th></th></tr></thead>
                <tbody>${rows}</tbody></table></div>
            <div class="calib-plot" data-p="${p.join(',')}" data-q="${q}"></div>
            <p class="hood-note">Bars: the scores of the ${n} calibration claims, i.e. how surprised the NLI model was by their true label. q̂ is the score that 90% of them stay below, so a label left of the line is no more surprising than a typical true label. Hover a marker for its p-value.</p>
            <div class="hood-math">${count != null ? tex(`p(${LBL[top]}) = \\frac{1 + \\#\\{\\, i : s_i \\ge ${fx(1 - Math.max(...p), 4)} \\,\\}}{n + 1} = \\frac{1 + ${count}}{${n + 1}} = ${fx(pv, 4)}`) : ''}
                ${tex(`\\mathcal C(X) = \\{ ${(st.set || []).map(l => LBL[l]).join(',\\, ')} \\}`)}</div>`);
    },

    routing: st => {
        const t1 = st.label === 'unknown', t2 = st.set_size > 1, t3 = st.unknown_in_set, t4 = st.shared_role && st.label === 'contradicted';
        return hoodStep('→', 'Routing', escapeHtml(NEXT[st.next] || st.next), `
            <div class="hood-math">${tex(`\\text{escalate} = \\underbrace{[\\hat y = \\text{unk}]}_{${tf(t1)}} \\lor \\underbrace{[|\\mathcal C| > 1]}_{${tf(t2)}} \\lor \\underbrace{[\\text{unk} \\in \\mathcal C]}_{${tf(t3)}} \\lor \\underbrace{[\\text{shared role} \\land \\hat y = \\text{con}]}_{${tf(t4)}} = ${tf(st.escalate)}`)}</div>
            <p class="hood-note">${st.next === 'review' ? 'Not escalated, but the evidence rule flagged a lone or contested contradiction, so the LLM reviews it (no knowledge graph).' :
                st.next === 'rewrite' ? 'The verdict is uncertain on the first pass, so the LLM rewrites the queries and retrieval runs again.' :
                st.next === 'escalate' ? 'The set is ambiguous: the knowledge graph adds multi-hop evidence and the LLM arbiter decides.' :
                'A single confident label: the NLI verdict is final and no LLM is called.'}</p>`);
    },

    rewrite: st => hoodStep('↺', 'Self-reflective query rewrite', `after ${escapeHtml(st.reason)}`, `
        <div class="hood-chipline">${(st.queries || []).map(q => `<span class="q-chip">${escapeHtml(q)}</span>`).join('') || '<span class="muted">no new queries</span>'}</div>
        <p class="hood-note">Evidence from both passes is pooled, so a worse second search cannot discard what the first one found.</p>`),

    kg: st => hoodStep('G', 'Knowledge graph', `${st.nodes} entit${st.nodes === 1 ? 'y' : 'ies'} · ${st.edges} relation${st.edges === 1 ? '' : 's'} (${st.verified_edges} verified)`, `
        <p class="hood-note">Relations are kept as evidence only if their quote is found verbatim in a source (provenance check). Added to the arbiter's evidence by similarity to the claim:</p>
        ${(st.added || []).map(a => `<div class="ev kg"><span class="ev-label">${escapeHtml(a.node)}</span>${escapeHtml(a.text)}…</div>`).join('') || '<p class="muted">Nothing relevant enough was added.</p>'}`),

    arbiter: st => {
        const head = st.mode === 'review' ? 'LLM review' : 'LLM arbiter';
        if (st.error) return hoodStep('L', head, 'could not run', `<div class="ev-note">The LLM could not be reached (${escapeHtml(llmErrorReason(st.error))}), so the verdict was left Uncertain instead of guessed.</div>`);
        const text = st.reasoning || '';
        const parts = text.split(/\[PERSPECTIVE ([ABC])[^\]]*\]/i);
        const fmt = t => escapeHtml(t.trim()).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n+/g, '<br>');
        let body;
        if (parts.length >= 7) {
            const names = { A: ['Defense', 'argues the claim is supported', 'good'], B: ['Prosecution', 'argues the claim is contradicted', 'critical'], C: ['Judgement', 'weighs both against the evidence', 'accent'] };
            body = '<div class="debate">' + [1, 3, 5].map(i => {
                const [name, sub, cls] = names[parts[i].toUpperCase()] || [parts[i], '', 'accent'];
                return `<div class="debate-col ${cls}"><div class="debate-head">${name}<span>${sub}</span></div><div class="debate-text">${fmt(parts[i + 1])}</div></div>`;
            }).join('') + '</div>';
        } else {
            body = `<div class="debate-text single">${fmt(text) || '<span class="muted">No reasoning text.</span>'}</div>`;
        }
        const v = VERDICT[st.label] || VERDICT.unknown;
        return hoodStep('L', head, `${st.mode === 'review' ? 'one call · no knowledge graph' : 'dialectical: defense, prosecution, judgement'}`, `
            <p class="hood-note">The model's actual written reasoning; the verdict is read from its final <code>VERDICT:</code> line.</p>
            ${body}
            <p class="hood-verdict">Verdict: <span class="verdict-badge ${st.label}"><span class="status-icon ${v.cls}">${v.icon}</span>${v.text}</span></p>`);
    },

    absence: st => hoodStep('∅', 'Absence evaluation', 'no evidence after the rewrite', st.error
        ? `<div class="ev-note">The LLM could not be reached (${escapeHtml(llmErrorReason(st.error))}).</div>`
        : `<p class="hood-note">The LLM judged whether such a claim would certainly be documented if true. Answer: <strong>${st.label === 'contradicted' ? 'yes, so its absence makes it false' : 'not necessarily, so it stays uncertain'}</strong>.</p>`),
};

function initHood(claims) {
    run.traces = claims.map(() => []);
    run.hoodMsg = claims.map(() => '');
    run.hoodClaim = 0;
    run.hoodShown = 0;
    $('hoodSection').classList.remove('hidden');
    renderHoodChips();
    renderHood();
}

function renderHoodChips() {
    $('hoodChips').innerHTML = run.claims.map((c, i) => {
        const r = run.results[i];
        const state = r ? (VERDICT[r.verification.label] ? r.verification.label : 'unknown') : run.claimStart[i] != null ? 'active' : 'pending';
        return `<button type="button" class="hood-chip ${i === run.hoodClaim ? 'active' : ''}" data-i="${i}" title="${escapeHtml(c)}">
            <span class="hood-dot ${state}"></span>Claim ${i + 1}</button>`;
    }).join('');
    $('hoodChips').querySelectorAll('.hood-chip').forEach(b => b.addEventListener('click', () => selectHoodClaim(+b.dataset.i)));
}

function selectHoodClaim(i) {
    run.hoodClaim = i;
    run.hoodShown = 0;
    renderHoodChips();
    renderHood();
}

// Appends the steps not shown yet (so new steps animate in while earlier ones stay put)
function renderHood() {
    const i = run.hoodClaim;
    const steps = (run.traces && run.traces[i]) || [];
    $('hoodClaim').innerHTML = `<span class="muted">Claim ${i + 1}:</span> ${escapeHtml(run.claims[i] || '')}`;
    let html = '';
    if (run.hoodShown === 0) {
        $('hoodSteps').innerHTML = '';
        html = decompStepHtml(i);
    }
    for (let k = run.hoodShown; k < steps.length; k++) {
        const fn = HOOD_RENDER[steps[k].kind];
        if (fn) html += fn(steps[k]);
    }
    run.hoodShown = steps.length;
    const r = run.results[i];
    $('hoodSteps').querySelectorAll('.hood-final, .hood-waiting').forEach(x => x.remove());
    $('hoodSteps').insertAdjacentHTML('beforeend', html);
    if (r) {
        const v = VERDICT[r.verification.label] || VERDICT.unknown;
        $('hoodSteps').insertAdjacentHTML('beforeend', `<li class="hood-step hood-final"><div class="hood-num">✓</div><div class="hood-body">
            <div class="hood-head"><h3>Final verdict</h3></div>
            <span class="verdict-badge ${r.verification.label}"><span class="status-icon ${v.cls}">${v.icon}</span>${v.text}</span>
            ${!steps.length ? `<p class="hood-note">${run.replay
                ? 'This recorded run was made before the computation trace existed. Use <em>Run live instead</em> to see every step.'
                : 'The server sent no computation trace with this result. It is probably running an older version of the code: restart it (Ctrl+C, then <code>python server.py</code>) and check the claim again.'}</p>` : ''}</div></li>`);
    } else {
        const msg = run.hoodMsg && run.hoodMsg[i];
        $('hoodSteps').insertAdjacentHTML('beforeend', `<li class="hood-step hood-waiting"><div class="hood-num"><span class="mini-spinner"></span></div>
            <div class="hood-body"><div class="hood-head"><h3>${run.claimStart[i] != null ? 'Working…' : 'Waiting to start…'}</h3></div>
            ${msg ? `<p class="hood-note mono">${escapeHtml(msg)}</p>` : ''}</div></li>`);
    }
    requestAnimationFrame(() => $('hoodSteps').querySelectorAll('.hood-step.enter').forEach(x => x.classList.remove('enter')));
    drawCalibPlots($('hoodSteps'));
}

// Where the selected claim came from: its sentence of the input, and how it was split off
function decompStepHtml(i) {
    const sentences = decompSentences();
    const k = run.sentOwner[i] ?? 0;
    const sent = sentences[k] || run.text;
    const siblings = run.sentOwner.filter(x => x === k).length;
    const route = decompRoute();
    const tag = { fast_path: 'fast path · no LLM call', llm: 'LLM · heavy tier', fallback: 'NLTK sentence split (LLM unavailable)' }[route];
    return hoodStep('D', 'Claim decomposition', tag, `
        <p class="hood-note">Sentence ${k + 1} of the input${route === 'fast_path' ? ' is a single short sentence without clause markers, so it is already atomic and is verified as it stands' : ''}:</p>
        <div class="hood-quote">${escapeHtml(sent)}</div>
        ${route === 'fast_path' ? '' : `<p class="hood-note">became claim ${i + 1}${siblings > 1 ? `, one of ${siblings} atomic claims split from this sentence` : ''}. Underlined words are not in the sentence: the LLM added them to make the claim self-contained.</p>
        <div class="hood-quote claim">${markAdded(run.claims[i], sent)}</div>`}`);
}

// ---- conformal calibration plot ----------------------------------------------

let calibPromise = null;

function fetchCalibration() {
    if (!calibPromise) {
        calibPromise = fetch('/api/calibration').then(res => {
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            return res.json();
        });
        calibPromise.catch(() => { calibPromise = null; });
    }
    return calibPromise;
}

async function drawCalibPlots(root) {
    const boxes = [...root.querySelectorAll('.calib-plot:not(.drawn)')];
    if (!boxes.length) return;
    let cal;
    try { cal = await fetchCalibration(); } catch { boxes.forEach(b => b.remove()); return; }
    boxes.forEach(b => {
        if (b.classList.contains('drawn')) return;
        b.classList.add('drawn');
        b.innerHTML = calibSvg(cal.scores, b.dataset.p.split(',').map(Number), Number(b.dataset.q));
        b.querySelectorAll('[data-tip]').forEach(el => {
            el.addEventListener('mouseenter', e => showTip(el.dataset.tip, e));
            el.addEventListener('mousemove', moveTip);
            el.addEventListener('mouseleave', hideTip);
        });
    });
}

// Histogram of the calibration non-conformity scores with q̂, and where this claim's three labels fall:
// a label enters the prediction set when its score s = 1 − π is left of q̂
function calibSvg(scores, probs, q) {
    const W = 660, H = 196, m = { l: 14, r: 14, t: 66, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const x = v => m.l + Math.max(0, Math.min(1, v)) * iw;
    const bins = 20, counts = new Array(bins).fill(0);
    scores.forEach(v => { counts[Math.min(bins - 1, Math.floor(v * bins))] += 1; });
    const ymax = Math.max(...counts, 1);
    const y = c => m.t + ih - (c / ymax) * ih;
    const n = scores.length;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration scores, the conformal threshold and this claim's label scores">`;
    svg += `<rect class="calib-zone" x="${x(0)}" y="${m.t}" width="${x(q) - x(0)}" height="${ih}"></rect>`;
    svg += `<text class="calib-zone-label" x="${(x(0) + x(q)) / 2}" y="${m.t + 13}" text-anchor="middle">enters the set (s ≤ q̂)</text>`;
    svg += `<text class="calib-zone-label" x="${x(1) - 4}" y="${m.t + 13}" text-anchor="end">excluded</text>`;
    counts.forEach((c, k) => {
        if (!c) return;
        const x0 = x(k / bins) + 1, w = iw / bins - 2;
        const tip = `<div class="tt-title">Scores ${(k / bins).toFixed(2)}–${((k + 1) / bins).toFixed(2)}</div><div class="tt-row"><span>Calibration claims</span><span>${c} of ${n}</span></div>`;
        svg += `<rect class="calib-bar" x="${x0}" y="${y(c)}" width="${w}" height="${m.t + ih - y(c)}" rx="2" data-tip="${escapeHtml(tip)}"></rect>`;
    });
    svg += `<line class="axis-line" x1="${x(0)}" x2="${x(1)}" y1="${m.t + ih}" y2="${m.t + ih}"></line>`;
    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
        svg += `<text class="tick" x="${x(t)}" y="${m.t + ih + 14}" text-anchor="${t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}">${t}</text>`;
    });
    svg += `<text class="tick" x="${x(0.5)}" y="${H - 4}" text-anchor="middle">non-conformity score s = 1 − π(label)</text>`;
    svg += `<line class="calib-q" x1="${x(q)}" x2="${x(q)}" y1="${m.t - 6}" y2="${m.t + ih}"></line>`;
    svg += `<text class="calib-q-label" x="${x(q) - 4}" y="${m.t + ih - 6}" text-anchor="end">q̂ = ${q.toFixed(3)}</text>`;

    // one row per label above the histogram, so markers at similar scores never overlap
    LABEL_ORDER.forEach((l, k) => {
        const sc = 1 - probs[k], cx = x(sc), rowY = 12 + k * 17;
        const inSet = sc <= q;
        const pv = (1 + scores.filter(v => v >= sc).length) / (n + 1);
        const right = cx > m.l + iw * 0.6;
        const tip = `<div class="tt-title">${VERDICT[l].text}</div>
            <div class="tt-row"><span>π (NLI probability)</span><span>${probs[k].toFixed(4)}</span></div>
            <div class="tt-row"><span>s = 1 − π</span><span>${sc.toFixed(4)}</span></div>
            <div class="tt-row"><span>vs q̂</span><span>${inSet ? '≤' : '>'} ${q.toFixed(4)}</span></div>
            <div class="tt-row"><span>p-value</span><span>${pv.toFixed(4)}</span></div>
            <div class="tt-row"><span>${inSet ? '✓ in the prediction set' : 'not in the set'}</span></div>`;
        svg += `<g class="calib-mark ${l}${inSet ? ' in' : ''}" data-tip="${escapeHtml(tip)}">
            <line x1="${cx}" x2="${cx}" y1="${rowY}" y2="${m.t + ih}"></line>
            <circle cx="${cx}" cy="${rowY}" r="5"></circle>
            <text x="${right ? cx - 9 : cx + 9}" y="${rowY}" dy="0.35em" text-anchor="${right ? 'end' : 'start'}">${VERDICT[l].text} · s = ${sc.toFixed(3)}${inSet ? ' · in set' : ''}</text>
        </g>`;
    });
    return svg + '</svg>';
}

function hoodEvent(i, data) {
    if (!run.traces || !run.traces[i]) return;
    run.traces[i].push(data);
    if (i === run.hoodClaim) renderHood();
}

function hoodResult(i, result) {
    if (!run.traces) return;
    const trace = result.verification && result.verification.trace;
    // a recorded or older stream may not have sent the steps live: take them from the result
    if (trace && trace.length > run.traces[i].length) {
        run.traces[i] = trace.slice();
        if (i === run.hoodClaim) run.hoodShown = 0;
    }
    renderHoodChips();
    if (i === run.hoodClaim) renderHood();
}

// ---- example inputs --------------------------------------------------------

async function loadSamples() {
    const box = $('samples');
    try {
        const res = await fetch('/api/samples');
        if (!res.ok) throw new Error(res.statusText);
        const samples = await res.json();
        box.innerHTML = '<span class="samples-label">Try:</span>' + samples.map((s, k) => `
            <button type="button" class="chip-btn" data-k="${k}"
                title="${s.recorded ? 'A recorded run exists: it replays instantly, with no API calls' : 'Runs live'}">
                ${escapeHtml(s.label)}${s.recorded ? '<span class="chip-flag">instant</span>' : ''}
            </button>`).join('');
        box.querySelectorAll('.chip-btn').forEach(b => b.addEventListener('click', () => {
            $('inputText').value = samples[+b.dataset.k].text;
            $('inputText').focus();
        }));
        box.classList.remove('hidden');
    } catch {
        box.classList.add('hidden');
    }
}

// ---- stage timeline ----------------------------------------------------------

// Segments are coloured by the cascade phase their stage belongs to (three categorical slots, which
// stay distinguishable in every pairing); the exact stage is in each segment's tooltip
const PHASES = [
    { key: 'search', name: 'Search', color: 'var(--series-1)', stages: ['queries', 'retrieve', 'rewrite'] },
    { key: 'nli', name: 'NLI scoring', color: 'var(--series-2)', stages: ['nli', 'conformal'] },
    { key: 'llm', name: 'LLM escalation', color: 'var(--series-3)', stages: ['kg', 'arbiter'] },
];
const phaseOf = stage => PHASES.find(p => p.stages.includes(stage));
const stageColor = stage => (phaseOf(stage) || { color: 'var(--ink-3)' }).color;

function niceStep(totalSec) {
    return [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120].find(s => totalSec / s <= 8) || 300;
}

function renderTimeline() {
    if (!run.claims.length || !run.segments.some(s => s.length)) return;
    $('timelineSection').classList.remove('hidden');
    const end = Math.max(run.lastMs, ...run.claimEnd.filter(x => x != null), 1);
    const rows = [
        { label: 'Input', segs: run.claimsAt != null ? [{ stage: 'extract', start: 0, end: Math.max(run.claimsAt, 1) }] : [] },
        ...run.claims.map((_, i) => ({
            label: `Claim ${i + 1}`,
            segs: run.segments[i].map((sg, k, arr) => ({
                stage: sg.stage, start: sg.start,
                end: k + 1 < arr.length ? arr[k + 1].start : (run.claimEnd[i] ?? end),
            })),
        })),
    ];
    const W = 1000, labelW = 84, rowH = 20, gap = 10, top = 6, axisH = 26;
    const plotW = W - labelW - 12;
    const H = top + rows.length * (rowH + gap) + axisH;
    const x = ms => labelW + (ms / end) * plotW;
    const step = niceStep(end / 1000);
    const ticks = [];
    for (let s = 0; s * 1000 <= end + 1; s += step) ticks.push(s);

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Time spent in each pipeline stage per claim">`;
    ticks.forEach(s => {
        const tx = x(s * 1000);
        svg += `<line class="grid-line" x1="${tx}" x2="${tx}" y1="${top}" y2="${H - axisH + 4}"/>`;
        svg += `<text class="tick" x="${tx}" y="${H - 8}" text-anchor="middle">${s}s</text>`;
    });
    rows.forEach((row, r) => {
        const y = top + r * (rowH + gap);
        svg += `<text class="tl-row-label" x="0" y="${y + rowH / 2}" dy="0.35em">${escapeHtml(row.label)}</text>`;
        row.segs.forEach(sg => {
            const x0 = x(sg.start), w = Math.max(x(sg.end) - x0 - 2, 1.5);   // 2px surface gap between segments
            const name = sg.stage === 'extract' ? 'Claim extraction' : (STAGES.find(s => s.key === sg.stage) || {}).name || sg.stage;
            const phase = phaseOf(sg.stage);
            const tipHtml = `<div class="tt-title">${escapeHtml(row.label)} · ${escapeHtml(name)}</div>
                ${phase ? `<div class="tt-row"><span>Phase</span><span>${phase.name}</span></div>` : ''}
                <div class="tt-row"><span>Duration</span><span>${((sg.end - sg.start) / 1000).toFixed(2)} s</span></div>
                <div class="tt-row"><span>From – to</span><span>${(sg.start / 1000).toFixed(1)}–${(sg.end / 1000).toFixed(1)} s</span></div>`;
            svg += `<rect class="tl-seg${sg.stage === 'extract' ? ' tl-extract' : ''}" x="${x0}" y="${y}" width="${w}" height="${rowH}" rx="3"
                fill="${stageColor(sg.stage)}" data-tip="${escapeHtml(tipHtml)}"></rect>`;
        });
    });
    svg += '</svg>';
    $('timelineChart').innerHTML = svg;
    $('timelineChart').querySelectorAll('.tl-seg').forEach(seg => {
        seg.addEventListener('mouseenter', e => showTip(seg.dataset.tip, e));
        seg.addEventListener('mousemove', moveTip);
        seg.addEventListener('mouseleave', hideTip);
    });

    const used = new Set(rows.flatMap(r => r.segs.map(s => s.stage)));
    $('timelineLegend').innerHTML =
        (used.has('extract') ? '<span><span class="legend-swatch" style="background:var(--ink-3);opacity:.45"></span>Claim extraction</span>' : '') +
        PHASES.filter(p => p.stages.some(s => used.has(s))).map(p =>
            `<span><span class="legend-swatch" style="background:${p.color}"></span>${p.name}</span>`).join('');

    // Table view of the same data: seconds per phase for each claim
    const claimRows = rows.slice(1);
    $('timelineTable').innerHTML = `
        <thead><tr><th>Claim</th>${PHASES.map(p => `<th>${p.name}</th>`).join('')}<th>Total</th></tr></thead>
        <tbody>${claimRows.map(r => {
            const per = PHASES.map(p => r.segs.filter(sg => p.stages.includes(sg.stage)).reduce((a, sg) => a + sg.end - sg.start, 0));
            const total = r.segs.length ? r.segs[r.segs.length - 1].end - r.segs[0].start : 0;
            return `<tr><td>${escapeHtml(r.label)}</td>${per.map(ms => `<td>${ms ? (ms / 1000).toFixed(1) + ' s' : '—'}</td>`).join('')}<td>${(total / 1000).toFixed(1)} s</td></tr>`;
        }).join('')}</tbody>`;
    $('timelineCaption').textContent = `${(end / 1000).toFixed(1)} s in total${run.replay ? ' (original timing of the recorded run)' : ''}`;
}

// ---- knowledge graph -----------------------------------------------------

let graphData = null;

function renderMainGraph() {
    const nodes = new Map(), edges = new Map();
    run.results.filter(Boolean).forEach(r => {
        ((r.graph || {}).nodes || []).forEach(n => { if (!nodes.has(n.id)) nodes.set(n.id, { ...n }); });
        ((r.graph || {}).edges || []).forEach(e => {
            const key = [e.source, e.target].sort().join('||');
            if (!edges.has(key)) edges.set(key, { ...e });
        });
    });
    const nodeList = [...nodes.values()];
    const edgeList = [...edges.values()].filter(e => nodes.has(e.source) && nodes.has(e.target));
    if (!nodeList.length) return;
    $('kgSection').classList.remove('hidden');
    $('kgCaption').textContent = `${nodeList.length} entit${nodeList.length === 1 ? 'y' : 'ies'} · ${edgeList.length} relation${edgeList.length === 1 ? '' : 's'}`;
    graphData = { nodes: nodeList, edges: edgeList };
    drawGraph(graphData);
}

function drawGraph({ nodes, edges }) {
    const wrap = $('kgWrap');
    const width = wrap.clientWidth || 800, height = wrap.clientHeight || 400;
    const svg = d3.select('#kgSvg').attr('viewBox', `0 0 ${width} ${height}`);
    svg.selectAll('*').remove();

    const defs = svg.append('defs');
    [['arrow', 'kg-arrow'], ['arrow-v', 'kg-arrow-v']].forEach(([id, cls]) => {
        defs.append('marker').attr('id', id).attr('viewBox', '0 -5 10 10').attr('refX', 30).attr('refY', 0)
            .attr('orient', 'auto').attr('markerWidth', 6).attr('markerHeight', 6)
            .append('path').attr('d', 'M0,-5L10,0L0,5').attr('class', cls);
    });

    const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(150))
        .force('charge', d3.forceManyBody().strength(-420))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(48))
        .force('x', d3.forceX(width / 2).strength(0.04))
        .force('y', d3.forceY(height / 2).strength(0.08));

    const edgeHtml = d => {
        const s = d.source.label || d.source.id, t = d.target.label || d.target.id;
        return `<div class="tt-title">${escapeHtml(s)} → ${escapeHtml(t)}</div>
            <div>${escapeHtml(d.relationship || 'related to')}</div>
            <div class="tt-row"><span>${d.verified ? '✓ Provenance-verified' : 'Extracted, not verified'}</span></div>
            ${d.quote ? `<div class="tt-quote">“${escapeHtml(d.quote)}”</div>` : ''}`;
    };

    const linkG = svg.append('g');
    const link = linkG.selectAll('line.kg-link').data(edges).enter().append('line')
        .attr('class', d => `kg-link${d.verified ? ' verified' : ''}`)
        .attr('marker-end', d => `url(#${d.verified ? 'arrow-v' : 'arrow'})`);
    const hit = linkG.selectAll('line.kg-link-hit').data(edges).enter().append('line').attr('class', 'kg-link-hit')
        .on('mouseenter', (e, d) => { link.classed('hl', l => l === d); showTip(edgeHtml(d), e); })
        .on('mousemove', moveTip)
        .on('mouseleave', () => { link.classed('hl', false); hideTip(); });
    const elabel = svg.append('g').selectAll('text').data(edges).enter().append('text')
        .attr('class', 'kg-elabel').attr('text-anchor', 'middle')
        .text(d => { const r = d.relationship || ''; return r.length > 28 ? r.slice(0, 27) + '…' : r; });

    const node = svg.append('g').selectAll('g').data(nodes).enter().append('g').attr('class', 'kg-node')
        .style('cursor', 'pointer')
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
        .on('click', (e, d) => { if (d.url && d.url !== '#') window.open(d.url, '_blank', 'noopener'); })
        .on('mouseenter', (e, d) => {
            link.classed('hl', l => l.source.id === d.id || l.target.id === d.id);
            showTip(`<div class="tt-title">${escapeHtml(d.label || d.id)}</div>
                <div>${escapeHtml((d.summary_snippet || 'Entity').slice(0, 260))}</div>
                ${d.url && d.url !== '#' ? '<div class="tt-row" style="margin-top:4px"><span>Click to open source ↗</span></div>' : ''}`, e);
        })
        .on('mousemove', moveTip)
        .on('mouseleave', () => { link.classed('hl', false); hideTip(); });
    node.append('circle').attr('r', 26);
    node.append('text').attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .text(d => { const l = d.label || d.id; return l.length > 13 ? l.slice(0, 12) + '…' : l; });

    const clampX = x => Math.max(30, Math.min(width - 30, x));
    const clampY = y => Math.max(30, Math.min(height - 30, y));
    sim.on('tick', () => {
        nodes.forEach(d => { d.x = clampX(d.x); d.y = clampY(d.y); });
        [link, hit].forEach(sel => sel
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y));
        elabel.attr('x', d => (d.source.x + d.target.x) / 2).attr('y', d => (d.source.y + d.target.y) / 2 - 4);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

// ================================================================ BENCHMARKS

const BENCH_META = {
    fever: {
        title: 'FEVER',
        desc: 'Claims written from Wikipedia sentences, labelled SUPPORTS / REFUTES (Thorne et al., 2018). Tests entity retrieval and NLI on static encyclopedic facts.',
    },
    freshqa: {
        title: 'FreshQA',
        desc: 'Never- and slow-changing questions converted to claims, including false premises (Vu et al., 2023). Tests web retrieval, time and false-premise detection.',
    },
};

const ROUTES = [
    { key: 'conformal_singleton', name: 'NLI verdict stands', color: 'var(--series-1)' },
    { key: 'contradiction_arbitration', name: 'LLM review', color: 'var(--series-2)' },
    { key: 'multi_agent_debate', name: 'Full escalation', color: 'var(--series-3)' },
];

let benchmarkData = null;

// Fetched once and shared by the Benchmarks and Method tabs
function fetchBenchmarks() {
    if (!benchmarkData) {
        benchmarkData = fetch('/api/benchmarks').then(res => {
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            return res.json();
        });
        benchmarkData.catch(() => { benchmarkData = null; });   // allow a retry after a failure
    }
    return benchmarkData;
}

async function loadBenchmarks() {
    try {
        const data = await fetchBenchmarks();
        benchLoaded = true;
        $('benchLoading').classList.add('hidden');
        const keys = ['fever', 'freshqa'].filter(k => data[k] && data[k].pooled && data[k].set_results);
        if (!keys.length) {
            $('benchGrid').innerHTML = '<p class="muted">No five-set results found. Run <code>python evaluation/run_5_runs.py --dataset all</code>.</p>';
            return;
        }
        $('benchGrid').innerHTML = keys.map(k => benchCard(k, data[k])).join('');
        keys.forEach(k => {
            drawSetChart(k, data[k]);
            bindRouteTips(k, data[k]);
        });
        if (keys.length === 2) $('benchGrid').insertAdjacentHTML('afterend', compareCard(data.fever.pooled, data.freshqa.pooled));
        loadExplorer();
    } catch (err) {
        $('benchLoading').textContent = `Could not load benchmark results: ${err.message}`;
    }
}

function benchCard(key, d) {
    const p = d.pooled, meta = BENCH_META[key];
    const n = p.evaluated_claims;
    const nTrue = Math.round(n / 2);
    const uTrue = Math.max(0, nTrue - p.tn - p.fp);
    const uFalse = Math.max(0, p.unknown - uTrue);
    const fnSup = Math.max(0, p.fn - uFalse);
    const total = Object.values(d.routing || {}).reduce((a, b) => a + b, 0) || 1;

    return `
    <div class="card bench-card" id="bench-${key}">
        <h2>${meta.title}</h2>
        <p class="bench-desc">${meta.desc}</p>
        <div class="bench-stats">
            <div class="bench-stat"><span class="stat-label">Accuracy</span><span class="stat-value">${pct(p.accuracy, 0)}</span>
                <span class="stat-sub">95% CI ${pct(p.ci_lower_95, 0)}–${pct(p.ci_upper_95, 0)}</span></div>
            <div class="bench-stat"><span class="stat-label">Precision</span><span class="stat-value">${p.precision.toFixed(2)}</span><span class="stat-sub">of flags correct</span></div>
            <div class="bench-stat"><span class="stat-label">Recall</span><span class="stat-value">${p.recall.toFixed(2)}</span><span class="stat-sub">of hallucinations caught</span></div>
            <div class="bench-stat"><span class="stat-label">F1</span><span class="stat-value">${p.f1.toFixed(2)}</span><span class="stat-sub">${p.correct_claims}/${n} correct</span></div>
        </div>

        <div class="bench-block">
            <h3>Accuracy per set (20 claims each)</h3>
            <div class="chart" id="chart-${key}"></div>
        </div>

        <div class="bench-block">
            <h3>Confusion matrix (${n} claims)</h3>
            <div class="confusion">
                <span></span>
                <span class="hd">Flagged hallucinated</span><span class="hd">Judged supported</span><span class="hd">Uncertain</span>
                <span class="rh">False claim</span>
                <span class="cell ok">${p.tp}<small>caught (TP)</small></span>
                <span class="cell bad">${fnSup}<small>missed (FN)</small></span>
                <span class="cell abst">${uFalse}<small>abstained (FN)</small></span>
                <span class="rh">True claim</span>
                <span class="cell bad">${p.fp}<small>false alarm (FP)</small></span>
                <span class="cell ok">${p.tn}<small>passed (TN)</small></span>
                <span class="cell abst">${uTrue}<small>abstained</small></span>
            </div>
        </div>

        <div class="bench-block">
            <h3>How claims were decided</h3>
            <div class="route-bar" id="route-${key}">
                ${ROUTES.map(r => `<div class="route-seg" data-key="${r.key}" style="flex-grow:${d.routing[r.key] || 0};background:${r.color}"></div>`).join('')}
            </div>
            <div class="route-legend">
                ${ROUTES.map(r => `<span><span class="legend-swatch" style="background:${r.color}"></span>${r.name} <b>${d.routing[r.key] || 0}</b> <span class="muted">(${pct((d.routing[r.key] || 0) / total, 0)})</span></span>`).join('')}
            </div>
        </div>

        ${routeAccuracyHtml(d)}

        ${latencyHtml(d)}

        <div class="bench-block">
            <h3>Per-set results</h3>
            <table class="data">
                <thead><tr><th>Set</th><th>Acc.</th><th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Unc.</th><th>Latency (measured)</th></tr></thead>
                <tbody>
                    ${d.set_results.map(s => setRow(s.set, s)).join('')}
                    ${setRow('Pooled', p, true)}
                </tbody>
            </table>
        </div>
    </div>`;
}

// Measured latency per claim: the full run, and the current settings on the first set of claims
function latencyHtml(d) {
    const l = d.latency;
    if (!l || !l.full_run) return '';
    const sec = x => `${x.toFixed(1)} s`;
    const row = (name, m, cls = '') => m ? `<tr class="${cls}"><td>${name}</td><td>${sec(m.mean)}</td><td>${sec(m.p50)}</td><td>${sec(m.p90)}</td></tr>` : '';
    return `<div class="bench-block">
        <h3>Latency per claim (measured)</h3>
        <table class="data">
            <thead><tr><th></th><th>Mean</th><th>Median</th><th>90th pct.</th></tr></thead>
            <tbody>
                ${row(`Current settings · ${l.current_settings ? l.current_settings.claims : ''} claims (Set 1)`, l.current_settings, 'pooled')}
                ${row(`Same ${l.same_claims_full_run ? l.same_claims_full_run.claims : ''} claims in the full run`, l.same_claims_full_run)}
                ${row(l.current_settings ? `Full run · ${l.full_run.claims} claims` : `Current settings · ${l.full_run.claims} claims`, l.full_run, l.current_settings ? '' : 'pooled')}
            </tbody>
        </table>
        ${l.current_settings ? '<p class="hint">Current settings send every LLM call to gemini-3.5-flash-lite first; the full run sent supporting calls to the slower gemini-3.1-flash-lite (Docs, §3.4).</p>' : ''}
    </div>`;
}

// Share of claims and accuracy for each decision route
function routeAccuracyHtml(d) {
    const ro = d.route_outcomes || {};
    const total = Object.values(ro).reduce((a, r) => a + r.claims, 0);
    if (!total) return '';
    const rows = ROUTES.filter(r => ro[r.key]).map(r => {
        const { claims, correct } = ro[r.key];
        const acc = correct / claims;
        return `
            <div class="ra-row">
                <span class="ra-name"><span class="legend-swatch" style="background:${r.color}"></span>${r.name}</span>
                <span class="ra-share">${claims} claim${claims === 1 ? '' : 's'}</span>
                <div class="ra-track"><div class="ra-fill" style="width:${(acc * 100).toFixed(1)}%"></div></div>
                <span class="ra-val">${pct(acc, 0)} correct</span>
            </div>`;
    }).join('');
    return `<div class="bench-block"><h3>Accuracy by route</h3><div class="ra-list">${rows}</div></div>`;
}

function setRow(name, s, pooled = false) {
    return `<tr${pooled ? ' class="pooled"' : ''}><td>${escapeHtml(name)}</td><td>${pct(s.accuracy, 0)}</td>
        <td>${s.precision.toFixed(2)}</td><td>${s.recall.toFixed(2)}</td><td>${s.f1.toFixed(2)}</td>
        <td>${s.tp}</td><td>${s.fp}</td><td>${s.fn}</td><td>${s.tn}</td><td>${s.unknown}</td>
        <td>${s.mean_latency_sec.toFixed(1)} s</td></tr>`;
}

function drawSetChart(key, d) {
    const sets = d.set_results;
    const W = 460, H = 190, m = { t: 18, r: 64, b: 26, l: 36 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const x = d3.scaleBand().domain(sets.map(s => s.set)).range([0, iw]).padding(0.38);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    const svg = d3.select(`#chart-${key}`).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img')
        .attr('aria-label', `${BENCH_META[key].title} accuracy per set`);
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);

    [0, 0.25, 0.5, 0.75, 1].forEach(t => {
        g.append('line').attr('class', 'grid-line').attr('x1', 0).attr('x2', iw).attr('y1', y(t)).attr('y2', y(t));
        g.append('text').attr('class', 'tick').attr('x', -8).attr('y', y(t)).attr('dy', '0.32em').attr('text-anchor', 'end').text(`${t * 100}%`);
    });
    g.append('line').attr('class', 'axis-line').attr('x1', 0).attr('x2', iw).attr('y1', ih).attr('y2', ih);

    const r = 4;
    const barPath = s => {
        const x0 = x(s.set), w = x.bandwidth(), y0 = y(s.accuracy), h = ih - y0;
        const rr = Math.min(r, h);
        return `M${x0},${ih}V${y0 + rr}Q${x0},${y0} ${x0 + rr},${y0}H${x0 + w - rr}Q${x0 + w},${y0} ${x0 + w},${y0 + rr}V${ih}Z`;
    };
    const bars = g.selectAll('path.bar').data(sets).enter().append('path').attr('class', 'bar').attr('d', barPath);
    g.selectAll('text.val-label').data(sets).enter().append('text').attr('class', 'val-label')
        .attr('x', s => x(s.set) + x.bandwidth() / 2).attr('y', s => y(s.accuracy) + 15).attr('text-anchor', 'middle')
        .text(s => pct(s.accuracy, 0));
    g.selectAll('text.xl').data(sets).enter().append('text').attr('class', 'tick')
        .attr('x', s => x(s.set) + x.bandwidth() / 2).attr('y', ih + 16).attr('text-anchor', 'middle').text(s => s.set);

    const py = y(d.pooled.accuracy);
    g.append('line').attr('class', 'ref-line').attr('x1', 0).attr('x2', iw).attr('y1', py).attr('y2', py);
    g.append('text').attr('class', 'ref-label').attr('x', iw + 6).attr('y', py).attr('dy', '0.32em').text(`pooled ${pct(d.pooled.accuracy, 0)}`);

    // hit targets span the full band height
    g.selectAll('rect.hit').data(sets).enter().append('rect')
        .attr('x', s => x(s.set) - x.step() * x.paddingInner() / 2).attr('width', x.step()).attr('y', 0).attr('height', ih)
        .attr('fill', 'transparent')
        .on('mouseenter', (e, s) => {
            bars.classed('dim', b => b !== s);
            showTip(`<div class="tt-title">${escapeHtml(s.set)}</div>
                <div class="tt-row"><span>Correct</span><span>${s.correct_claims}/${s.evaluated_claims}</span></div>
                <div class="tt-row"><span>Precision · Recall</span><span>${s.precision.toFixed(2)} · ${s.recall.toFixed(2)}</span></div>
                <div class="tt-row"><span>Uncertain</span><span>${s.unknown}</span></div>
                <div class="tt-row"><span>Mean latency</span><span>${s.mean_latency_sec.toFixed(1)} s</span></div>`, e);
        })
        .on('mousemove', moveTip)
        .on('mouseleave', () => { bars.classed('dim', false); hideTip(); });
}

function bindRouteTips(key, d) {
    const total = Object.values(d.routing || {}).reduce((a, b) => a + b, 0) || 1;
    document.querySelectorAll(`#route-${key} .route-seg`).forEach(seg => {
        const r = ROUTES.find(x => x.key === seg.dataset.key);
        const n = d.routing[r.key] || 0;
        seg.addEventListener('mouseenter', e => showTip(`<div class="tt-title">${r.name}</div>
            <div class="tt-row"><span>Claims</span><span>${n} (${pct(n / total, 0)})</span></div>`, e));
        seg.addEventListener('mousemove', moveTip);
        seg.addEventListener('mouseleave', hideTip);
    });
}

// Describes the measured gap between the benchmarks from the numbers themselves, so the text stays
// true when the results are re-run
function compareCard(f, q) {
    const overlap = q.ci_upper_95 >= f.ci_lower_95;
    const dP = f.precision - q.precision, dR = f.recall - q.recall;
    const fmt = x => x.toFixed(2);
    const where = Math.abs(dP) > Math.abs(dR)
        ? `The gap is mainly in precision (${fmt(f.precision)} vs ${fmt(q.precision)}): FreshQA has more true claims wrongly flagged (${q.fp} vs ${f.fp}), while recall is closer (${fmt(f.recall)} vs ${fmt(q.recall)}).`
        : `The gap is mainly in recall (${fmt(f.recall)} vs ${fmt(q.recall)}): FreshQA has more hallucinations missed or left uncertain (${q.fn} vs ${f.fn}), while precision is closer (${fmt(f.precision)} vs ${fmt(q.precision)}).`;
    return `<div class="card compare">
        <h2 style="margin-bottom:6px">FEVER vs FreshQA</h2>
        <p>FEVER scores ${pct(f.accuracy, 0)} and FreshQA ${pct(q.accuracy, 0)}; the 95% intervals
        ${overlap ? 'overlap, so the gap is suggestive rather than conclusive at this sample size' : 'do not overlap, so FreshQA is genuinely harder'}.
        ${where} FreshQA is harder for reasons the error analysis shows: its claims depend on time and negation, which the NLI model handles poorly,
        and some of its gold answers are contradicted by current evidence (see the Docs, §3.5).</p>
    </div>`;
}

// ---- claim explorer ----------------------------------------------------------

const ROUTE_LABEL = {
    conformal_singleton: 'NLI verdict stands', contradiction_arbitration: 'LLM review',
    multi_agent_debate: 'Full escalation', absence_heuristic: 'Absence check',
};
const explore = { key: 'fever', outcome: 'all', route: '', q: '', open: null, data: {} };

async function loadExplorer() {
    $('explorer').classList.remove('hidden');
    const key = explore.key;
    if (!explore.data[key]) {
        $('exploreBody').innerHTML = '<tr><td colspan="6" class="muted">Loading claims…</td></tr>';
        try {
            const res = await fetch(`/api/benchmark_claims/${key}`);
            if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
            explore.data[key] = await res.json();
        } catch (err) {
            $('exploreBody').innerHTML = `<tr><td colspan="6" class="muted">Could not load the claims: ${escapeHtml(err.message)}</td></tr>`;
            return;
        }
        if (key !== explore.key) return;   // switched benchmark while loading
    }
    renderExplorer();
}

function goldPill(gold) {
    return gold === 'supported' ? '<span class="gold-pill true">True</span>' : '<span class="gold-pill false">False</span>';
}

function predPill(label) {
    const v = VERDICT[label] || VERDICT.unknown;
    return `<span class="verdict-badge ${VERDICT[label] ? label : 'unknown'}"><span class="status-icon ${v.cls}">${v.icon}</span>${v.text}</span>`;
}

function renderExplorer() {
    const rows = explore.data[explore.key] || [];
    const q = explore.q.trim().toLowerCase();
    const list = rows.map((r, k) => ({ r, k })).filter(({ r }) => {
        const ok = r.gold === r.prediction;
        if (explore.outcome === 'errors' && ok) return false;
        if (explore.outcome === 'correct' && !ok) return false;
        if (explore.route && r.decision_policy !== explore.route) return false;
        return !q || r.claim.toLowerCase().includes(q);
    });
    const errors = rows.filter(r => r.gold !== r.prediction).length;
    $('exploreCount').textContent = `${list.length} of ${rows.length} claims shown · ${errors} errors in this benchmark`;
    $('exploreBody').innerHTML = list.map(({ r, k }) => {
        const ok = r.gold === r.prediction;
        const open = explore.open === k;
        return `<tr class="ex-row${ok ? '' : ' wrong'}${open ? ' open' : ''}" data-k="${k}" tabindex="0" aria-expanded="${open}">
                <td>${r.set}</td>
                <td class="ex-claim">${escapeHtml(r.claim)}${r.fact_type ? ` <span class="tag">${escapeHtml(r.fact_type)}</span>` : ''}</td>
                <td>${goldPill(r.gold)}</td>
                <td><span class="ex-pred">${predPill(r.prediction)}<span class="${ok ? 'ex-ok' : 'ex-bad'}" title="${ok ? 'Correct' : 'Wrong'}">${ok ? '✓' : '✕'}</span></span></td>
                <td>${escapeHtml(ROUTE_LABEL[r.decision_policy] || r.decision_policy || '—')}</td>
                <td>${Number(r.latency_sec).toFixed(1)} s</td>
            </tr>${open ? `<tr class="ex-detail"><td colspan="6">${exploreDetailHtml(r)}</td></tr>` : ''}`;
    }).join('') || '<tr><td colspan="6" class="muted">No claims match these filters.</td></tr>';

    $('exploreBody').querySelectorAll('.ex-row').forEach(tr => {
        const toggle = () => {
            const k = +tr.dataset.k;
            explore.open = explore.open === k ? null : k;
            renderExplorer();
        };
        tr.addEventListener('click', toggle);
        tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    });
    const detail = $('exploreBody').querySelector('.ex-detail');
    if (detail) {
        bindClaimDetails(detail);
        animateProbBars(detail);
        requestAnimationFrame(() => detail.querySelectorAll('.hood-step.enter').forEach(x => x.classList.remove('enter')));
        drawCalibPlots(detail);
    }
}

function exploreDetailHtml(r) {
    const result = r.result || { claim: r.claim, evidence: r.evidence, verification: { label: r.prediction } };
    const steps = (result.verification && result.verification.trace) || [];
    const ok = r.gold === r.prediction;
    const models = [...new Set((r.llm_log || []).map(c => c.model).filter(Boolean))];
    const outcome = ok ? '' : `<div class="ev-note ex-error">The claim is <strong>${r.gold === 'supported' ? 'true' : 'false'}</strong>, but the pipeline said
        <strong>${(VERDICT[r.prediction] || VERDICT.unknown).text}</strong>${r.prediction === 'unknown' ? ' (it abstained)' : r.prediction === 'contradicted' ? ' (a false alarm)' : ' (a missed hallucination)'}.
        The steps below show where it went wrong.</div>`;
    return `<div class="ex-body">
        ${outcome}
        ${claimDetailsHtml(result)}
        <h3 class="ex-sub">Computation, step by step</h3>
        ${steps.length ? `<ol class="hood-steps">${steps.map(st => (HOOD_RENDER[st.kind] || (() => ''))(st)).join('')}</ol>` : '<p class="muted">No computation trace was recorded for this claim.</p>'}
        <p class="hint">${r.llm_calls} LLM call${r.llm_calls === 1 ? '' : 's'}${models.length ? ` (${escapeHtml(models.join(', '))})` : ''} · ${Number(r.latency_sec).toFixed(1)} s end to end.</p>
    </div>`;
}

// ================================================================ METHOD

const METHOD = [
    ['Claim decomposition', 'always', 'The input is split into atomic, self-contained factual claims by an LLM (<code>agents/claim_extractor.py</code>), so one false fact cannot hide inside a true paragraph.'],
    ['Query generation', 'always', 'Each claim becomes entity-focused search queries, a domain tag and a flag for time-sensitive claims; simple claims use a rule-based fast path with no LLM call.'],
    ['Evidence retrieval and re-ranking', 'always', 'Wikipedia, web search and arXiv are queried in parallel; paragraphs are re-ranked by a hybrid lexical and semantic score, and web sentences that merely quote the claim are removed.'],
    ['NLI evidence scoring', 'always', 'A BART-large-MNLI cross-encoder scores every snippet. Decisive support settles a claim; a contradiction is accepted directly only when at least two decisive snippets agree.'],
    ['Conformal prediction set', 'always', 'Split conformal prediction, calibrated on 100 FEVER claims, turns the NLI posterior into a label set with ≥ 90% coverage. A single-label set is accepted; a multi-label set triggers escalation.'],
    ['Self-reflective query rewrite', 'only if uncertain', 'When the evidence is inconclusive, the LLM rewrites the queries once and retrieval runs again.'],
    ['Knowledge-graph multi-hop', 'only if escalated', 'Entities are linked into a graph whose edges are kept only when a source sentence attests them (provenance check), adding multi-hop evidence.'],
    ['LLM arbiter', 'only if escalated or contested', 'A dialectical prompt argues for and against the claim, then judges against the raw evidence. It also reviews lone or contested contradictions, which are unreliable on their own.'],
    ['Report', 'always', 'Per-claim verdicts, conformal sets, p-values, evidence and sources are aggregated; one contradicted claim makes the whole answer a hallucination.'],
];

// Mirrors §5 of pipeline.md
const LIMITS = [
    ['Missing specific facts', 'When the evidence does not contain the specific number, date or ranking a claim asserts, the arbiter abstains: 9 of the 22 errors across both benchmarks are such abstentions.'],
    ['NLI blind spots', 'The NLI model does not reason about time ("currently"), mishandles negations and accepts near-matching names; a decisive snippet settles a claim without LLM review, so these errors reach the output (4 of the 17 FreshQA errors).'],
    ['Drifting labels', 'Even slow-changing FreshQA answers go out of date: on 4 of the 17 FreshQA errors, current evidence contradicts the gold answer (e.g. Lewandowski has left Barcelona).'],
    ['Scope of the conformal guarantee', 'The conformal layer is calibrated on FEVER claims with gold evidence; on the pipeline\'s own retrieved evidence its measured coverage is 85%, below 90%, so the prediction set acts as a calibrated escalation trigger. Recalibrating restores coverage but escalates many more claims without improving accuracy.'],
    ['Dependence on a hosted LLM', 'Rate limits are absorbed by switching Gemini models, but another model may judge a borderline claim differently, and the arbiter abstains if every model fails.'],
    ['Evaluation size and repeatability', '100 claims per benchmark give ±4 (FEVER) and ±7 (FreshQA) point confidence intervals, and live retrieval and a non-deterministic LLM mean a re-run can differ slightly.'],
];

// ---- cascade diagram ---------------------------------------------------------

const CASCADE_LANES = [
    { route: 'conformal_singleton', label: 'NLI verdict stands', note: 'no LLM call' },
    { route: 'contradiction_arbitration', label: 'LLM review', note: 'one LLM call' },
    { route: 'multi_agent_debate', label: 'KG + LLM arbiter', note: 'full escalation' },
];
let cascadeKey = 'fever';

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

async function loadCascade() {
    try {
        renderCascade(await fetchBenchmarks());
    } catch (err) {
        $('cascadeWrap').innerHTML = `<p class="muted">Could not load the benchmark routing: ${escapeHtml(err.message)}</p>`;
    }
}

function renderCascade(data) {
    const d = data && data[cascadeKey];
    const ro = d && d.route_outcomes;
    const total = ro ? Object.values(ro).reduce((a, r) => a + r.claims, 0) : 0;
    if (!total) {
        $('cascadeWrap').innerHTML = '<p class="muted">No per-claim benchmark results available.</p>';
        return;
    }
    const W = 1000, H = 330, trunkY = 165, laneY = [62, 165, 268];
    const colors = ROUTES.map(r => cssVar(r.color.replace('var(', '').replace(')', '')));
    const lanes = CASCADE_LANES.map((l, k) => {
        const o = ro[l.route] || { claims: 0, correct: 0 };
        return { ...l, y: laneY[k], color: colors[k], share: o.claims / total, acc: o.claims ? o.correct / o.claims : 0, claims: o.claims };
    });
    const path = y => `M 66 ${trunkY} L 636 ${trunkY} C 664 ${trunkY} 664 ${y} 692 ${y} L 858 ${y} C 884 ${y} 884 ${trunkY} 910 ${trunkY} L 944 ${trunkY}`;
    const box = (x, y, w, label, sub, cls = '') => `
        <g class="cz-node ${cls}">
            <rect x="${x}" y="${y - 22}" width="${w}" height="44" rx="10"></rect>
            <text x="${x + w / 2}" y="${sub ? y - 3 : y}" dy="0.35em" text-anchor="middle" class="cz-label">${label}</text>
            ${sub ? `<text x="${x + w / 2}" y="${y + 12}" dy="0.35em" text-anchor="middle" class="cz-sub">${sub}</text>` : ''}
        </g>`;

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Claims flowing through the pipeline into three routes">`;
    // flow lanes, width proportional to the share of claims
    lanes.forEach((l, k) => {
        svg += `<path id="cz-lane-${k}" d="${path(l.y)}" class="cz-lane" stroke="${l.color}" stroke-width="${(4 + l.share * 26).toFixed(1)}"></path>`;
    });
    // moving dots go under the stage boxes, so claims visibly pass through each stage
    svg += '<g id="czDots"></g>';
    // nodes
    svg += box(10, trunkY, 112, 'Input text');
    svg += box(146, trunkY, 126, 'Atomic claims');
    svg += box(296, trunkY, 150, 'Evidence retrieval');
    svg += box(470, trunkY, 160, 'NLI + conformal set', 'decides the route');
    lanes.forEach(l => {
        svg += box(692, l.y, 166, l.label, l.note, 'cz-route');
        svg += `<text x="775" y="${l.y + 38}" text-anchor="middle" class="cz-stat">${pct(l.share, 0)} of claims · <tspan class="cz-acc">${pct(l.acc, 0)} correct</tspan></text>`;
    });
    svg += box(910, trunkY, 80, 'Verdict');
    svg += '</svg>';
    $('cascadeWrap').innerHTML = svg;

    // insight sentence from the same numbers
    const [nli, review, esc] = lanes;
    const hard = review.claims + esc.claims;
    const hardAcc = hard ? ((review.acc * review.claims) + (esc.acc * esc.claims)) / hard : 0;
    $('cascadeInsight').innerHTML = `On ${BENCH_META[cascadeKey].title}, the NLI model settled <strong>${pct(nli.share, 0)}</strong> of claims on its own, and <strong>${pct(nli.acc, 0)}</strong> of those verdicts were correct. ` +
        `The <strong>${pct(1 - nli.share, 0)}</strong> it passed to the LLM were the harder claims: <strong>${pct(hardAcc, 0)}</strong> correct after review. The conformal layer sends the right claims to the expensive stage.`;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    // moving dots: 20 per cycle, split between the lanes in the measured proportions
    const n = 20;
    const counts = lanes.map(l => Math.floor(l.share * n));
    lanes.map((l, k) => [l.share * n - counts[k], k]).sort((a, b) => b[0] - a[0])
        .slice(0, n - counts.reduce((a, b) => a + b, 0)).forEach(([, k]) => counts[k]++);
    const order = [];
    for (let i = 0; order.length < n; i++) counts.forEach((c, k) => { if (i < c) order.push(k); });   // interleave lanes
    const neutral = cssVar('--ink-3');
    const dur = 6;
    const splitAt = (() => {
        const p = document.getElementById('cz-lane-0');
        return p ? Math.min(0.9, 574 / p.getTotalLength()) : 0.55;
    })();
    $('czDots').innerHTML = order.map((k, i) => {
        const begin = `${(i * dur / n).toFixed(2)}s`;
        // invisible until its animation begins; crosses the shared stages (mostly behind the boxes) in
        // the first 30% of the cycle, then travels its route slowly, where the split is visible
        return `<circle r="5.5" class="cz-dot" fill="${neutral}" opacity="0">
            <animateMotion dur="${dur}s" begin="${begin}" repeatCount="indefinite" calcMode="linear"
                keyPoints="0;${splitAt.toFixed(3)};1" keyTimes="0;0.3;1"><mpath href="#cz-lane-${k}"/></animateMotion>
            <animate attributeName="fill" dur="${dur}s" begin="${begin}" repeatCount="indefinite"
                values="${neutral};${neutral};${lanes[k].color};${lanes[k].color}" keyTimes="0;0.29;0.34;1"/>
            <animate attributeName="opacity" dur="${dur}s" begin="${begin}" repeatCount="indefinite"
                values="0;1;1;0" keyTimes="0;0.04;0.94;1"/>
        </circle>`;
    }).join('');
}

function renderMethod() {
    $('limitsList').innerHTML = LIMITS.map(([title, text]) => `<li><strong>${title}.</strong> ${text}</li>`).join('');
    $('methodList').innerHTML = METHOD.map(([title, when, text], i) => `
        <li class="method-item${when === 'always' ? '' : ' optional'}">
            <span class="method-num">${i + 1}</span>
            <div><h3>${title} <span class="when">${when}</span></h3><p>${text}</p></div>
        </li>`).join('');
}

// ================================================================ DOCS

let docsLoaded = false;
let docsPromise = null;

// Renders pipeline.md like GitHub: math is pulled out before markdown parsing (so markdown never
// mangles TeX) and typeset with KaTeX afterwards.
function renderMarkdownWithMath(md) {
    const stash = [];
    const hold = (value) => `@@MD${stash.push(value) - 1}@@`;

    // fenced code first, so nothing inside it is read as math
    md = md.replace(/^```[^\n]*\n[\s\S]*?\n```[ \t]*$/gm, m => hold({ raw: m }));
    // display math: $$ … $$ (GitHub needs \\\\ and \\% inside these blocks; KaTeX wants \\ and \%)
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) =>
        hold({ tex: tex.replace(/\\\\\\\\/g, '\\\\').replace(/\\\\%/g, '\\%'), display: true }));
    // inline math: $`…`$ then $…$
    md = md.replace(/\$`([^`]+)`\$/g, (_, tex) => hold({ tex, display: false }));
    md = md.replace(/(?<![\\$\w])\$(?![\s$])([^$\n]+?)(?<![\s\\])\$(?![\w$])/g, (_, tex) => hold({ tex, display: false }));
    // code fences go back before parsing; math stays as placeholders
    md = md.replace(/@@MD(\d+)@@/g, (m, i) => (stash[i].raw !== undefined ? stash[i].raw : m));

    let html = marked.parse(md, { gfm: true });
    html = html.replace(/@@MD(\d+)@@/g, (m, i) => {
        const item = stash[i];
        if (!item || item.tex === undefined) return m;
        try {
            return katex.renderToString(item.tex.trim(), { displayMode: item.display, throwOnError: false });
        } catch {
            return escapeHtml(item.tex);
        }
    });
    return html;
}

function slugify(text, used) {
    let base = text.toLowerCase().replace(/[^\w\s.-]/g, '').trim().replace(/[\s.]+/g, '-') || 'section';
    let slug = base, n = 1;
    while (used.has(slug)) slug = `${base}-${n++}`;
    used.add(slug);
    return slug;
}

async function loadDocs() {
    const body = $('docsBody');
    try {
        if (typeof marked === 'undefined' || typeof katex === 'undefined') throw new Error('markdown or math library failed to load (check the internet connection)');
        const res = await fetch('/api/docs');
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        body.innerHTML = renderMarkdownWithMath(await res.text());
        docsLoaded = true;
    } catch (err) {
        body.innerHTML = `<p class="muted">Could not load pipeline.md: ${escapeHtml(err.message)}</p>`;
        return;
    }

    body.querySelectorAll('table').forEach(t => {
        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        t.replaceWith(wrap);
        wrap.appendChild(t);
    });
    body.querySelectorAll('a[href^="http"]').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });

    const used = new Set();
    const heads = [...body.querySelectorAll('h2, h3')];
    heads.forEach(h => { h.id = slugify(h.textContent, used); });
    $('docsToc').innerHTML = heads.map(h =>
        `<a href="#" data-target="${h.id}" class="toc-${h.tagName.toLowerCase()}">${escapeHtml(h.textContent.replace(/\s*\([^)]*\.py[^)]*\)\s*$/, ''))}</a>`).join('');
    $('docsToc').querySelectorAll('a').forEach(a => a.addEventListener('click', e => {
        e.preventDefault();
        document.getElementById(a.dataset.target).scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));

    // highlight the section currently in view
    const links = new Map([...$('docsToc').querySelectorAll('a')].map(a => [a.dataset.target, a]));
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            links.forEach(a => a.classList.remove('active'));
            const a = links.get(entry.target.id);
            if (a) {
                a.classList.add('active');
                a.scrollIntoView({ block: 'nearest' });
            }
        });
    }, { rootMargin: '-70px 0px -70% 0px' });
    heads.forEach(h => observer.observe(h));
}

// ================================================================ init

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => showView(t.dataset.view)));
    document.querySelectorAll('[data-goto]').forEach(a => a.addEventListener('click', async e => {
        e.preventDefault();
        showView(a.dataset.goto);
        window.scrollTo(0, 0);
        if (a.dataset.anchor && docsPromise) {
            await docsPromise;
            const target = document.getElementById(a.dataset.anchor);
            if (target) target.scrollIntoView({ block: 'start' });
        }
    }));
    $('themeToggle').addEventListener('click', () => {
        const next = document.documentElement.dataset.themeEffective === 'dark' ? 'light' : 'dark';
        try { localStorage.setItem('verifact-theme', next); } catch { /* storage unavailable */ }
        applyTheme(next);
        if ($('view-method').classList.contains('active')) loadCascade();
    });
    document.querySelectorAll('[data-cascade]').forEach(b => b.addEventListener('click', () => {
        cascadeKey = b.dataset.cascade;
        document.querySelectorAll('[data-cascade]').forEach(x => x.classList.toggle('active', x === b));
        loadCascade();
    }));
    document.querySelectorAll('[data-explore]').forEach(b => b.addEventListener('click', () => {
        explore.key = b.dataset.explore;
        explore.open = null;
        document.querySelectorAll('[data-explore]').forEach(x => x.classList.toggle('active', x === b));
        loadExplorer();
    }));
    document.querySelectorAll('[data-outcome]').forEach(b => b.addEventListener('click', () => {
        explore.outcome = b.dataset.outcome;
        document.querySelectorAll('[data-outcome]').forEach(x => x.classList.toggle('active', x === b));
        renderExplorer();
    }));
    $('exploreRoute').addEventListener('change', e => { explore.route = e.target.value; renderExplorer(); });
    $('exploreSearch').addEventListener('input', e => { explore.q = e.target.value; renderExplorer(); });
    loadSamples();
    $('analyzeBtn').addEventListener('click', () => verify(false));
    $('runLiveBtn').addEventListener('click', () => verify(true));
    $('inputText').addEventListener('keydown', e => {
        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !$('analyzeBtn').disabled) verify(false);
    });
    $('expandAllBtn').addEventListener('click', () => {
        const cards = [...document.querySelectorAll('.claim-card')].filter((_, i) => run.results[i]);
        const open = !cards.every(c => c.classList.contains('open'));
        cards.forEach(c => {
            c.classList.toggle('open', open);
            c.querySelector('.claim-top').setAttribute('aria-expanded', open);
            if (open) animateProbBars(c);
        });
        $('expandAllBtn').textContent = open ? 'Collapse all' : 'Expand all';
    });
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => { if (graphData && !$('kgSection').classList.contains('hidden')) drawGraph(graphData); }, 200);
    });

    buildStepper();
    renderMethod();
    const initial = location.hash.replace('#', '');
    if (['verify', 'benchmarks', 'method', 'docs'].includes(initial)) showView(initial);
});
