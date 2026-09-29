/* Progressive enhancement only. No network APIs, analytics, model calls,
   frameworks, secrets, or build step. All scientific data lives in data.js. */
(() => {
  'use strict';
  const D = window.EVOGEN_DATA;
  if (!D) return;
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));
  const text = (id, value) => { const node = document.getElementById(id); if (node) node.textContent = value; };
  const escape = (value) => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let selectedCase = 'policy';
  let toastTimer;
  function notify(message) {
    const toast = $('#toast');
    toast.textContent = message;
    toast.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
  }
  async function copyText(value, successMessage) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
      } else {
        const area = document.createElement('textarea');
        area.value = value;
        area.setAttribute('readonly', '');
        Object.assign(area.style, {position:'fixed', top:'-9999px', opacity:'0'});
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand('copy');
        area.remove();
        if (!ok) throw new Error('Clipboard unavailable');
      }
      notify(successMessage);
    } catch (_) {
      notify('Copy unavailable in this browser. Please select and copy the text or URL manually.');
    }
  }
  function activateTabs(selector, key, value) {
    $$(selector).forEach(button => {
      const active = button.dataset[key] === value;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
  }
  function tabKeys(selector, handler, key) {
    const buttons = $$(selector);
    buttons.forEach((button, index) => {
      button.addEventListener('keydown', event => {
        const map = {ArrowRight:(index + 1) % buttons.length, ArrowLeft:(index + buttons.length - 1) % buttons.length, Home:0, End:buttons.length - 1};
        if (!(event.key in map)) return;
        event.preventDefault();
        const target = buttons[map[event.key]];
        handler(target.dataset[key]);
        target.focus({preventScroll:true});
        target.scrollIntoView({block:'nearest', inline:'nearest', behavior:'instant'});
      });
    });
  }
  function setPrompt(node, prompt, focus) {
    node.replaceChildren();
    const index = prompt.indexOf(focus);
    if (index < 0) { node.textContent = prompt; return; }
    node.append(document.createTextNode(prompt.slice(0, index)));
    const mark = document.createElement('mark'); mark.textContent = focus; node.append(mark);
    node.append(document.createTextNode(prompt.slice(index + focus.length)));
  }
  function selectCase(id) {
    const c = D.cases.find(item => item.id === id);
    if (!c) return;
    selectedCase = id;
    activateTabs('[data-case]', 'case', id);
    $('#case-panel').setAttribute('aria-labelledby', `case-tab-${id}`);
    text('case-title', c.title);
    text('case-topic', `${c.label} / ${c.topic}`);
    text('case-before-label', c.beforeLabel);
    text('case-after-label', c.afterLabel);
    text('case-changed', c.changed);
    text('case-description', c.description);
    setPrompt($('#case-prompt'), c.prompt, c.focus);
    for (const state of ['before', 'after']) {
      const src = `assets/images/${id}-${state}.webp`;
      const img = $(`#case-${state}`);
      img.src = src;
      img.alt = `${state === 'before' ? 'Before' : 'After'} external harness evolution — ${c.topic}. ${c.prompt}`;
      const zoom = $(`#${state}-zoom`);
      zoom.dataset.zoom = src;
      zoom.dataset.caption = `${c.label} · ${state === 'before' ? c.beforeLabel : c.afterLabel}`;
      $(`#slider-${state}`).src = src;
      $(`#slider-${state}`).alt = img.alt;
    }
    $('#case-source').href = `${D.paper}#page=${c.page}`;
    $('#case-source').replaceChildren(document.createTextNode(c.ref + ' ↗'));
    $('#compare-range').value = 50;
    updateReveal();
  }
  $$('[data-case]').forEach(b => b.addEventListener('click', () => selectCase(b.dataset.case)));
  tabKeys('[data-case]', selectCase, 'case');
  $$('[data-jump-case]').forEach(b => b.addEventListener('click', () => selectCase(b.dataset.jumpCase)));
  $$('[data-view]').forEach(button => button.addEventListener('click', () => {
    const slider = button.dataset.view === 'slider';
    $('#compare-pair').hidden = slider;
    $('#slider-view').hidden = !slider;
    $$('[data-view]').forEach(b => { const active = b === button; b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active)); });
  }));
  function updateReveal() {
    const value = Number($('#compare-range').value);
    $('#comparison-slider').style.setProperty('--reveal', `${value}%`);
    $('#compare-range').setAttribute('aria-valuetext', `${value} percent before, ${100-value} percent after`);
  }
  $('#compare-range').addEventListener('input', updateReveal);
  $('#share-case').addEventListener('click', () => {
    const url = new URL(D.siteUrl);
    url.searchParams.set('case', selectedCase);
    url.hash = 'examples';
    copyText(url.href, 'Example link copied.');
  });
  function selectProgress(id) {
    const item = D.progressions.find(p => p.id === id);
    if (!item) return;
    activateTabs('[data-progress]', 'progress', id);
    $('#progress-panel').setAttribute('aria-labelledby', `progress-tab-${id}`);
    text('progress-prompt', `“${item.prompt}”`);
    ['before','policy','final'].forEach((suffix, index) => {
      const src = `assets/images/${id}-${suffix}.webp`;
      const img = $(`#progress-image-${index}`);
      img.src = src;
      img.alt = `${['Initial output','After Policy edit','After further evolution'][index]}: ${item.states[index].join('; ')}. Prompt: ${item.prompt}`;
      const zoom = $(`[data-progress-zoom="${index}"]`);
      zoom.dataset.zoom = src;
      zoom.dataset.caption = `${item.name} · ${['Initial output','After Policy edit','After further evolution'][index]}`;
      $(`#progress-status-${index}`).replaceChildren();
      item.states[index].forEach(label => {
        const span = document.createElement('span');
        span.className = label.includes('repaired') ? 'ok' : 'miss';
        span.textContent = label;
        $(`#progress-status-${index}`).append(span);
      });
    });
  }
  $$('[data-progress]').forEach(b => b.addEventListener('click', () => selectProgress(b.dataset.progress)));
  tabKeys('[data-progress]', selectProgress, 'progress');
  function selectResponsibility(id) {
    const item = D.responsibilities.find(r => r.id === id);
    if (!item) return;
    activateTabs('[data-responsibility]', 'responsibility', id);
    $('#responsibility-panel').setAttribute('aria-labelledby', `responsibility-tab-${id}`);
    text('responsibility-subtitle', item.subtitle);
    text('responsibility-question', item.question);
    text('responsibility-body', item.body);
    text('responsibility-recall', item.recall);
  }
  $$('[data-responsibility]').forEach(b => b.addEventListener('click', () => selectResponsibility(b.dataset.responsibility)));
  tabKeys('[data-responsibility]', selectResponsibility, 'responsibility');
  function selectBenchmark(id) {
    const b = D.benchmarks.find(item => item.id === id);
    if (!b) return;
    activateTabs('[data-benchmark]', 'benchmark', id);
    $('#benchmark-panel').setAttribute('aria-labelledby', `bench-tab-${id}`);
    text('benchmark-name', b.name);
    text('benchmark-metric', b.metric);
    text('benchmark-score', b.value);
    text('benchmark-delta', `${b.delta} vs. best evaluated baseline`);
    text('benchmark-description', b.description);
    text('results-caption', `${b.name} · Report Table ${b.table}`);
    text('benchmark-source', `Table ${b.table} ↗`);
    $('#benchmark-source').href = `${D.paper}#page=${b.page}`;
    const sorted = [...b.rows].sort((a, c) => c[1] - a[1]);
    $('#benchmark-chart').innerHTML = sorted.slice(0, 5).map(([name, score]) => {
      const ours = name === 'EvoGen-Harness';
      const width = Math.min(100, Math.max(0, Number(score) * 100));
      return `<div class="chart-row${ours ? ' is-ours' : ''}"><div class="chart-label">${escape(name)}${ours ? '<span>Ours</span>' : ''}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${Number(score).toFixed(4)}</strong></div>`;
    }).join('');
    $('#results-tbody').innerHTML = sorted.map(([name, score]) => `<tr${name === 'EvoGen-Harness' ? ' class="is-ours"' : ''}><td>${escape(name)}</td><td>${Number(score).toFixed(4)}</td></tr>`).join('');
  }
  $$('[data-benchmark]').forEach(b => b.addEventListener('click', () => selectBenchmark(b.dataset.benchmark)));
  $$('[data-bench-jump]').forEach(b => b.addEventListener('click', () => selectBenchmark(b.dataset.benchJump)));
  tabKeys('[data-benchmark]', selectBenchmark, 'benchmark');
  $$('[data-backbone-state]').forEach(button => button.addEventListener('click', () => {
    const state = button.dataset.backboneState;
    $$('[data-backbone-state]').forEach(b => { const active = b === button; b.classList.toggle('active', active); b.setAttribute('aria-pressed', String(active)); });
    $$('[data-backbone-image]').forEach(img => {
      const key = img.dataset.backboneImage;
      const name = {flux:'FLUX.1-dev',qwen:'Qwen-Image',janus:'Janus-Pro'}[key];
      const src = `assets/images/${key}-${state}.webp`;
      img.src = src;
      img.alt = `${name} ${state === 'before' ? 'before' : 'after'} external harness evolution. Selected qualitative example from report Figure 7.`;
      const zoom = $(`[data-backbone-zoom="${key}"]`);
      zoom.dataset.zoom = src;
      zoom.dataset.caption = `${name} · ${state === 'before' ? 'Before evolution' : 'With EvoGen-Harness'} · Figure 7`;
    });
  }));
  // Native dialog provides focus containment and Escape-to-close behavior.
  const dialog = $('#image-dialog');
  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-zoom]');
    if (!trigger) return;
    const path = trigger.dataset.zoom;
    if (!path || !path.startsWith('assets/images/')) return;
    if (typeof dialog.showModal !== 'function') { window.open(path, '_blank', 'noopener'); return; }
    $('#dialog-image').src = path;
    $('#dialog-image').alt = trigger.dataset.caption || 'Paper image';
    text('dialog-caption', trigger.dataset.caption || 'Paper image');
    $('#dialog-open-original').href = path;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  });
  $('#close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => { document.body.style.overflow = ''; });
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  });
  // Mobile navigation and keyboard behavior.
  const toggle = $('.menu-toggle');
  function closeNav() { $('#site-nav').classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); toggle.setAttribute('aria-label','Open navigation'); }
  toggle.addEventListener('click', () => {
    const expanded = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', expanded ? 'Close navigation' : 'Open navigation');
    $('#site-nav').classList.toggle('open', expanded);
  });
  $$('#site-nav a').forEach(a => a.addEventListener('click', closeNav));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeNav(); });
  document.addEventListener('click', e => { if (!e.target.closest('.site-header')) closeNav(); });
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        $$('#site-nav a[href^="#"]').forEach(a => a.classList.toggle('is-current', a.getAttribute('href') === `#${entry.target.id}`));
      });
    }, {rootMargin:'-12% 0px -65% 0px', threshold:0});
    ['examples','method','results','paper'].forEach(id => observer.observe(document.getElementById(id)));
  }
  $('#copy-citation').addEventListener('click', () => copyText($('#bibtex').textContent, 'BibTeX citation copied.'));
  $('#share-project').addEventListener('click', async () => {
    const shareData = {title:'EvoGen-Harness: Same model. Smarter harness.', text:'Learning where and how to evolve image-generation harnesses.', url:D.siteUrl};
    if (navigator.share) {
      try { await navigator.share(shareData); return; }
      catch (error) { if (error.name === 'AbortError') return; }
    }
    copyText(D.siteUrl, 'Project link copied.');
  });
  const requested = new URLSearchParams(window.location.search).get('case');
  if (requested && D.cases.some(c => c.id === requested)) selectCase(requested);
})();
