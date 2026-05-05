/**
 * MediPredict v7 — script.js
 * Dark mode · symptom selection · counter · loading · result animations
 */

/* ── 1. Theme ───────────────────────────────────────────────── */
(function(){
  const saved = localStorage.getItem('mp-theme');
  const sys   = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', saved || sys);
})();

document.addEventListener('DOMContentLoaded', () => {
  const themeBtn = document.getElementById('themeBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('mp-theme', next);
    });
  }

  if (document.getElementById('mainForm'))       initIndex();
  if (document.querySelector('.result-hero'))    initResult();
});

/* ══════════════════════════════════════════════════════════════
   INDEX PAGE
   ══════════════════════════════════════════════════════════════ */
function initIndex() {
  const form       = document.getElementById('mainForm');
  const btnAnalyse = document.getElementById('btnAnalyse');
  const btnClear   = document.getElementById('btnClear');
  const cbNum      = document.getElementById('cbNum');
  const cbDot      = document.getElementById('cbDot');
  const cbMsg      = document.getElementById('cbMsg');
  const cbFill     = document.getElementById('cbFill');
  const scSub      = document.getElementById('scSub');
  const overlay    = document.getElementById('loadingOverlay');
  const loadingMsg = document.getElementById('loadingMsg');

  const MIN = 2, GOOD = 5;

  const MSGS = [
    'Running ensemble model…',
    'Analysing symptom patterns…',
    'Cross-referencing 41 diseases…',
    'Calculating confidence scores…',
    'Almost there…',
  ];

  /* ── onChipChange (called from HTML onchange) ────────────── */
  window.onChipChange = function(cb) {
    const count = document.querySelectorAll('input[name="symptoms"]:checked').length;

    /* Animate counter */
    cbNum.textContent = count;
    cbNum.classList.remove('bump');
    void cbNum.offsetWidth;
    cbNum.classList.add('bump');

    /* Progress bar */
    const pct = Math.min(100, (count / MIN) * 100);
    cbFill.style.width = pct + '%';
    cbFill.classList.toggle('done', count >= MIN);

    /* Dot */
    cbDot.classList.toggle('active', count >= MIN);

    /* Status message */
    if (count === 0)       cbMsg.textContent = 'Select at least 2 symptoms to begin';
    else if (count < MIN)  cbMsg.textContent = `Select ${MIN - count} more symptom${MIN-count>1?'s':''} to unlock analysis`;
    else if (count < GOOD) cbMsg.textContent = `${count} symptoms — add more for higher accuracy`;
    else                   cbMsg.textContent = `${count} symptoms selected — good coverage`;

    /* Button */
    btnAnalyse.disabled = count < MIN;
    btnClear.disabled   = count === 0;

    /* Sub text */
    if (scSub) {
      if (count === 0)       scSub.textContent = 'Select symptoms above to get started';
      else if (count < GOOD) scSub.textContent = `${count} symptom${count>1?'s':''} selected — more improves accuracy`;
      else                   scSub.textContent = `${count} symptoms — ready for a confident analysis`;
    }

    /* Per-section counts */
    updateSectionCounts();
  };

  /* ── updateSectionCounts ─────────────────────────────────── */
  function updateSectionCounts() {
    document.querySelectorAll('.sym-section').forEach(sec => {
      const n    = sec.querySelectorAll('input[name="symptoms"]:checked').length;
      const badge = sec.querySelector('.sym-cat-count');
      if (badge) {
        badge.textContent = n + ' selected';
        badge.classList.toggle('sel', n > 0);
      }
    });
  }

  /* ── clearAll ────────────────────────────────────────────── */
  window.clearAll = function() {
    document.querySelectorAll('input[name="symptoms"]').forEach(cb => { cb.checked = false; });
    window.onChipChange(null);
  };

  /* ── Form submit ─────────────────────────────────────────── */
  form.addEventListener('submit', function(e) {
    const count = document.querySelectorAll('input[name="symptoms"]:checked').length;
    if (count < MIN) { e.preventDefault(); return; }

    if (overlay) {
      overlay.removeAttribute('hidden');
      let i = 0;
      const iv = setInterval(() => {
        i = (i + 1) % MSGS.length;
        if (loadingMsg) loadingMsg.textContent = MSGS[i];
      }, 750);
      setTimeout(() => clearInterval(iv), 15000);
    }
  });

  /* ── Keyboard: space/enter to toggle chips ───────────────── */
  document.querySelectorAll('.sym-chip').forEach(chip => {
    chip.addEventListener('keydown', e => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        const cb = chip.querySelector('input');
        cb.checked = !cb.checked;
        window.onChipChange(cb);
      }
    });
  });

  /* Initial render */
  window.onChipChange(null);
}

/* ══════════════════════════════════════════════════════════════
   RESULT PAGE
   ══════════════════════════════════════════════════════════════ */
function initResult() {

  /* ── Confidence ring animation ───────────────────────────── */
  const ringFill = document.querySelector('.cr-fill');
  if (ringFill) {
    const target = parseFloat(ringFill.getAttribute('stroke-dashoffset')) || 232;
    ringFill.style.strokeDashoffset = '232';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ringFill.style.strokeDashoffset = target;
    }));
  }

  /* ── Count-up for confidence percentage ──────────────────── */
  const crPct = document.querySelector('.cr-pct');
  if (crPct) {
    const t = parseFloat(crPct.getAttribute('data-t')) || 0;
    countUp(crPct, 0, t, 1200);
  }

  /* ── Bar animations ──────────────────────────────────────── */
  const bars = document.querySelectorAll('.pred-fill');
  bars.forEach((bar, i) => {
    setTimeout(() => bar.classList.add('animating'), 100 + i * 70);
  });

  /* ── Intersection observer for cards ─────────────────────── */
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.06 });

    document.querySelectorAll('.card').forEach((card, i) => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(14px)';
      card.style.transition = `opacity .4s ease ${i * 0.07}s, transform .4s ease ${i * 0.07}s`;
      obs.observe(card);
    });
  }
}

/* ── Utility: count-up number ─────────────────────────────── */
function countUp(el, from, to, duration) {
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const e = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
