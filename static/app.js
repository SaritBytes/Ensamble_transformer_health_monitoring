// ════════════════════════════════════════════════════════════
//  Transformer Health Monitor – Fully Client-Side Dashboard
//  Loads pre-computed predictions from demo_data.json and
//  simulates a live sensor feed in the browser.
// ════════════════════════════════════════════════════════════

// ── Color mapping for health classes ──
const CLASS_COLORS = {
    'VG': '#10b981',   // green
    'G':  '#22d3ee',   // cyan
    'M':  '#f59e0b',   // amber
    'B':  '#f97316',   // orange
    'VB': '#ef4444',   // red
};

const CLASS_NAMES = {
    'VG': 'Very Good',
    'G':  'Good',
    'M':  'Moderate',
    'B':  'Bad',
    'VB': 'Very Bad',
};

// ── State ──
let trendChart = null;
let demoRecords = [];
let demoClasses = [];
let demoFeatures = [];
let recordIndex = 0;

const BUFFER_CAPACITY = 1440;
const ALERT_THRESHOLD = 3;
const FEED_INTERVAL_MS = 2000;
const memoryBuffer = [];          // FIFO ring buffer (in-browser)
let alertState = { active: false, message: '', level: 'ok' };

// ══════════════════════════════════════════
//  Chart.js Setup
// ══════════════════════════════════════════
function initChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Confidence (%)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.08)',
                borderWidth: 2.5,
                pointBackgroundColor: [],
                pointBorderColor: [],
                pointRadius: 5,
                pointHoverRadius: 7,
                fill: true,
                tension: 0.35,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { callback: v => v + '%' },
                },
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 },
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(11,17,32,0.95)',
                    titleColor: '#fff',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: ctx => `${ctx.parsed.y.toFixed(1)}% confidence`
                    }
                }
            }
        }
    });
}

// ══════════════════════════════════════════
//  Build KPI Cards
// ══════════════════════════════════════════
function initKPIs(features) {
    const grid = document.getElementById('kpi-grid');
    grid.innerHTML = '';
    features.forEach(name => {
        const card = document.createElement('div');
        card.className = 'kpi-card';
        card.innerHTML = `
            <div class="kpi-name">${name}</div>
            <div class="kpi-val" id="kpi-${name}">--</div>
        `;
        grid.appendChild(card);
    });
}

// ══════════════════════════════════════════
//  Build Probability Bars
// ══════════════════════════════════════════
function initProbBars(classes) {
    const container = document.getElementById('prob-bars');
    container.innerHTML = '';
    classes.forEach(cls => {
        const color = CLASS_COLORS[cls] || '#94a3b8';
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span class="prob-label">${CLASS_NAMES[cls] || cls}</span>
            <div class="prob-track">
                <div class="prob-fill" id="prob-fill-${cls}" style="background:${color}"></div>
            </div>
            <span class="prob-val" id="prob-val-${cls}">--</span>
        `;
        container.appendChild(row);
    });
}

// ══════════════════════════════════════════
//  Alert Engine (mirrors server.py logic)
// ══════════════════════════════════════════
function evaluateAlerts() {
    if (memoryBuffer.length < ALERT_THRESHOLD) return;

    const recent = memoryBuffer.slice(-ALERT_THRESHOLD).map(r => r.status);

    if (recent.every(s => s === 'B' || s === 'VB')) {
        alertState = {
            active: true,
            message: `CRITICAL: Last ${ALERT_THRESHOLD} readings show Bad/Very Bad health!`,
            level: 'danger'
        };
    } else if (recent.every(s => s === 'M')) {
        alertState = {
            active: true,
            message: `WARNING: Sustained Moderate health over ${ALERT_THRESHOLD} readings.`,
            level: 'warning'
        };
    } else {
        alertState = { active: false, message: '', level: 'ok' };
    }

    // Render
    const banner = document.getElementById('alert-banner');
    if (alertState.active) {
        banner.className = `alert-banner ${alertState.level}`;
        document.getElementById('alert-text').textContent = alertState.message;
    } else {
        banner.className = 'alert-banner hidden';
    }
}

// ══════════════════════════════════════════
//  Update Stats Row
// ══════════════════════════════════════════
function updateStats() {
    document.getElementById('stat-buffer').textContent =
        `${memoryBuffer.length} / ${BUFFER_CAPACITY}`;

    if (memoryBuffer.length > 0) {
        const avgConf = memoryBuffer.reduce((s, r) => s + r.confidence, 0) / memoryBuffer.length;
        document.getElementById('stat-avg-conf').textContent = `${avgConf.toFixed(2)}%`;
    }
}

// ══════════════════════════════════════════
//  Update All UI Panels
// ══════════════════════════════════════════
function updateUI() {
    if (!memoryBuffer.length) return;

    const latest = memoryBuffer[memoryBuffer.length - 1];
    const last20 = memoryBuffer.slice(-20);

    // ── Chart ──
    trendChart.data.labels = last20.map(r =>
        new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    );
    trendChart.data.datasets[0].data = last20.map(r => r.confidence);
    trendChart.data.datasets[0].pointBackgroundColor = last20.map(r => CLASS_COLORS[r.status] || '#94a3b8');
    trendChart.data.datasets[0].pointBorderColor     = last20.map(r => CLASS_COLORS[r.status] || '#94a3b8');
    trendChart.update('none');

    // ── Health Badge ──
    const label = document.getElementById('badge-label');
    const conf  = document.getElementById('badge-conf');
    const color = CLASS_COLORS[latest.status] || '#94a3b8';

    label.textContent = latest.readable_status || latest.status;
    label.style.color = color;
    conf.textContent  = `${latest.confidence}% confidence`;

    // ── Stats Row ──
    document.getElementById('stat-health').textContent = latest.readable_status || latest.status;
    document.getElementById('stat-health').style.color = color;
    document.getElementById('stat-time').textContent   = new Date(latest.timestamp).toLocaleTimeString();

    // ── Live Indicator ──
    document.getElementById('live-dot').classList.add('active');
    document.getElementById('live-label').textContent = 'Simulation Active';
    document.getElementById('live-label').style.color = 'var(--green)';

    // ── Probability Bars ──
    if (latest.probabilities) {
        for (const [cls, pct] of Object.entries(latest.probabilities)) {
            const fill = document.getElementById(`prob-fill-${cls}`);
            const val  = document.getElementById(`prob-val-${cls}`);
            if (fill) fill.style.width = `${pct}%`;
            if (val)  val.textContent  = `${pct.toFixed(1)}%`;
        }
    }

    // ── KPI Cards ──
    if (latest.features) {
        for (const [key, val] of Object.entries(latest.features)) {
            const el = document.getElementById(`kpi-${key}`);
            if (el) el.textContent = typeof val === 'number' ? val.toFixed(3) : val;
        }
    }
}

// ══════════════════════════════════════════
//  Simulation Feed – ingest one record
// ══════════════════════════════════════════
function ingestNext() {
    if (!demoRecords.length) return;

    const baseRecord = demoRecords[recordIndex % demoRecords.length];
    recordIndex++;

    // Add a small random jitter to features so each cycle feels fresh
    const jitteredFeatures = {};
    for (const [key, val] of Object.entries(baseRecord.features)) {
        const jitter = val * (0.97 + Math.random() * 0.06);   // ±3%
        jitteredFeatures[key] = parseFloat(jitter.toFixed(4));
    }

    const record = {
        timestamp: new Date().toISOString(),
        features: jitteredFeatures,
        status: baseRecord.status,
        readable_status: baseRecord.readable_status,
        confidence: baseRecord.confidence,
        probabilities: baseRecord.probabilities,
    };

    // FIFO
    memoryBuffer.push(record);
    if (memoryBuffer.length > BUFFER_CAPACITY) memoryBuffer.shift();

    updateUI();
    updateStats();
    evaluateAlerts();
}

// ══════════════════════════════════════════
//  Initialization
// ══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    initChart();

    // Load pre-computed demo data
    try {
        const res = await fetch('./demo_data.json');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        demoRecords  = data.records  || [];
        demoClasses  = data.classes  || [];
        demoFeatures = data.features || [];

        initKPIs(demoFeatures);
        initProbBars(demoClasses);

        // Start simulation loop
        setInterval(ingestNext, FEED_INTERVAL_MS);
        // Immediately ingest the first record
        ingestNext();

    } catch (err) {
        console.error('Failed to load demo data:', err);
        document.getElementById('live-label').textContent = 'Data Load Failed';
        document.getElementById('live-label').style.color = 'var(--red)';
        // Fallback KPIs
        initKPIs(['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']);
    }
});
