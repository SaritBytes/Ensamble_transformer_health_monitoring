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

let trendChart = null;
let knownFeatures = [];

// ── Chart.js Setup ──
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

// ── Build KPI Cards ──
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

// ── Build Probability Bars ──
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

// ── Update Everything ──
function updateUI(history) {
    if (!history.length) return;

    const latest = history[history.length - 1];
    const last20 = history.slice(-20);

    // Chart
    trendChart.data.labels = last20.map(r =>
        new Date(r.timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})
    );
    trendChart.data.datasets[0].data = last20.map(r => r.confidence);
    trendChart.data.datasets[0].pointBackgroundColor = last20.map(r => CLASS_COLORS[r.status] || '#94a3b8');
    trendChart.data.datasets[0].pointBorderColor     = last20.map(r => CLASS_COLORS[r.status] || '#94a3b8');
    trendChart.update('none');

    // Health badge
    const badge  = document.getElementById('health-badge');
    const label  = document.getElementById('badge-label');
    const conf   = document.getElementById('badge-conf');
    const color  = CLASS_COLORS[latest.status] || '#94a3b8';

    label.textContent = latest.readable_status || latest.status;
    label.style.color = color;
    conf.textContent  = `${latest.confidence}% confidence`;

    // Stats row
    document.getElementById('stat-health').textContent = latest.readable_status || latest.status;
    document.getElementById('stat-health').style.color = color;
    document.getElementById('stat-time').textContent =
        new Date(latest.timestamp).toLocaleTimeString();

    // Live indicator
    const dot = document.getElementById('live-dot');
    const lbl = document.getElementById('live-label');
    dot.classList.add('active');
    lbl.textContent = 'System Active';
    lbl.style.color = 'var(--green)';

    // Probability bars
    if (latest.probabilities) {
        for (const [cls, pct] of Object.entries(latest.probabilities)) {
            const fill = document.getElementById(`prob-fill-${cls}`);
            const val  = document.getElementById(`prob-val-${cls}`);
            if (fill) fill.style.width = `${pct}%`;
            if (val)  val.textContent  = `${pct.toFixed(1)}%`;
        }
    }

    // KPI cards
    if (latest.features) {
        for (const [key, val] of Object.entries(latest.features)) {
            const el = document.getElementById(`kpi-${key}`);
            if (el) el.textContent = typeof val === 'number' ? val.toFixed(3) : val;
        }
    }
}

// ── Fetch Stats ──
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        if (res.ok) {
            const s = await res.json();
            document.getElementById('stat-buffer').textContent = `${s.buffer_size} / ${s.buffer_capacity}`;
            if (s.avg_confidence != null)
                document.getElementById('stat-avg-conf').textContent = `${s.avg_confidence}%`;
        }
    } catch (e) {}
}

// ── Fetch Alerts ──
async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts');
        if (res.ok) {
            const a = await res.json();
            const banner = document.getElementById('alert-banner');
            if (a.active) {
                banner.className = `alert-banner ${a.level}`;
                document.getElementById('alert-text').textContent = a.message;
            } else {
                banner.className = 'alert-banner hidden';
            }
        }
    } catch (e) {}
}

// ── Main Poll Loop ──
async function poll() {
    try {
        const res = await fetch('/api/history');
        if (res.ok) {
            const history = await res.json();
            updateUI(history);

            // Initialize prob bars on first data if not done yet
            if (history.length && !document.getElementById('prob-fill-G')) {
                const classes = Object.keys(history[0].probabilities);
                initProbBars(classes);
            }
        }
    } catch (e) {
        document.getElementById('live-label').textContent = 'Connection Lost';
        document.getElementById('live-dot').classList.remove('active');
    }

    await fetchStats();
    await fetchAlerts();

    setTimeout(poll, 2000);
}

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    initChart();

    // Fetch expected features from server
    try {
        const res = await fetch('/api/features');
        if (res.ok) {
            const data = await res.json();
            knownFeatures = data.features || [];
            initKPIs(knownFeatures);
        }
    } catch (e) {
        initKPIs(['Water','Acidity','DBV','DF','TDCG','Furan']);
    }

    poll();
});
