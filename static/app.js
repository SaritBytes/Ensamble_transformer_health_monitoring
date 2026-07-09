const FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
];

let trendChart = null;

// Initialize Chart.js
function initChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";
    
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Health Confidence (%)',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointBackgroundColor: '#10b981',
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            }
        }
    });
}

// Generate KPI Cards structure
function initKPICards() {
    const container = document.getElementById('kpi-container');
    FEATURE_COLS.forEach(col => {
        const card = document.createElement('div');
        card.className = 'kpi-card';
        card.innerHTML = `
            <div class="kpi-label">${col.replace('_', ' ')}</div>
            <div class="kpi-value" id="kpi-${col}">--</div>
        `;
        container.appendChild(card);
    });
}

// Update UI elements with latest data
function updateUI(history) {
    if (history.length === 0) return;
    
    // Update Chart
    const labels = history.map(item => new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}));
    const dataPoints = history.map(item => {
        // We want to graph "Health Confidence" - so if it's unhealthy, maybe it's lower.
        // Actually, confidence is just max probability. 
        // Let's graph the probability of being 'Healthy' to show a clear trend line.
        return item.probabilities['Healthy'] * 100;
    });
    
    trendChart.data.labels = labels.slice(-20); // show last 20 points
    trendChart.data.datasets[0].data = dataPoints.slice(-20);
    trendChart.data.datasets[0].label = 'Healthy Probability (%)';
    trendChart.update();
    
    // Update Latest Status
    const latest = history[history.length - 1];
    
    const statusText = document.getElementById('current-health');
    statusText.innerText = latest.status;
    
    const confText = document.getElementById('current-conf');
    confText.innerText = latest.confidence.toFixed(1);
    
    const indicator = document.querySelector('.status-indicator');
    const globalStatus = document.getElementById('global-status');
    globalStatus.innerText = "System Active";
    
    if (latest.status === 'Healthy') {
        statusText.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        indicator.style.color = 'var(--status-healthy)';
    } else if (latest.status === 'About to be Unhealthy') {
        statusText.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        indicator.style.color = 'var(--status-warning)';
    } else {
        statusText.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
        indicator.style.color = 'var(--status-danger)';
    }
    statusText.style.webkitBackgroundClip = 'text';
    
    // Update Probabilities
    ['Healthy', 'About to be Unhealthy', 'Unhealthy'].forEach(cls => {
        let id_suffix = cls === 'Healthy' ? 'healthy' : (cls === 'Unhealthy' ? 'danger' : 'warning');
        let prob = (latest.probabilities[cls] * 100).toFixed(1);
        
        document.getElementById(`val-${id_suffix}`).innerText = `${prob}%`;
        document.getElementById(`prob-${id_suffix}`).style.width = `${prob}%`;
    });
    
    // Update KPI Cards
    for (const [key, val] of Object.entries(latest.features)) {
        const el = document.getElementById(`kpi-${key}`);
        if (el) {
            // Add slight animation class if value changed
            const oldVal = el.innerText;
            const newVal = val.toFixed(2);
            if(oldVal !== newVal && oldVal !== '--') {
                el.parentElement.style.transform = 'scale(1.05)';
                setTimeout(() => el.parentElement.style.transform = 'translateY(0)', 200);
            }
            el.innerText = newVal;
        }
    }
}

// Fetch Alerts
async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts');
        const alertState = await res.json();
        
        const banner = document.getElementById('alert-banner');
        const alertText = document.getElementById('alert-text');
        
        if (alertState.active) {
            banner.classList.remove('hidden');
            alertText.innerText = alertState.message;
        } else {
            banner.classList.add('hidden');
        }
    } catch (e) {
        console.error("Alert fetch failed", e);
    }
}

// Polling Loop
async function pollData() {
    try {
        const res = await fetch('/api/history');
        if (res.ok) {
            const history = await res.json();
            updateUI(history);
        }
    } catch(e) {
        console.error("Failed to fetch history:", e);
        document.getElementById('global-status').innerText = "Connection Lost";
        document.querySelector('.status-indicator').style.color = "var(--status-danger)";
    }
    
    await fetchAlerts();
    
    setTimeout(pollData, 2000); // Poll every 2 seconds
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initKPICards();
    pollData();
});
