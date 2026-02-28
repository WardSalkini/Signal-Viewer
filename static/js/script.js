// script.js — Market Terminal
// Supports: Stocks | Currencies | Minerals
// Features: Live prices, history charts, watchlist, AI prediction (Prophet)

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ───────────────────────────────────────────────────────────
    const watchlistEl      = document.getElementById('watchlist');
    const watchlistCount   = document.getElementById('watchlist-count');
    const watchlistLabel   = document.getElementById('watchlist-label');
    const stockInfo        = document.getElementById('stock-info');
    const searchInput      = document.getElementById('search');
    const stockTableBody   = document.querySelector('#stock-table tbody');
    const chartCanvas      = document.getElementById('stock-chart');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    const chartTitle       = document.getElementById('chart-title');
    const timeframeBtns    = document.querySelectorAll('.timeframe-btn');
    const assetTabs        = document.querySelectorAll('.asset-tab');
    const liveClock        = document.getElementById('live-clock');

    // ── State ──────────────────────────────────────────────────────────────
    let stockChart     = null;
    let selectedAsset  = null;   // { symbol, name, type }
    let selectedPeriod = '1M';
    let predictionMode = false;
    let activeTab      = 'stocks';

    // ── Live clock ─────────────────────────────────────────────────────────
    function updateClock() {
        const now = new Date();
        liveClock.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }
    updateClock();
    setInterval(updateClock, 1000);

    // ══════════════════════════════════════════════════════════════════════
    // ASSET MASTER LISTS
    // ══════════════════════════════════════════════════════════════════════

    const stocksMeta = [
        { symbol: 'AAPL',  name: 'Apple Inc.' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.' },
        { symbol: 'AMZN',  name: 'Amazon.com Inc.' },
        { symbol: 'MSFT',  name: 'Microsoft Corporation' },
        { symbol: 'META',  name: 'Meta Platforms Inc.' },
        { symbol: 'TSLA',  name: 'Tesla Inc.' },
        { symbol: 'NFLX',  name: 'Netflix Inc.' },
        { symbol: 'NVDA',  name: 'NVIDIA Corporation' },
        { symbol: 'PYPL',  name: 'PayPal Holdings Inc.' },
        { symbol: 'INTC',  name: 'Intel Corporation' },
        { symbol: 'CSCO',  name: 'Cisco Systems Inc.' },
        { symbol: 'PEP',   name: 'PepsiCo Inc.' },
        { symbol: 'KO',    name: 'The Coca-Cola Company' },
        { symbol: 'PFE',   name: 'Pfizer Inc.' },
        { symbol: 'JNJ',   name: 'Johnson & Johnson' },
        { symbol: 'V',     name: 'Visa Inc.' },
        { symbol: 'MA',    name: 'Mastercard Inc.' },
        { symbol: 'DIS',   name: 'The Walt Disney Company' },
        { symbol: 'ADBE',  name: 'Adobe Inc.' },
        { symbol: 'CRM',   name: 'Salesforce Inc.' },
        { symbol: 'ORCL',  name: 'Oracle Corporation' },
        { symbol: 'IBM',   name: 'IBM Corporation' },
        { symbol: 'UBER',  name: 'Uber Technologies Inc.' },
        { symbol: 'SNAP',  name: 'Snap Inc.' },
        { symbol: 'SPOT',  name: 'Spotify Technology S.A.' },
        { symbol: 'WMT',   name: 'Walmart Inc.' },
        { symbol: 'NKE',   name: 'NIKE Inc.' },
        { symbol: 'BA',    name: 'The Boeing Company' },
        { symbol: 'GE',    name: 'General Electric Company' },
        { symbol: 'F',     name: 'Ford Motor Company' },
    ];

    // yfinance symbols for forex pairs: BASE/QUOTE → BASEQUOTE=X
    const currenciesMeta = [
        { symbol: 'EURUSD=X', name: 'Euro / US Dollar',          pair: 'EUR/USD' },
        { symbol: 'GBPUSD=X', name: 'British Pound / US Dollar', pair: 'GBP/USD' },
        { symbol: 'USDJPY=X', name: 'US Dollar / Japanese Yen',  pair: 'USD/JPY' },
        { symbol: 'USDCHF=X', name: 'US Dollar / Swiss Franc',   pair: 'USD/CHF' },
        { symbol: 'AUDUSD=X', name: 'Australian Dollar / USD',   pair: 'AUD/USD' },
        { symbol: 'USDCAD=X', name: 'US Dollar / Canadian Dollar',pair: 'USD/CAD' },
        { symbol: 'NZDUSD=X', name: 'New Zealand Dollar / USD',  pair: 'NZD/USD' },
        { symbol: 'USDCNY=X', name: 'US Dollar / Chinese Yuan',  pair: 'USD/CNY' },
        { symbol: 'USDINR=X', name: 'US Dollar / Indian Rupee',  pair: 'USD/INR' },
        { symbol: 'USDBRL=X', name: 'US Dollar / Brazilian Real',pair: 'USD/BRL' },
        { symbol: 'USDMXN=X', name: 'US Dollar / Mexican Peso',  pair: 'USD/MXN' },
        { symbol: 'USDZAR=X', name: 'US Dollar / South African Rand', pair: 'USD/ZAR' },
        { symbol: 'EURGBP=X', name: 'Euro / British Pound',      pair: 'EUR/GBP' },
        { symbol: 'EURJPY=X', name: 'Euro / Japanese Yen',       pair: 'EUR/JPY' },
        { symbol: 'GBPJPY=X', name: 'British Pound / Japanese Yen', pair: 'GBP/JPY' },
        { symbol: 'USDEGP=X', name: 'US Dollar / Egyptian Pound',pair: 'USD/EGP' },
        { symbol: 'USDSAR=X', name: 'US Dollar / Saudi Riyal',   pair: 'USD/SAR' },
        { symbol: 'USDAED=X', name: 'US Dollar / UAE Dirham',    pair: 'USD/AED' },
        { symbol: 'USDTRY=X', name: 'US Dollar / Turkish Lira',  pair: 'USD/TRY' },
        { symbol: 'USDKWD=X', name: 'US Dollar / Kuwaiti Dinar', pair: 'USD/KWD' },
    ];

    // yfinance futures symbols for commodities/minerals
    const mineralsMeta = [
        { symbol: 'GC=F',  name: 'Gold Futures',       unit: 'USD/oz' },
        { symbol: 'SI=F',  name: 'Silver Futures',      unit: 'USD/oz' },
        { symbol: 'PL=F',  name: 'Platinum Futures',    unit: 'USD/oz' },
        { symbol: 'PA=F',  name: 'Palladium Futures',   unit: 'USD/oz' },
        { symbol: 'HG=F',  name: 'Copper Futures',      unit: 'USD/lb' },
        { symbol: 'ALI=F', name: 'Aluminum Futures',    unit: 'USD/lb' },
        { symbol: 'ZN=F',  name: 'Zinc Futures',        unit: 'USD/t' },
        { symbol: 'NI=F',  name: 'Nickel Futures',      unit: 'USD/t' },
        { symbol: 'TIN=F', name: 'Tin Futures',         unit: 'USD/t' },
        { symbol: 'CL=F',  name: 'Crude Oil (WTI)',     unit: 'USD/bbl' },
        { symbol: 'BZ=F',  name: 'Crude Oil (Brent)',   unit: 'USD/bbl' },
        { symbol: 'NG=F',  name: 'Natural Gas Futures', unit: 'USD/MMBtu' },
        { symbol: 'LBS=F', name: 'Lumber Futures',      unit: 'USD/1000 bd ft' },
        { symbol: 'ZC=F',  name: 'Corn Futures',        unit: 'USD/bu' },
        { symbol: 'ZW=F',  name: 'Wheat Futures',       unit: 'USD/bu' },
    ];

    // Build runtime arrays with price slots
    let allStocks     = stocksMeta.map(s    => ({ ...s, type: 'stock',    price: null, change: null }));
    let allCurrencies = currenciesMeta.map(c => ({ ...c, type: 'currency', price: null, change: null }));
    let allMinerals   = mineralsMeta.map(m   => ({ ...m, type: 'mineral',  price: null, change: null }));

    function getCurrentList() {
        if (activeTab === 'stocks')     return allStocks;
        if (activeTab === 'currencies') return allCurrencies;
        return allMinerals;
    }

    // ── Watchlist (persisted in memory, spans all types) ──────────────────
    // Each entry: { symbol, name, type }
    let watchlistItems = [
        { symbol: 'AAPL',     name: 'Apple Inc.',            type: 'stock'    },
        { symbol: 'EURUSD=X', name: 'Euro / US Dollar',      type: 'currency' },
        { symbol: 'GC=F',     name: 'Gold Futures',           type: 'mineral'  },
    ];

    // ── Cache ──────────────────────────────────────────────────────────────
    const quoteCache      = {};
    const historyCache    = {};
    const predictionCache = {};
    const CACHE_TTL       = 60_000;
    const PRED_CACHE_TTL  = 600_000; // 10 min

    // ══════════════════════════════════════════════════════════════════════
    // HELPERS
    // ══════════════════════════════════════════════════════════════════════
    function showLoading(el, msg = 'Loading…') {
        el.innerHTML = `<p class="placeholder-text">${msg}</p>`;
    }
    function showError(el, msg) {
        el.innerHTML = `<p class="placeholder-text" style="color:var(--red)">${msg}</p>`;
    }

    function getPeriodParam(period) {
        return { '1W': '5d', '1M': '1mo', '3M': '3mo', '1Y': '1y' }[period] || '1mo';
    }

    function formatPrice(price, type) {
        if (price == null) return '—';
        if (type === 'currency') return price.toFixed(4);
        if (price >= 1000)       return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        return price.toFixed(2);
    }

    function getDisplayName(asset) {
        return asset.pair || asset.name;
    }

    function getDisplaySymbol(asset) {
        return asset.pair || asset.symbol;
    }

    // ══════════════════════════════════════════════════════════════════════
    // API
    // ══════════════════════════════════════════════════════════════════════
    async function fetchQuote(symbol) {
        const cached = quoteCache[symbol];
        if (cached && Date.now() - cached.timestamp < CACHE_TTL) return cached.data;
        const res  = await fetch(`/api/stocks/quote?symbol=${encodeURIComponent(symbol)}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch quote');
        quoteCache[symbol] = { data: json, timestamp: Date.now() };
        return json;
    }

    async function fetchBulkQuotes(symbols) {
        const res  = await fetch(`/api/stocks/bulk?symbols=${symbols.join(',')}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch bulk');
        return json.quotes;
    }

    async function fetchHistory(symbol, period) {
        const key    = `${symbol}_${period}`;
        const cached = historyCache[key];
        if (cached && Date.now() - cached.timestamp < CACHE_TTL) return cached.data;
        const yPeriod = getPeriodParam(period);
        const res  = await fetch(`/api/stocks/history?symbol=${encodeURIComponent(symbol)}&period=${yPeriod}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch history');
        historyCache[key] = { data: json, timestamp: Date.now() };
        return json;
    }

    async function fetchPrediction(symbol, days = 30) {
        const key    = `${symbol}_${days}`;
        const cached = predictionCache[key];
        if (cached && Date.now() - cached.timestamp < PRED_CACHE_TTL) return cached.data;
        const res  = await fetch(`/api/stocks/predict?symbol=${encodeURIComponent(symbol)}&days=${days}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Prediction failed');
        predictionCache[key] = { data: json, timestamp: Date.now() };
        return json;
    }

    // ══════════════════════════════════════════════════════════════════════
    // CHART RENDERING
    // ══════════════════════════════════════════════════════════════════════
    const CHART_GRID   = 'rgba(255,255,255,0.04)';
    const CHART_TICK   = '#505a72';

    function renderChart(labels, prices, asset, period) {
        const isUp      = prices[prices.length - 1] >= prices[0];
        const lineColor = isUp ? '#22d47a' : '#ff4f6a';

        chartCanvas.style.display      = 'block';
        chartPlaceholder.style.display = 'none';
        chartTitle.textContent         = `${getDisplaySymbol(asset)} · ${period}`;

        if (stockChart) stockChart.destroy();

        const ctx  = chartCanvas.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, chartCanvas.height);
        grad.addColorStop(0, isUp ? 'rgba(34,212,122,0.18)' : 'rgba(255,79,106,0.18)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');

        const days = prices.length;

        stockChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: getDisplaySymbol(asset),
                    data: prices,
                    borderColor:          lineColor,
                    backgroundColor:      grad,
                    borderWidth:          2,
                    fill:                 true,
                    tension:              0.35,
                    pointRadius:          days <= 7 ? 3 : 0,
                    pointHoverRadius:     5,
                    pointBackgroundColor: lineColor,
                    pointBorderColor:     '#0d0f14',
                    pointBorderWidth:     2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(19,22,30,0.95)',
                        titleColor:  '#e2e8f0',
                        bodyColor:   '#8892aa',
                        borderColor: lineColor,
                        borderWidth: 1,
                        padding:     12,
                        displayColors: false,
                        titleFont:   { family: "'DM Mono', monospace", size: 11 },
                        bodyFont:    { family: "'DM Mono', monospace", size: 12 },
                        callbacks: {
                            label: ctx => formatPrice(ctx.parsed.y, asset.type)
                        }
                    }
                },
                scales: {
                    x: {
                        grid:  { display: false },
                        border:{ display: false },
                        ticks: {
                            maxTicksLimit: days <= 7 ? 5 : days <= 30 ? 8 : 7,
                            color: CHART_TICK,
                            font: { family: "'DM Mono', monospace", size: 10 }
                        }
                    },
                    y: {
                        grid:  { color: CHART_GRID },
                        border:{ display: false },
                        ticks: {
                            color: CHART_TICK,
                            font: { family: "'DM Mono', monospace", size: 10 },
                            callback: v => formatPrice(v, asset.type)
                        }
                    }
                }
            }
        });
    }

    function renderPredictionChart(predData, asset) {
        const { history, forecast, trend } = predData;

        chartCanvas.style.display      = 'block';
        chartPlaceholder.style.display = 'none';
        const days = forecast.labels.length;
        chartTitle.textContent = `${getDisplaySymbol(asset)} · AI Forecast ${days}d · ${trend === 'up' ? '▲ Upward' : '▼ Downward'}`;

        if (stockChart) stockChart.destroy();

        const allLabels  = [...history.labels, ...forecast.labels];
        const histData   = [...history.prices, ...forecast.labels.map(() => NaN)];
        const fcData     = [...history.labels.map(() => NaN), ...forecast.yhat];
        const upperData  = [...history.labels.map(() => NaN), ...forecast.yhat_upper];
        const lowerData  = [...history.labels.map(() => NaN), ...forecast.yhat_lower];

        const isUp      = trend === 'up';
        const histColor = isUp ? '#22d47a' : '#ff4f6a';
        const fcColor   = '#7c6dfa';

        const ctx = chartCanvas.getContext('2d');

        stockChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: 'Historical',
                        data: histData,
                        borderColor:     histColor,
                        backgroundColor: isUp ? 'rgba(34,212,122,0.08)' : 'rgba(255,79,106,0.08)',
                        borderWidth: 2, fill: true, tension: 0.35,
                        pointRadius: 0, spanGaps: false,
                    },
                    {
                        label: 'Confidence High',
                        data: upperData,
                        borderColor: 'transparent', backgroundColor: 'rgba(124,109,250,0.1)',
                        borderWidth: 0, fill: '+1', tension: 0.35,
                        pointRadius: 0, spanGaps: false,
                    },
                    {
                        label: 'Confidence Low',
                        data: lowerData,
                        borderColor: 'rgba(124,109,250,0.25)', backgroundColor: 'transparent',
                        borderWidth: 1, borderDash: [4,4], fill: false, tension: 0.35,
                        pointRadius: 0, spanGaps: false,
                    },
                    {
                        label: 'Forecast',
                        data: fcData,
                        borderColor: fcColor, backgroundColor: 'transparent',
                        borderWidth: 2, borderDash: [6,3], fill: false, tension: 0.35,
                        pointRadius: 0, pointHoverRadius: 5, spanGaps: false,
                        pointBackgroundColor: fcColor,
                    },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            filter: item => !['Confidence High','Confidence Low'].includes(item.text),
                            color: CHART_TICK,
                            font: { family: "'DM Mono', monospace", size: 10 },
                            boxWidth: 12,
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(19,22,30,0.95)',
                        titleColor: '#e2e8f0', bodyColor: '#8892aa',
                        padding: 12, displayColors: true,
                        titleFont: { family: "'DM Mono', monospace", size: 11 },
                        bodyFont:  { family: "'DM Mono', monospace", size: 11 },
                        callbacks: {
                            label: ctx => {
                                if (isNaN(ctx.parsed.y)) return null;
                                return `${ctx.dataset.label}: ${formatPrice(ctx.parsed.y, asset.type)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false }, border: { display: false },
                        ticks: { maxTicksLimit: 10, color: CHART_TICK, font: { family: "'DM Mono', monospace", size: 10 } }
                    },
                    y: {
                        grid: { color: CHART_GRID }, border: { display: false },
                        ticks: { color: CHART_TICK, font: { family: "'DM Mono', monospace", size: 10 },
                            callback: v => formatPrice(v, asset.type) }
                    }
                }
            }
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // PREDICTION PANEL
    // ══════════════════════════════════════════════════════════════════════
    function buildPredictPanel(asset) {
        const existing = document.getElementById('predict-panel');
        if (existing) existing.remove();

        const panel = document.createElement('div');
        panel.id = 'predict-panel';
        panel.innerHTML = `
            <div class="predict-header">
                <span class="predict-title">🤖 AI Price Prediction</span>
                <span class="predict-subtitle" id="predict-model-badge">Chronos-T5-Large · Amazon (pretrained)</span>
            </div>
            <div class="predict-controls">
                <label for="predict-days">Forecast horizon:</label>
                <select id="predict-days">
                    <option value="7">7 days</option>
                    <option value="14">14 days</option>
                    <option value="30" selected>30 days</option>
                    <option value="60">60 days</option>
                    <option value="90">90 days</option>
                </select>
                <button id="run-predict-btn" class="predict-btn">Run Forecast</button>
            </div>
            <div id="predict-result"></div>
            <p class="predict-disclaimer">⚠️ For educational purposes only. Not financial advice.</p>
        `;

        const chartContainer = document.getElementById('chart-container');
        chartContainer.parentNode.insertBefore(panel, chartContainer);
        document.getElementById('run-predict-btn').addEventListener('click', () => runPrediction(asset));
    }

    async function runPrediction(asset) {
        const daysEl   = document.getElementById('predict-days');
        const resultEl = document.getElementById('predict-result');
        const btn      = document.getElementById('run-predict-btn');
        if (!daysEl || !resultEl) return;

        const days = parseInt(daysEl.value, 10);
        btn.disabled    = true;
        btn.textContent = 'Forecasting…';
        resultEl.innerHTML = `<p class="placeholder-text">Running Chronos transformer… <span style="font-size:0.85em;color:var(--text3)">(first run downloads ~1.5 GB weights)</span></p>`;

        try {
            const pred = await fetchPrediction(asset.symbol, days);
            predictionMode = true;

            // Update model badge with actual model name from backend
            const badgeEl = document.getElementById('predict-model-badge');
            if (badgeEl && pred.model) {
                badgeEl.textContent = pred.model.replace('amazon/', '') + ' · Amazon · zero-shot';
            }

            const diff    = pred.predicted_price - pred.current_price;
            const pct     = (diff / pred.current_price) * 100;
            const isUp    = diff >= 0;
            const arrow   = isUp ? '▲' : '▼';
            const cls     = isUp ? 'positive' : 'negative';

            resultEl.innerHTML = `
                <div class="predict-summary">
                    <div class="predict-stat">
                        <span class="predict-stat-label">Current</span>
                        <span class="predict-stat-value">${formatPrice(pred.current_price, asset.type)}</span>
                    </div>
                    <div class="predict-stat">
                        <span class="predict-stat-label">Predicted (${days}d)</span>
                        <span class="predict-stat-value ${cls}">${formatPrice(pred.predicted_price, asset.type)}</span>
                    </div>
                    <div class="predict-stat">
                        <span class="predict-stat-label">Expected Change</span>
                        <span class="predict-stat-value ${cls}">
                            ${arrow} ${Math.abs(diff).toFixed(asset.type === 'currency' ? 4 : 2)} (${Math.abs(pct).toFixed(1)}%)
                        </span>
                    </div>
                </div>
            `;

            renderPredictionChart(pred, asset);
        } catch (err) {
            resultEl.innerHTML = `<p class="placeholder-text" style="color:var(--red)">Error: ${err.message}</p>`;
        } finally {
            btn.disabled    = false;
            btn.textContent = 'Run Forecast';
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // DISPLAY ASSET DETAIL
    // ══════════════════════════════════════════════════════════════════════
    async function displayAssetInfo(asset) {
        selectedAsset  = asset;
        predictionMode = false;
        showLoading(stockInfo, 'Fetching live quote…');
        chartCanvas.style.display      = 'none';
        chartPlaceholder.style.display = 'block';
        chartPlaceholder.textContent   = 'Loading chart…';

        const old = document.getElementById('predict-panel');
        if (old) old.remove();

        try {
            const quote       = await fetchQuote(asset.symbol);
            const changeClass = quote.change_pct >= 0 ? 'positive' : 'negative';
            const arrow       = quote.change_pct >= 0 ? '▲' : '▼';
            const typeLabel   = asset.type;

            let metaHtml = '';
            if (asset.type === 'stock' && quote.market_cap) {
                metaHtml = `
                    <div class="detail-meta">
                        <span>Mkt Cap: $${(quote.market_cap/1e9).toFixed(2)}B</span>
                        <span>Vol: ${(quote.volume/1e6).toFixed(2)}M</span>
                        <span>${quote.currency || 'USD'}</span>
                    </div>`;
            } else if (asset.type === 'currency') {
                metaHtml = `<div class="detail-meta"><span>Forex · ${asset.pair}</span></div>`;
            } else if (asset.type === 'mineral') {
                metaHtml = `<div class="detail-meta"><span>${asset.unit}</span></div>`;
            }

            stockInfo.innerHTML = `
                <div class="stock-info-row">
                    <div>
                        <span class="asset-type-tag ${typeLabel}">${typeLabel}</span>
                        <h3>${getDisplayName(asset)} <span style="color:var(--text3);font-size:0.8em">(${getDisplaySymbol(asset)})</span></h3>
                        <p class="stock-price">${formatPrice(quote.price, asset.type)}</p>
                        <p class="stock-change ${changeClass}">
                            ${arrow} ${Math.abs(quote.change).toFixed(asset.type === 'currency' ? 4 : 2)}
                            (${Math.abs(quote.change_pct).toFixed(2)}%)
                        </p>
                        ${metaHtml}
                        <p class="data-source">Live data via Yahoo Finance</p>
                    </div>
                    <button id="show-predict-btn" class="predict-toggle-btn">🤖 AI Predict</button>
                </div>
            `;

            document.getElementById('show-predict-btn').addEventListener('click', () => {
                buildPredictPanel(asset);
                document.getElementById('show-predict-btn').style.display = 'none';
            });

            await loadChart(asset, selectedPeriod);

        } catch (err) {
            showError(stockInfo, `Could not load ${asset.symbol}: ${err.message}`);
        }
    }

    async function loadChart(asset, period) {
        try {
            const hist = await fetchHistory(asset.symbol, period);
            renderChart(hist.labels, hist.prices, asset, period);
        } catch (err) {
            chartPlaceholder.style.display = 'block';
            chartPlaceholder.textContent   = `Chart error: ${err.message}`;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // WATCHLIST
    // ══════════════════════════════════════════════════════════════════════
    async function updateWatchlist() {
        if (watchlistItems.length === 0) {
            watchlistEl.innerHTML = `<li style="color:var(--text3);padding:10px;font-size:0.85em">No items — add assets below</li>`;
            watchlistCount.textContent = '0';
            return;
        }

        watchlistEl.innerHTML = '<li style="color:var(--text3);padding:8px;font-size:0.82em">Refreshing…</li>';
        watchlistCount.textContent = watchlistItems.length;

        try {
            const symbols = watchlistItems.map(w => w.symbol);
            const quotes  = await fetchBulkQuotes(symbols);

            watchlistEl.innerHTML = '';
            watchlistItems.forEach(item => {
                const quote = quotes[item.symbol] || {};
                const price = quote.price ?? null;
                const pct   = quote.change_pct ?? 0;

                const li          = document.createElement('li');
                const changeClass = pct >= 0 ? 'positive' : 'negative';
                const arrow       = pct >= 0 ? '▲' : '▼';
                const displaySym  = item.pair || item.symbol;

                li.innerHTML = `
                    <span class="watchlist-type-badge ${item.type}">${item.type.slice(0,3)}</span>
                    <span class="watchlist-name">${displaySym}</span>
                    <span class="watchlist-price">${price != null ? formatPrice(price, item.type) : '—'}</span>
                    <span class="watchlist-change ${changeClass}">${arrow} ${Math.abs(pct).toFixed(2)}%</span>
                `;

                const removeBtn = document.createElement('button');
                removeBtn.textContent = 'Remove';
                removeBtn.classList.add('remove');
                removeBtn.addEventListener('click', e => {
                    e.stopPropagation();
                    watchlistItems = watchlistItems.filter(w => w.symbol !== item.symbol);
                    updateWatchlist();
                    renderTable(filteredAssets());
                });
                li.appendChild(removeBtn);
                li.addEventListener('click', () => displayAssetInfo(item));
                watchlistEl.appendChild(li);
            });
            watchlistCount.textContent = watchlistItems.length;
        } catch (err) {
            watchlistEl.innerHTML = `<li style="color:var(--red);font-size:0.82em;padding:8px">Error: ${err.message}</li>`;
        }
    }

    function isInWatchlist(symbol) {
        return watchlistItems.some(w => w.symbol === symbol);
    }

    // ══════════════════════════════════════════════════════════════════════
    // TABLE
    // ══════════════════════════════════════════════════════════════════════
    function filteredAssets() {
        const term = searchInput.value.toLowerCase();
        return getCurrentList().filter(a =>
            a.name.toLowerCase().includes(term) ||
            a.symbol.toLowerCase().includes(term) ||
            (a.pair && a.pair.toLowerCase().includes(term))
        );
    }

    function renderTable(assets) {
        stockTableBody.innerHTML = '';
        assets.forEach(asset => {
            const priceStr  = formatPrice(asset.price, asset.type);
            const changeStr = asset.change != null
                ? `${asset.change >= 0 ? '+' : ''}${asset.change.toFixed(2)}%`
                : '—';
            const changeClass = asset.change != null
                ? (asset.change >= 0 ? 'positive' : 'negative')
                : '';

            const typeBadge = asset.type !== 'stock'
                ? `<span class="table-type-badge ${asset.type}">${asset.type}</span>`
                : '';

            const displayName = asset.pair
                ? `${asset.pair} <span class="stock-symbol">${asset.name}</span>`
                : `${asset.name} <span class="stock-symbol">(${asset.symbol})</span>`;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${displayName}${typeBadge}</td>
                <td class="price-cell">${priceStr}</td>
                <td class="change-cell ${changeClass}">${changeStr}</td>
                <td class="actions"></td>
            `;

            const actionsTd = tr.querySelector('.actions');
            const inWL = isInWatchlist(asset.symbol);
            const btn  = document.createElement('button');
            btn.textContent = inWL ? 'Remove' : 'Add';
            btn.classList.add(inWL ? 'remove' : 'add');
            btn.addEventListener('click', e => {
                e.stopPropagation();
                if (isInWatchlist(asset.symbol)) {
                    watchlistItems = watchlistItems.filter(w => w.symbol !== asset.symbol);
                } else {
                    watchlistItems.push({ symbol: asset.symbol, name: asset.name, type: asset.type, pair: asset.pair });
                }
                updateWatchlist();
                renderTable(filteredAssets());
            });
            actionsTd.appendChild(btn);
            tr.addEventListener('click', () => displayAssetInfo(asset));
            stockTableBody.appendChild(tr);
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // BULK QUOTE LOADERS
    // ══════════════════════════════════════════════════════════════════════
    async function loadBulkForList(list) {
        const symbols = list.map(a => a.symbol);
        try {
            const quotes = await fetchBulkQuotes(symbols);
            list.forEach(a => {
                const q = quotes[a.symbol];
                a.price  = q?.price      ?? null;
                a.change = q?.change_pct ?? null;
            });
        } catch (err) {
            console.warn('Bulk quote failed:', err.message);
        }
        renderTable(filteredAssets());
    }

    // ══════════════════════════════════════════════════════════════════════
    // TABS
    // ══════════════════════════════════════════════════════════════════════
    assetTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            assetTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeTab = tab.dataset.asset;
            searchInput.value = '';

            // Reset detail panel
            selectedAsset  = null;
            predictionMode = false;
            stockInfo.innerHTML = `<p class="placeholder-text">Select an asset to view details</p>`;
            chartCanvas.style.display      = 'none';
            chartPlaceholder.style.display = 'block';
            chartPlaceholder.textContent   = 'Select an asset to view its price chart';
            const old = document.getElementById('predict-panel');
            if (old) old.remove();

            renderTable(filteredAssets());

            // Lazy-load prices if not yet fetched
            const list = getCurrentList();
            if (list.some(a => a.price === null)) {
                loadBulkForList(list);
            }
        });
    });

    // ══════════════════════════════════════════════════════════════════════
    // TIMEFRAME BUTTONS
    // ══════════════════════════════════════════════════════════════════════
    timeframeBtns.forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            timeframeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedPeriod = btn.dataset.period;
            if (selectedAsset && !predictionMode) loadChart(selectedAsset, selectedPeriod);
        });
    });

    // ── Search ─────────────────────────────────────────────────────────────
    searchInput.addEventListener('input', () => renderTable(filteredAssets()));

    // ── Auto-refresh every 60 s ────────────────────────────────────────────
    setInterval(() => {
        updateWatchlist();
        if (selectedAsset && !predictionMode) displayAssetInfo(selectedAsset);
    }, 60_000);

    // ══════════════════════════════════════════════════════════════════════
    // INIT
    // ══════════════════════════════════════════════════════════════════════
    renderTable(allStocks);         // show table immediately
    updateWatchlist();              // load watchlist
    loadBulkForList(allStocks);     // fetch stock prices
});
