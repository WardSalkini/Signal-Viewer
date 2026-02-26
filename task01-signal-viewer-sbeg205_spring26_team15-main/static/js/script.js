// script.js — Real data version using Yahoo Finance via Flask backend

document.addEventListener('DOMContentLoaded', () => {
    const watchlist        = document.getElementById('watchlist');
    const stockInfo        = document.getElementById('stock-info');
    const searchInput      = document.getElementById('search');
    const stockTableBody   = document.querySelector('#stock-table tbody');
    const chartCanvas      = document.getElementById('stock-chart');
    const chartPlaceholder = document.getElementById('chart-placeholder');
    const chartTitle       = document.getElementById('chart-title');
    const timeframeBtns    = document.querySelectorAll('.timeframe-btn');

    let stockChart    = null;
    let selectedStock = null;
    let selectedPeriod = '1M';

    // ── Default watchlist symbols ─────────────────────────────────────────────
    let watchlistSymbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT'];

    // ── Master stock list (symbols + display names) ───────────────────────────
    // Prices/changes will be fetched from the backend; static names stay here.
    const allStocksMeta = [
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
        { symbol: 'ZM',    name: 'Zoom Video Communications Inc.' },
        { symbol: 'SHOP',  name: 'Shopify Inc.' },
        { symbol: 'WMT',   name: 'Walmart Inc.' },
        { symbol: 'TGT',   name: 'Target Corporation' },
        { symbol: 'MCD',   name: "McDonald's Corporation" },
        { symbol: 'SBUX',  name: 'Starbucks Corporation' },
        { symbol: 'NKE',   name: 'NIKE Inc.' },
        { symbol: 'BA',    name: 'The Boeing Company' },
        { symbol: 'GE',    name: 'General Electric Company' },
        { symbol: 'F',     name: 'Ford Motor Company' },
        { symbol: 'GM',    name: 'General Motors Company' },
        { symbol: 'BABA',  name: 'Alibaba Group Holding Ltd.' },
        { symbol: 'HSBC',  name: 'HSBC Holdings plc' },
    ];

    // Runtime array — enriched with live price/change after fetch
    let allStocks = allStocksMeta.map(s => ({ ...s, price: null, change: null }));

    // ── Cache to avoid hammering the API ─────────────────────────────────────
    const quoteCache   = {};  // symbol → { price, change, timestamp }
    const historyCache = {};  // `${symbol}_${period}` → { labels, data }
    const CACHE_TTL_MS = 60_000; // 60 seconds

    // ── Helpers ───────────────────────────────────────────────────────────────
    function showLoading(el, msg = 'Loading…') {
        el.innerHTML = `<p class="placeholder-text">${msg}</p>`;
    }

    function showError(el, msg) {
        el.innerHTML = `<p class="placeholder-text" style="color:#ef4444">${msg}</p>`;
    }

    function getPeriodParam(period) {
        // Maps UI period label → yfinance period string
        switch (period) {
            case '1W': return '5d';
            case '1M': return '1mo';
            case '3M': return '3mo';
            case '1Y': return '1y';
            default:   return '1mo';
        }
    }

    // ── API calls to Flask backend ────────────────────────────────────────────

    /**
     * Fetch real-time quote for one symbol.
     * GET /api/stocks/quote?symbol=AAPL
     * Returns { symbol, price, change, change_pct, name, market_cap, volume }
     */
    async function fetchQuote(symbol) {
        const cached = quoteCache[symbol];
        if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
            return cached.data;
        }

        const res  = await fetch(`/api/stocks/quote?symbol=${encodeURIComponent(symbol)}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch quote');

        quoteCache[symbol] = { data: json, timestamp: Date.now() };
        return json;
    }

    /**
     * Fetch bulk quotes for many symbols at once.
     * GET /api/stocks/bulk?symbols=AAPL,MSFT,TSLA
     * Returns { success, quotes: { AAPL: {...}, MSFT: {...}, ... } }
     */
    async function fetchBulkQuotes(symbols) {
        const res  = await fetch(`/api/stocks/bulk?symbols=${symbols.join(',')}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch bulk quotes');
        return json.quotes; // { SYMBOL: { price, change_pct, ... } }
    }

    /**
     * Fetch historical OHLCV data.
     * GET /api/stocks/history?symbol=AAPL&period=1mo
     * Returns { success, labels: [...], prices: [...] }
     */
    async function fetchHistory(symbol, period) {
        const key    = `${symbol}_${period}`;
        const cached = historyCache[key];
        if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
            return cached.data;
        }

        const yPeriod = getPeriodParam(period);
        const res  = await fetch(`/api/stocks/history?symbol=${encodeURIComponent(symbol)}&period=${yPeriod}`);
        const json = await res.json();
        if (!json.success) throw new Error(json.error || 'Failed to fetch history');

        historyCache[key] = { data: json, timestamp: Date.now() };
        return json;
    }

    // ── Chart rendering ───────────────────────────────────────────────────────
    function renderChart(labels, prices, symbol, period) {
        const priceStart = prices[0];
        const priceEnd   = prices[prices.length - 1];
        const isPositive = priceEnd >= priceStart;
        const lineColor  = isPositive ? '#22c55e' : '#ef4444';

        chartCanvas.style.display   = 'block';
        chartPlaceholder.style.display = 'none';
        chartTitle.textContent      = `${symbol} Price History (${period})`;

        if (stockChart) stockChart.destroy();

        const ctx      = chartCanvas.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, chartCanvas.height);
        gradient.addColorStop(0, isPositive ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)');
        gradient.addColorStop(1, 'rgba(255,255,255,0)');

        const days = prices.length;

        stockChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: `${symbol} Price ($)`,
                    data: prices,
                    borderColor:       lineColor,
                    backgroundColor:   gradient,
                    borderWidth:       2,
                    fill:              true,
                    tension:           0.3,
                    pointRadius:       days <= 7 ? 4 : days <= 30 ? 2 : 0,
                    pointHoverRadius:  5,
                    pointBackgroundColor: lineColor,
                    pointBorderColor:     '#fff',
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
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#fff',
                        bodyColor:  '#fff',
                        borderColor: lineColor,
                        borderWidth: 1,
                        padding: 10,
                        displayColors: false,
                        callbacks: {
                            label: ctx => `$${ctx.parsed.y.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            maxTicksLimit: days <= 7 ? 5 : days <= 30 ? 8 : days <= 90 ? 6 : 12,
                            color: '#666',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.05)' },
                        ticks: {
                            color: '#666',
                            font: { size: 11 },
                            callback: v => '$' + v.toFixed(0)
                        }
                    }
                }
            }
        });
    }

    // ── Display stock details + chart ─────────────────────────────────────────
    async function displayStockInfo(stock) {
        selectedStock = stock;
        showLoading(stockInfo, 'Fetching live quote…');
        chartCanvas.style.display = 'none';
        chartPlaceholder.style.display = 'block';
        chartPlaceholder.textContent = 'Loading chart…';

        try {
            const quote = await fetchQuote(stock.symbol);

            const changeClass = quote.change_pct >= 0 ? 'positive' : 'negative';
            const arrow       = quote.change_pct >= 0 ? '&#9650;' : '&#9660;';
            const mcap        = quote.market_cap
                ? `<p style="color:#888;font-size:0.9em">Mkt Cap: $${(quote.market_cap / 1e9).toFixed(2)}B &nbsp;|&nbsp; Vol: ${(quote.volume / 1e6).toFixed(2)}M</p>`
                : '';

            stockInfo.innerHTML = `
                <h3>${quote.name || stock.name} <span style="color:#888">(${stock.symbol})</span></h3>
                <p class="stock-price">$${quote.price.toFixed(2)}</p>
                <p class="stock-change ${changeClass}">
                    ${arrow} ${Math.abs(quote.change).toFixed(2)} (${Math.abs(quote.change_pct).toFixed(2)}%)
                </p>
                ${mcap}
                <p style="color:#aaa;font-size:0.8em;margin-top:6px">Live data via Yahoo Finance</p>
            `;

            // Fetch and render price history
            await loadChart(stock.symbol, selectedPeriod);

        } catch (err) {
            showError(stockInfo, `Could not load ${stock.symbol}: ${err.message}`);
        }
    }

    async function loadChart(symbol, period) {
        try {
            const hist = await fetchHistory(symbol, period);
            renderChart(hist.labels, hist.prices, symbol, period);
        } catch (err) {
            chartPlaceholder.textContent = `Chart error: ${err.message}`;
        }
    }

    // ── Watchlist ─────────────────────────────────────────────────────────────
    async function updateWatchlist() {
        watchlist.innerHTML = '<li style="color:#999;padding:8px">Refreshing…</li>';
        try {
            const quotes = await fetchBulkQuotes(watchlistSymbols);
            watchlist.innerHTML = '';

            watchlistSymbols.forEach(symbol => {
                const meta  = allStocksMeta.find(s => s.symbol === symbol) || { name: symbol };
                const quote = quotes[symbol] || {};
                const price = quote.price ?? 0;
                const pct   = quote.change_pct ?? 0;

                const li          = document.createElement('li');
                const changeClass = pct >= 0 ? 'positive' : 'negative';
                const arrow       = pct >= 0 ? '▲' : '▼';

                li.innerHTML = `
                    <span class="watchlist-name">${symbol}</span>
                    <span class="watchlist-price">$${price.toFixed(2)}</span>
                    <span class="watchlist-change ${changeClass}">${arrow} ${Math.abs(pct).toFixed(2)}%</span>
                `;

                const removeBtn = document.createElement('button');
                removeBtn.textContent = 'Remove';
                removeBtn.classList.add('remove');
                removeBtn.addEventListener('click', e => {
                    e.stopPropagation();
                    watchlistSymbols = watchlistSymbols.filter(s => s !== symbol);
                    updateWatchlist();
                    displayStockTable(filteredStocks());
                });
                li.appendChild(removeBtn);

                li.addEventListener('click', () => {
                    displayStockInfo({ symbol, name: meta.name });
                });

                watchlist.appendChild(li);
            });
        } catch (err) {
            watchlist.innerHTML = `<li style="color:#ef4444">Error: ${err.message}</li>`;
        }
    }

    // ── Stock table ───────────────────────────────────────────────────────────
    function filteredStocks() {
        const term = searchInput.value.toLowerCase();
        return allStocks.filter(s =>
            s.name.toLowerCase().includes(term) || s.symbol.toLowerCase().includes(term)
        );
    }

    function displayStockTable(stocks) {
        stockTableBody.innerHTML = '';
        stocks.forEach(stock => {
            const price  = stock.price  != null ? `$${stock.price.toFixed(2)}` : '—';
            const change = stock.change != null
                ? `${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)}%`
                : '—';
            const changeClass = stock.change != null
                ? (stock.change >= 0 ? 'positive' : 'negative')
                : '';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${stock.name} <span class="stock-symbol">(${stock.symbol})</span></td>
                <td>${price}</td>
                <td class="${changeClass}">${change}</td>
                <td class="actions"></td>
            `;

            const actionsTd = tr.querySelector('.actions');
            const inWatchlist = watchlistSymbols.includes(stock.symbol);

            const btn = document.createElement('button');
            btn.textContent = inWatchlist ? 'Remove' : 'Add';
            btn.classList.add(inWatchlist ? 'remove' : 'add');
            btn.addEventListener('click', e => {
                e.stopPropagation();
                if (inWatchlist) {
                    watchlistSymbols = watchlistSymbols.filter(s => s !== stock.symbol);
                } else {
                    watchlistSymbols.push(stock.symbol);
                }
                updateWatchlist();
                displayStockTable(filteredStocks());
            });
            actionsTd.appendChild(btn);

            tr.addEventListener('click', () => displayStockInfo(stock));
            stockTableBody.appendChild(tr);
        });
    }

    // ── Load initial bulk quotes for the whole table ──────────────────────────
    async function loadAllQuotes() {
        const symbols = allStocks.map(s => s.symbol);
        try {
            const quotes = await fetchBulkQuotes(symbols);
            allStocks = allStocks.map(s => ({
                ...s,
                price:  quotes[s.symbol]?.price  ?? null,
                change: quotes[s.symbol]?.change_pct ?? null,
            }));
            displayStockTable(filteredStocks());
        } catch (err) {
            console.warn('Bulk quote load failed:', err.message);
            displayStockTable(filteredStocks()); // show table with dashes
        }
    }

    // ── Timeframe buttons ─────────────────────────────────────────────────────
    timeframeBtns.forEach(btn => {
        btn.addEventListener('click', e => {
            e.stopPropagation();
            timeframeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedPeriod = btn.dataset.period;
            if (selectedStock) loadChart(selectedStock.symbol, selectedPeriod);
        });
    });

    // ── Search ────────────────────────────────────────────────────────────────
    searchInput.addEventListener('input', () => displayStockTable(filteredStocks()));

    // ── Auto-refresh every 60 seconds ─────────────────────────────────────────
    setInterval(() => {
        updateWatchlist();
        if (selectedStock) displayStockInfo(selectedStock);
    }, 60_000);

    // ── Init ──────────────────────────────────────────────────────────────────
    displayStockTable(allStocks);   // show table immediately (prices show as —)
    updateWatchlist();              // load watchlist prices
    loadAllQuotes();                // then populate all prices
});