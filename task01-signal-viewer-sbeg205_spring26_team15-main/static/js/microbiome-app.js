/**
 * app.js — Main application controller
 */
(function () {
  let appData = null;

  const els = {
    csvUpload: document.getElementById('csvUpload'),
    statsSection: document.getElementById('stats-section'),
    statsCards: document.getElementById('stats-cards'),
    vizSection: document.getElementById('viz-section'),
    chartContainer: document.getElementById('chart-container'),
    profilerSection: document.getElementById('profiler-section'),
    patientSelect: document.getElementById('patientSelect'),
    patientProfile: document.getElementById('patient-profile'),
    patientChart: document.getElementById('patient-chart'),
  };

  // ---- Event Listeners ----

  els.csvUpload.addEventListener('change', async (e) => {
    if (e.target.files.length === 0) return;
    try {
      appData = await DataLoader.parseCSV(e.target.files[0]);
      initDashboard();
    } catch (err) {
      alert('Error parsing file: ' + err.message);
    }
  });

  // Tabs
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      renderChart(tab.dataset.tab);
    });
  });

  // Patient selector
  els.patientSelect.addEventListener('change', () => {
    const sample = appData.find((r) => r.SampleID === els.patientSelect.value);
    if (sample) {
      PatientProfiler.renderProfile(sample, appData, els.patientProfile, els.patientChart);
    }
  });

  // ---- Dashboard Init ----

  function initDashboard() {
    if (!appData || appData.length === 0) return;

    // Show sections
    els.statsSection.classList.remove('hidden');
    els.vizSection.classList.remove('hidden');
    els.profilerSection.classList.remove('hidden');

    // Stats
    const bacteria = DataLoader.getBacteriaColumns(appData);
    const diagnoses = [...new Set(appData.map((r) => r.Diagnosis))];
    const patients = [...new Set(appData.map((r) => r.PatientID))];

    els.statsCards.innerHTML = `
      <div class="stat-card"><div class="value">${appData.length}</div><div class="label">Samples</div></div>
      <div class="stat-card"><div class="value">${patients.length}</div><div class="label">Patients</div></div>
      <div class="stat-card"><div class="value">${bacteria.length}</div><div class="label">Bacterial Taxa</div></div>
      <div class="stat-card"><div class="value">${diagnoses.length}</div><div class="label">Diagnosis Groups</div></div>
    `;

    // Populate patient selector
    els.patientSelect.innerHTML = appData
      .map((r) => `<option value="${r.SampleID}">${r.SampleID} — ${r.PatientID} (${r.Diagnosis})</option>`)
      .join('');

    // Default chart
    renderChart('abundance');

    // Default patient profile
    PatientProfiler.renderProfile(appData[0], appData, els.patientProfile, els.patientChart);
  }

  function renderChart(type) {
    els.chartContainer.innerHTML = '';
    switch (type) {
      case 'abundance': Charts.abundanceBar(appData, els.chartContainer); break;
      case 'heatmap': Charts.heatmap(appData, els.chartContainer); break;
      case 'pie': Charts.compositionPie(appData, els.chartContainer); break;
      case 'diversity': Charts.diversityPlot(appData, els.chartContainer); break;
    }
  }
})();