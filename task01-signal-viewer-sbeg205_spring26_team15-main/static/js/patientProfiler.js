/**
 * patientProfiler.js — Estimate patient risk/profile from microbiome signature
 */
const PatientProfiler = {
  /**
   * Known microbiome-disease associations (simplified, literature-based)
   */
  diseaseSignatures: {
    IBD: { Proteobacteria: 'high', Bacteroides: 'low', Firmicutes: 'low' },
    Obesity: { Firmicutes: 'high', Bacteroides: 'low' },
    T2D: { Actinobacteria: 'low', Firmicutes: 'high', Proteobacteria: 'high' },
    CRC: { Fusobacteria: 'high', Firmicutes: 'low' },
  },

  /**
   * Compute reference stats from the full dataset
   */
  computeReference(data) {
    const bacteria = DataLoader.getBacteriaColumns(data);
    const ref = {};
    bacteria.forEach((b) => {
      const values = data.map((r) => r[b]).filter((v) => v != null);
      const mean = values.reduce((a, c) => a + c, 0) / values.length;
      const std = Math.sqrt(values.reduce((a, c) => a + (c - mean) ** 2, 0) / values.length);
      ref[b] = { mean: +mean.toFixed(2), std: +std.toFixed(2) };
    });
    return ref;
  },

  /**
   * Estimate risk scores for a single patient sample
   */
  estimateProfile(sample, reference) {
    const bacteria = DataLoader.getBacteriaColumns([sample]);
    const results = {};

    for (const [disease, signature] of Object.entries(this.diseaseSignatures)) {
      let score = 0;
      let maxScore = 0;

      for (const [bact, direction] of Object.entries(signature)) {
        if (sample[bact] == null || !reference[bact]) continue;
        const zScore = (sample[bact] - reference[bact].mean) / (reference[bact].std || 1);
        maxScore += 2;

        if (direction === 'high' && zScore > 1) score += Math.min(zScore, 3);
        else if (direction === 'low' && zScore < -1) score += Math.min(Math.abs(zScore), 3);
      }

      const risk = maxScore > 0 ? Math.min((score / maxScore) * 100, 100) : 0;
      results[disease] = {
        score: +risk.toFixed(1),
        level: risk > 60 ? 'High' : risk > 30 ? 'Medium' : 'Low',
      };
    }

    return results;
  },

  /**
   * Render patient profile card and radar chart
   */
  renderProfile(sample, data, profileContainer, chartContainer) {
    const reference = this.computeReference(data);
    const risks = this.estimateProfile(sample, reference);
    const bacteria = DataLoader.getBacteriaColumns(data);

    // Shannon diversity
    const values = bacteria.map((b) => sample[b] / 100).filter((v) => v > 0);
    const shannon = -values.reduce((sum, p) => sum + p * Math.log(p), 0);

    // Profile card
    let html = `<h3>Patient: ${sample.PatientID} | Sample: ${sample.SampleID}</h3>`;
    html += `<div class="profile-item"><span class="profile-label">Age</span><span class="profile-value">${sample.Age}</span></div>`;
    html += `<div class="profile-item"><span class="profile-label">Sex</span><span class="profile-value">${sample.Sex}</span></div>`;
    html += `<div class="profile-item"><span class="profile-label">BMI</span><span class="profile-value">${sample.BMI}</span></div>`;
    html += `<div class="profile-item"><span class="profile-label">Body Site</span><span class="profile-value">${sample.BodySite}</span></div>`;
    html += `<div class="profile-item"><span class="profile-label">Diagnosis</span><span class="profile-value">${sample.Diagnosis}</span></div>`;
    html += `<div class="profile-item"><span class="profile-label">Shannon Diversity</span><span class="profile-value">${shannon.toFixed(3)}</span></div>`;
    html += `<hr style="border-color:#30363d;margin:12px 0">`;
    html += `<h4 style="color:#f0883e">Disease Risk Estimation</h4>`;

    for (const [disease, info] of Object.entries(risks)) {
      const badgeClass = info.level === 'High' ? 'risk-high' : info.level === 'Medium' ? 'risk-medium' : 'risk-low';
      html += `<div class="profile-item">
        <span class="profile-label">${disease}</span>
        <span class="profile-value">
          <span class="risk-badge ${badgeClass}">${info.level} (${info.score}%)</span>
        </span>
      </div>`;
    }
    profileContainer.innerHTML = html;

    // Radar chart for bacterial composition vs population mean
    const radarTrace1 = {
      type: 'scatterpolar', fill: 'toself', name: 'This Patient',
      r: bacteria.map((b) => sample[b]),
      theta: bacteria, opacity: 0.7,
    };
    const radarTrace2 = {
      type: 'scatterpolar', fill: 'toself', name: 'Population Mean',
      r: bacteria.map((b) => reference[b].mean),
      theta: bacteria, opacity: 0.5,
    };

    Plotly.newPlot(chartContainer, [radarTrace1, radarTrace2], {
      ...Charts.plotlyLayout,
      title: `Microbiome Profile: ${sample.SampleID} vs Population`,
      polar: {
        bgcolor: '#161b22',
        radialaxis: { visible: true, color: '#8b949e' },
        angularaxis: { color: '#e6edf3' },
      },
    });
  },
};