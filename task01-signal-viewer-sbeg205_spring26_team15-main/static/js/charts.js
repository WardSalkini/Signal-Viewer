/**
 * charts.js — All Plotly.js visualization functions
 */
const Charts = {
  plotlyLayout: {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#161b22',
    font: { color: '#e6edf3' },
    margin: { t: 50, b: 80, l: 60, r: 30 },
  },

  /** Stacked bar chart: bacterial abundance per sample, colored by phylum */
  abundanceBar(data, container) {
    const bacteria = DataLoader.getBacteriaColumns(data);
    const sampleIDs = data.map((r) => r.SampleID);

    const colors = [
      '#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7',
      '#79c0ff', '#56d364', '#e3b341', '#ff7b72', '#bc8cff',
    ];

    const traces = bacteria.map((b, i) => ({
      x: sampleIDs,
      y: data.map((r) => r[b]),
      name: b,
      type: 'bar',
      marker: { color: colors[i % colors.length] },
    }));

    Plotly.newPlot(container, traces, {
      ...this.plotlyLayout,
      barmode: 'stack',
      title: 'Bacterial Abundance per Sample (%)',
      xaxis: { title: 'Sample ID', tickangle: -45 },
      yaxis: { title: 'Relative Abundance (%)' },
      legend: { orientation: 'h', y: -0.3 },
    });
  },

  /** Heatmap: samples vs bacteria */
  heatmap(data, container) {
    const bacteria = DataLoader.getBacteriaColumns(data);
    const sampleIDs = data.map((r) => `${r.SampleID} (${r.Diagnosis})`);
    const z = data.map((r) => bacteria.map((b) => r[b]));

    Plotly.newPlot(container, [{
      z, x: bacteria, y: sampleIDs,
      type: 'heatmap',
      colorscale: 'Viridis',
      colorbar: { title: 'Abundance %' },
    }], {
      ...this.plotlyLayout,
      title: 'Microbiome Heatmap (Samples × Bacteria)',
      xaxis: { title: 'Bacterial Phylum' },
      yaxis: { title: 'Sample', autorange: 'reversed' },
      height: Math.max(500, data.length * 18),
    });
  },

  /** Pie chart: average composition across all samples or a diagnosis group */
  compositionPie(data, container, diagnosis = null) {
    const bacteria = DataLoader.getBacteriaColumns(data);
    const filtered = diagnosis ? data.filter((r) => r.Diagnosis === diagnosis) : data;
    const label = diagnosis || 'All Samples';

    const avgValues = bacteria.map((b) => {
      const sum = filtered.reduce((acc, r) => acc + (r[b] || 0), 0);
      return +(sum / filtered.length).toFixed(2);
    });

    // If we have diagnoses, show one pie per diagnosis
    const diagnoses = [...new Set(data.map((r) => r.Diagnosis))];
    const traces = [];
    const cols = Math.min(diagnoses.length, 3);
    const rows = Math.ceil(diagnoses.length / cols);

    diagnoses.forEach((diag, i) => {
      const subset = data.filter((r) => r.Diagnosis === diag);
      const vals = bacteria.map((b) => {
        const sum = subset.reduce((acc, r) => acc + (r[b] || 0), 0);
        return +(sum / subset.length).toFixed(2);
      });
      const col = i % cols;
      const row = Math.floor(i / cols);
      traces.push({
        labels: bacteria, values: vals,
        type: 'pie', name: diag,
        title: { text: `${diag} (n=${subset.length})` },
        domain: {
          x: [col / cols + 0.02, (col + 1) / cols - 0.02],
          y: [1 - (row + 1) / rows + 0.05, 1 - row / rows - 0.05],
        },
        hole: 0.35,
        textinfo: 'label+percent',
        textfont: { size: 10 },
      });
    });

    Plotly.newPlot(container, traces, {
      ...this.plotlyLayout,
      title: 'Bacterial Composition by Diagnosis Group',
      height: rows * 350,
      showlegend: false,
    });
  },

  /** Shannon Diversity Index per sample, grouped by diagnosis */
  diversityPlot(data, container) {
    const bacteria = DataLoader.getBacteriaColumns(data);

    // Calculate Shannon Diversity for each sample
    const diversities = data.map((r) => {
      const values = bacteria.map((b) => r[b] / 100).filter((v) => v > 0);
      const H = -values.reduce((sum, p) => sum + p * Math.log(p), 0);
      return { sample: r.SampleID, diagnosis: r.Diagnosis, H: +H.toFixed(4) };
    });

    const diagnoses = [...new Set(data.map((r) => r.Diagnosis))];
    const traces = diagnoses.map((diag) => {
      const subset = diversities.filter((d) => d.diagnosis === diag);
      return {
        y: subset.map((d) => d.H),
        name: diag,
        type: 'box',
        boxmean: true,
      };
    });

    Plotly.newPlot(container, traces, {
      ...this.plotlyLayout,
      title: 'Shannon Diversity Index by Diagnosis',
      yaxis: { title: "Shannon Index (H')" },
      xaxis: { title: 'Diagnosis Group' },
    });
  },
};