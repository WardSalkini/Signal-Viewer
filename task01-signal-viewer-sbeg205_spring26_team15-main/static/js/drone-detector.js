(function () {
    const uploadInput = document.getElementById('drone-upload');
    const detectBtn = document.getElementById('detect-btn');
    const micBtn = document.getElementById('mic-btn');
    const thresholdSlider = document.getElementById('drone-threshold');
    const bandLowSlider = document.getElementById('drone-band-low');
    const bandHighSlider = document.getElementById('drone-band-high');
    const spectrumCanvas = document.getElementById('drone-spectrum-canvas');
    const timelineCanvas = document.getElementById('drone-timeline-canvas');
    const spCtx = spectrumCanvas.getContext('2d');
    const tlCtx = timelineCanvas.getContext('2d');

    let audioBuffer = null;
    let micStream = null;

    // Update labels
    thresholdSlider.addEventListener('input', () => {
        document.getElementById('threshold-val').textContent = thresholdSlider.value;
    });
    bandLowSlider.addEventListener('input', () => {
        document.getElementById('band-low-val').textContent = bandLowSlider.value;
    });
    bandHighSlider.addEventListener('input', () => {
        document.getElementById('band-high-val').textContent = bandHighSlider.value;
    });

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const arrayBuffer = await file.arrayBuffer();
        const actx = new (window.AudioContext || window.webkitAudioContext)();
        audioBuffer = await actx.decodeAudioData(arrayBuffer);
        actx.close();
        detectBtn.disabled = false;
    });

    detectBtn.addEventListener('click', () => {
        if (!audioBuffer) return;
        analyzeForDrone(audioBuffer);
    });

    // ===================== LIVE MICROPHONE MODE =====================
    micBtn.addEventListener('click', async () => {
        if (micStream) {
            // Stop
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
            micBtn.textContent = '🎤 Use Microphone (Live)';
            return;
        }

        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            micBtn.textContent = '⏹ Stop Microphone';

            const actx = new (window.AudioContext || window.webkitAudioContext)();
            const source = actx.createMediaStreamSource(micStream);
            const analyser = actx.createAnalyser();
            analyser.fftSize = 4096;
            source.connect(analyser);

            const freqData = new Float32Array(analyser.frequencyBinCount);
            const sampleRate = actx.sampleRate;

            function liveDetect() {
                if (!micStream) { actx.close(); return; }
                analyser.getFloatFrequencyData(freqData);

                const bandLow = parseFloat(bandLowSlider.value);
                const bandHigh = parseFloat(bandHighSlider.value);
                const threshold = parseFloat(thresholdSlider.value);

                // Convert to linear magnitudes
                const magnitudes = new Float64Array(freqData.length);
                for (let i = 0; i < freqData.length; i++) {
                    magnitudes[i] = Math.pow(10, freqData[i] / 20);
                }

                const result = detectDroneFeatures(magnitudes, sampleRate, analyser.fftSize, bandLow, bandHigh, threshold);

                drawSpectrum(magnitudes, sampleRate, analyser.fftSize, bandLow, bandHigh);
                updateResults(result);

                requestAnimationFrame(liveDetect);
            }
            liveDetect();
        } catch (err) {
            alert('Microphone access denied: ' + err.message);
        }
    });

    // ===================== FILE ANALYSIS =====================
    function analyzeForDrone(buffer) {
        const channelData = buffer.getChannelData(0);
        const sampleRate = buffer.sampleRate;
        const fftSize = 4096;
        const hopSize = fftSize / 2;
        const fft = new FFTProcessor(fftSize);
        const { times, freqs, data: frames } = fft.stft(channelData, sampleRate, hopSize);

        const bandLow = parseFloat(bandLowSlider.value);
        const bandHigh = parseFloat(bandHighSlider.value);
        const threshold = parseFloat(thresholdSlider.value);

        // Analyze each frame
        const detectionTimeline = [];
        let bestResult = null;
        let bestConfidence = 0;

        for (let i = 0; i < frames.length; i++) {
            const result = detectDroneFeatures(frames[i], sampleRate, fftSize, bandLow, bandHigh, threshold);
            detectionTimeline.push({ time: times[i], ...result });
            if (result.confidence > bestConfidence) {
                bestConfidence = result.confidence;
                bestResult = result;
            }
        }

        // Draw average spectrum
        const avgSpectrum = new Float64Array(frames[0].length);
        for (const frame of frames) {
            for (let i = 0; i < frame.length; i++) avgSpectrum[i] += frame[i];
        }
        for (let i = 0; i < avgSpectrum.length; i++) avgSpectrum[i] /= frames.length;

        drawSpectrum(avgSpectrum, sampleRate, fftSize, bandLow, bandHigh);
        drawTimeline(detectionTimeline);
        if (bestResult) updateResults(bestResult);
    }

    /**
     * Drone detection via spectral harmonic analysis.
     * Drones have rotors producing a fundamental frequency + harmonics.
     * We look for:
     * 1. Strong energy in the drone band
     * 2. Harmonic peaks (multiples of a fundamental)
     * 3. High ratio of in-band to out-of-band energy
     */
    function detectDroneFeatures(spectrum, sampleRate, fftSize, bandLow, bandHigh, threshold) {
        const iLow = Math.floor((bandLow * fftSize) / sampleRate);
        const iHigh = Math.ceil((bandHigh * fftSize) / sampleRate);

        // Energy in drone band vs total
        let bandEnergy = 0, totalEnergy = 0;
        for (let i = 0; i < spectrum.length; i++) {
            const e = spectrum[i] * spectrum[i];
            totalEnergy += e;
            if (i >= iLow && i <= iHigh) bandEnergy += e;
        }
        const bandRatio = bandEnergy / (totalEnergy + 1e-20);

        // Find dominant frequency in band
        let peakVal = 0, peakIdx = iLow;
        for (let i = iLow; i <= iHigh && i < spectrum.length; i++) {
            if (spectrum[i] > peakVal) { peakVal = spectrum[i]; peakIdx = i; }
        }
        const dominantFreq = (peakIdx * sampleRate) / fftSize;

        // Check for harmonics (2nd, 3rd, 4th)
        let harmonicsFound = 0;
        const harmonicTolerance = 3; // bins
        for (let h = 2; h <= 5; h++) {
            const expectedBin = peakIdx * h;
            if (expectedBin >= spectrum.length) break;
            // Check if there's a peak near expected harmonic
            let localMax = 0;
            for (let j = expectedBin - harmonicTolerance; j <= expectedBin + harmonicTolerance && j < spectrum.length; j++) {
                if (j >= 0 && spectrum[j] > localMax) localMax = spectrum[j];
            }
            // Harmonic should be at least 20% of the fundamental
            if (localMax > peakVal * 0.15) harmonicsFound++;
        }

        // Compute signal-to-noise ratio
        const noiseFloor = (totalEnergy - bandEnergy) / (spectrum.length - (iHigh - iLow + 1) + 1e-20);
        const signalPower = bandEnergy / (iHigh - iLow + 1);
        const snr = 10 * Math.log10(signalPower / (noiseFloor + 1e-20));

        // Confidence score: combination of factors
        const bandScore = Math.min(1, bandRatio * 5);         // high band ratio -> drone
        const harmonicScore = harmonicsFound / 4;              // more harmonics -> drone
        const snrScore = Math.min(1, Math.max(0, snr / 30));  // high SNR -> drone

        const confidence = (bandScore * 0.35 + harmonicScore * 0.4 + snrScore * 0.25);
        const detected = confidence >= threshold;

        return {
            detected,
            confidence,
            dominantFreq,
            harmonicsFound,
            snr,
            bandRatio
        };
    }

    function drawSpectrum(spectrum, sampleRate, fftSize, bandLow, bandHigh) {
        const w = spectrumCanvas.width;
        const h = spectrumCanvas.height;
        spCtx.fillStyle = '#0a0a1a';
        spCtx.fillRect(0, 0, w, h);

        // Show 0 to 2000 Hz
        const maxFreq = 2000;
        const maxBin = Math.ceil((maxFreq * fftSize) / sampleRate);
        const binsToShow = Math.min(maxBin, spectrum.length);

        // Find max for normalization
        let maxVal = 0;
        for (let i = 0; i < binsToShow; i++) {
            if (spectrum[i] > maxVal) maxVal = spectrum[i];
        }

        const barW = w / binsToShow;

        // Draw drone band highlight
        const bandLowX = (bandLow / maxFreq) * w;
        const bandHighX = (bandHigh / maxFreq) * w;
        spCtx.fillStyle = 'rgba(0, 212, 255, 0.08)';
        spCtx.fillRect(bandLowX, 0, bandHighX - bandLowX, h);
        spCtx.strokeStyle = '#00d4ff44';
        spCtx.setLineDash([4, 4]);
        spCtx.beginPath(); spCtx.moveTo(bandLowX, 0); spCtx.lineTo(bandLowX, h); spCtx.stroke();
        spCtx.beginPath(); spCtx.moveTo(bandHighX, 0); spCtx.lineTo(bandHighX, h); spCtx.stroke();
        spCtx.setLineDash([]);

        // Draw spectrum bars
        for (let i = 0; i < binsToShow; i++) {
            const val = spectrum[i] / (maxVal || 1);
            const barH = val * h * 0.9;
            const freq = (i * sampleRate) / fftSize;
            const inBand = freq >= bandLow && freq <= bandHigh;
            spCtx.fillStyle = inBand ? '#00d4ff' : '#555';
            spCtx.fillRect(i * barW, h - barH, Math.max(1, barW - 0.5), barH);
        }

        // Labels
        spCtx.fillStyle = '#999';
        spCtx.font = '11px monospace';
        for (let f = 0; f <= maxFreq; f += 200) {
            const x = (f / maxFreq) * w;
            spCtx.fillText(f + '', x, h - 3);
        }
        spCtx.fillText('Frequency (Hz)', w / 2 - 40, 15);
        spCtx.fillStyle = '#00d4ff';
        spCtx.fillText('Drone Band', bandLowX + 5, 30);
    }

    function drawTimeline(timeline) {
        const w = timelineCanvas.width;
        const h = timelineCanvas.height;
        tlCtx.fillStyle = '#0a0a1a';
        tlCtx.fillRect(0, 0, w, h);

        if (timeline.length === 0) return;

        const tMax = timeline[timeline.length - 1].time;
        const barW = w / timeline.length;

        for (let i = 0; i < timeline.length; i++) {
            const d = timeline[i];
            const x = i * barW;
            const barH = d.confidence * h * 0.8;

            // Color: green if detected, red if not
            if (d.detected) {
                tlCtx.fillStyle = `rgba(0, 255, 0, ${0.3 + d.confidence * 0.7})`;
            } else {
                tlCtx.fillStyle = `rgba(255, 50, 50, ${0.2 + d.confidence * 0.3})`;
            }
            tlCtx.fillRect(x, h - barH - 20, Math.max(1, barW - 0.5), barH);
        }

        // Threshold line
        const threshold = parseFloat(thresholdSlider.value);
        const threshY = h - threshold * h * 0.8 - 20;
        tlCtx.strokeStyle = '#ff0';
        tlCtx.setLineDash([5, 3]);
        tlCtx.beginPath(); tlCtx.moveTo(0, threshY); tlCtx.lineTo(w, threshY); tlCtx.stroke();
        tlCtx.setLineDash([]);
        tlCtx.fillStyle = '#ff0';
        tlCtx.font = '11px monospace';
        tlCtx.fillText('Threshold', 5, threshY - 5);

        // Time axis
        tlCtx.fillStyle = '#999';
        tlCtx.font = '11px monospace';
        tlCtx.fillText('Detection Confidence Over Time', w / 2 - 110, 15);
        tlCtx.fillText('0s', 5, h - 5);
        tlCtx.fillText(tMax.toFixed(1) + 's', w - 40, h - 5);
    }

    function updateResults(result) {
        const resultsDiv = document.getElementById('drone-results');
        resultsDiv.style.display = 'block';

        const statusEl = document.getElementById('drone-status');
        statusEl.textContent = result.detected ? '🚨 DRONE DETECTED' : '✅ No Drone Detected';
        statusEl.style.color = result.detected ? '#f00' : '#0f0';

        document.getElementById('drone-confidence').textContent = (result.confidence * 100).toFixed(1);
        document.getElementById('drone-freq').textContent = result.dominantFreq.toFixed(1);
        document.getElementById('drone-harmonics').textContent = result.harmonicsFound;
        document.getElementById('drone-snr').textContent = result.snr.toFixed(1);
    }
})();