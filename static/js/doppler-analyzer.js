(function () {
    const SPEED_OF_SOUND = 343;
    const uploadInput = document.getElementById('audio-upload');
    const analyzeBtn = document.getElementById('analyze-btn');
    const freqMinSlider = document.getElementById('freq-min');
    const freqMaxSlider = document.getElementById('freq-max');
    const spectrogramCanvas = document.getElementById('spectrogram-canvas');
    const dopplerCurveCanvas = document.getElementById('doppler-curve-canvas');
    const sCtx = spectrogramCanvas.getContext('2d');
    const dCtx = dopplerCurveCanvas.getContext('2d');

    let audioBuffer = null;

    // Update labels
    freqMinSlider.addEventListener('input', () => {
        document.getElementById('freq-min-val').textContent = freqMinSlider.value;
    });
    freqMaxSlider.addEventListener('input', () => {
        document.getElementById('freq-max-val').textContent = freqMaxSlider.value;
    });

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const arrayBuffer = await file.arrayBuffer();
        const actx = new (window.AudioContext || window.webkitAudioContext)();
        audioBuffer = await actx.decodeAudioData(arrayBuffer);
        actx.close();
        analyzeBtn.disabled = false;
    });

    analyzeBtn.addEventListener('click', () => {
        if (!audioBuffer) return;
        analyzeAudio();
    });

    function analyzeAudio() {
        const channelData = audioBuffer.getChannelData(0);
        const sampleRate = audioBuffer.sampleRate;
        const fftSize = 4096;
        const hopSize = fftSize / 4;
        const fft = new FFTProcessor(fftSize);

        const { times, freqs, data: frames } = fft.stft(channelData, sampleRate, hopSize);

        const fMinHz = parseFloat(freqMinSlider.value);
        const fMaxHz = parseFloat(freqMaxSlider.value);

        // Find frequency bin indices for the band of interest
        const iMin = Math.floor((fMinHz * fftSize) / sampleRate);
        const iMax = Math.ceil((fMaxHz * fftSize) / sampleRate);

        // Extract dominant frequency in band for each frame
        const peakFreqs = [];
        const peakMags = [];
        for (const frame of frames) {
            let maxVal = -Infinity;
            let maxIdx = iMin;
            for (let i = iMin; i <= iMax && i < frame.length; i++) {
                if (frame[i] > maxVal) {
                    maxVal = frame[i];
                    maxIdx = i;
                }
            }
            peakFreqs.push((maxIdx * sampleRate) / fftSize);
            peakMags.push(maxVal);
        }

        // Draw spectrogram
        drawSpectrogram(frames, freqs, times, sampleRate, fftSize, fMinHz, fMaxHz);

        // Draw doppler curve
        drawDopplerCurve(times, peakFreqs, fMinHz, fMaxHz);

        // Estimate speed: find the transition point (max frequency drop)
        estimateSpeed(times, peakFreqs, peakMags);
    }

    function drawSpectrogram(frames, freqs, times, sampleRate, fftSize, fMinHz, fMaxHz) {
        const w = spectrogramCanvas.width;
        const h = spectrogramCanvas.height;
        sCtx.fillStyle = '#0a0a1a';
        sCtx.fillRect(0, 0, w, h);

        if (frames.length === 0) return;

        const iMin = Math.floor((fMinHz * fftSize) / sampleRate);
        const iMax = Math.min(Math.ceil((fMaxHz * fftSize) / sampleRate), frames[0].length - 1);
        const freqBins = iMax - iMin + 1;

        // Find global max for normalization
        let globalMax = 0;
        for (const frame of frames) {
            for (let i = iMin; i <= iMax; i++) {
                if (frame[i] > globalMax) globalMax = frame[i];
            }
        }

        const colW = w / frames.length;
        const rowH = h / freqBins;

        for (let t = 0; t < frames.length; t++) {
            for (let f = iMin; f <= iMax; f++) {
                const val = frames[t][f] / (globalMax || 1);
                const dB = 20 * Math.log10(val + 1e-10);
                const normalized = Math.max(0, Math.min(1, (dB + 60) / 60));

                const r = Math.floor(normalized * 255);
                const g = Math.floor(normalized * normalized * 200);
                const b = Math.floor(normalized * 80);
                sCtx.fillStyle = `rgb(${r},${g},${b})`;

                const x = t * colW;
                const y = h - (f - iMin) * rowH - rowH;
                sCtx.fillRect(x, y, Math.ceil(colW), Math.ceil(rowH));
            }
        }

        // Axis labels
        sCtx.fillStyle = '#999';
        sCtx.font = '11px monospace';
        sCtx.fillText(fMinHz + ' Hz', 5, h - 5);
        sCtx.fillText(fMaxHz + ' Hz', 5, 15);
        sCtx.fillText('Time →', w / 2 - 20, h - 5);
    }

    function drawDopplerCurve(times, peakFreqs, fMinHz, fMaxHz) {
        const w = dopplerCurveCanvas.width;
        const h = dopplerCurveCanvas.height;
        dCtx.fillStyle = '#0a0a1a';
        dCtx.fillRect(0, 0, w, h);

        if (times.length === 0) return;

        // Smooth the peak frequencies (moving average)
        const windowSize = 5;
        const smoothed = [];
        for (let i = 0; i < peakFreqs.length; i++) {
            let sum = 0, count = 0;
            for (let j = Math.max(0, i - windowSize); j <= Math.min(peakFreqs.length - 1, i + windowSize); j++) {
                sum += peakFreqs[j];
                count++;
            }
            smoothed.push(sum / count);
        }

        const fMin = Math.min(...smoothed) * 0.95;
        const fMax = Math.max(...smoothed) * 1.05;
        const tMax = times[times.length - 1];

        // Grid
        dCtx.strokeStyle = '#222';
        dCtx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * h;
            dCtx.beginPath(); dCtx.moveTo(0, y); dCtx.lineTo(w, y); dCtx.stroke();
            const label = (fMax - (i / 4) * (fMax - fMin)).toFixed(0);
            dCtx.fillStyle = '#666';
            dCtx.font = '11px monospace';
            dCtx.fillText(label + ' Hz', 5, y + 12);
        }

        // Draw the curve
        dCtx.strokeStyle = '#ff6600';
        dCtx.lineWidth = 2;
        dCtx.beginPath();
        for (let i = 0; i < smoothed.length; i++) {
            const x = (times[i] / tMax) * w;
            const y = h - ((smoothed[i] - fMin) / (fMax - fMin)) * h;
            if (i === 0) dCtx.moveTo(x, y);
            else dCtx.lineTo(x, y);
        }
        dCtx.stroke();

        // Labels
        dCtx.fillStyle = '#ff6600';
        dCtx.font = '12px monospace';
        dCtx.fillText('Peak Frequency over Time', w / 2 - 100, 15);
    }

    function estimateSpeed(times, peakFreqs, peakMags) {
        // Smooth frequencies
        const windowSize = 7;
        const smoothed = [];
        for (let i = 0; i < peakFreqs.length; i++) {
            let sum = 0, count = 0;
            for (let j = Math.max(0, i - windowSize); j <= Math.min(peakFreqs.length - 1, i + windowSize); j++) {
                sum += peakFreqs[j];
                count++;
            }
            smoothed.push(sum / count);
        }

        // Find the largest frequency drop (transition point)
        let maxDrop = 0;
        let transitionIdx = Math.floor(smoothed.length / 2);
        for (let i = 1; i < smoothed.length; i++) {
            const drop = smoothed[i - 1] - smoothed[i];
            if (drop > maxDrop) {
                maxDrop = drop;
                transitionIdx = i;
            }
        }

        // Average frequency in first quarter (approach) and last quarter (recede)
        const q1End = Math.floor(smoothed.length * 0.25);
        const q3Start = Math.floor(smoothed.length * 0.75);

        let fApproach = 0;
        for (let i = 0; i < q1End; i++) fApproach += smoothed[i];
        fApproach /= q1End || 1;

        let fRecede = 0;
        for (let i = q3Start; i < smoothed.length; i++) fRecede += smoothed[i];
        fRecede /= (smoothed.length - q3Start) || 1;

        // Doppler formula: v = c * (f_approach - f_recede) / (f_approach + f_recede)
        const vEstimate = SPEED_OF_SOUND * (fApproach - fRecede) / (fApproach + fRecede);

        // Source frequency: geometric mean
        const fSource = Math.sqrt(fApproach * fRecede);

        // Display results
        const resultsDiv = document.getElementById('analysis-results');
        resultsDiv.style.display = 'block';
        document.getElementById('f-approach').textContent = fApproach.toFixed(1);
        document.getElementById('f-recede').textContent = fRecede.toFixed(1);
        document.getElementById('f-source').textContent = fSource.toFixed(1);
        document.getElementById('v-estimate').textContent = Math.abs(vEstimate).toFixed(1);
        document.getElementById('v-estimate-kmh').textContent = (Math.abs(vEstimate) * 3.6).toFixed(1);
    }
})();