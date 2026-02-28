(function () {
    const SPEED_OF_SOUND = 343; // m/s

    // DOM elements
    const velocitySlider = document.getElementById('velocity');
    const frequencySlider = document.getElementById('frequency');
    const durationSlider = document.getElementById('sim-duration');
    const distanceSlider = document.getElementById('min-distance');
    const playBtn = document.getElementById('play-doppler');
    const stopBtn = document.getElementById('stop-doppler');
    const canvas = document.getElementById('doppler-canvas');
    const ctx = canvas.getContext('2d');

    let audioCtx = null;
    let isPlaying = false;
    let animationId = null;

    // Update displayed values
    function updateLabels() {
        const v = parseFloat(velocitySlider.value);
        document.getElementById('velocity-val').textContent = v;
        document.getElementById('velocity-kmh').textContent = (v * 3.6).toFixed(0);
        document.getElementById('frequency-val').textContent = frequencySlider.value;
        document.getElementById('duration-val').textContent = durationSlider.value;
        document.getElementById('distance-val').textContent = distanceSlider.value;
    }

    [velocitySlider, frequencySlider, durationSlider, distanceSlider].forEach(s =>
        s.addEventListener('input', updateLabels)
    );

    playBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);

    function stopSimulation() {
        isPlaying = false;
        if (animationId) cancelAnimationFrame(animationId);
        if (audioCtx) { audioCtx.close(); audioCtx = null; }
        playBtn.disabled = false;
        stopBtn.disabled = true;
    }

    function startSimulation() {
        if (isPlaying) return;
        isPlaying = true;
        playBtn.disabled = true;
        stopBtn.disabled = false;

        const v = parseFloat(velocitySlider.value);       // vehicle speed m/s
        const f0 = parseFloat(frequencySlider.value);      // horn frequency Hz
        const halfDuration = parseFloat(durationSlider.value); // seconds before/after closest point
        const dMin = parseFloat(distanceSlider.value);     // closest approach distance

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        // Create oscillator (horn tone)
        const osc = audioCtx.createOscillator();
        osc.type = 'sawtooth'; // richer harmonic content for realism
        osc.frequency.value = f0;

        // Second oscillator for engine rumble
        const engineOsc = audioCtx.createOscillator();
        engineOsc.type = 'triangle';
        engineOsc.frequency.value = f0 * 0.25;

        // Gains
        const hornGain = audioCtx.createGain();
        hornGain.gain.value = 0;
        const engineGain = audioCtx.createGain();
        engineGain.gain.value = 0;

        // Stereo panner
        const panner = audioCtx.createStereoPanner();

        // Connect: osc -> hornGain -> panner -> dest
        osc.connect(hornGain);
        hornGain.connect(panner);
        engineOsc.connect(engineGain);
        engineGain.connect(panner);
        panner.connect(audioCtx.destination);

        osc.start();
        engineOsc.start();

        const totalDuration = halfDuration * 2;
        const startTime = audioCtx.currentTime;
        const freqHistory = [];

        // Clear canvas
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        function animate() {
            const elapsed = audioCtx.currentTime - startTime;
            const t = elapsed - halfDuration; // t: -halfDuration to +halfDuration

            if (elapsed >= totalDuration || !isPlaying) {
                stopSimulation();
                return;
            }

            // Position of car along road (x-axis), listener at origin
            const xCar = v * t;
            const distance = Math.sqrt(dMin * dMin + xCar * xCar);

            // Radial velocity component toward listener
            const vRadial = (v * xCar) / distance; // positive = moving away

            // Doppler shifted frequency
            const fObserved = f0 * (SPEED_OF_SOUND / (SPEED_OF_SOUND + vRadial));

            // Set frequencies
            osc.frequency.setValueAtTime(fObserved, audioCtx.currentTime);
            engineOsc.frequency.setValueAtTime(fObserved * 0.25, audioCtx.currentTime);

            // Volume: inverse-square-ish fall-off
            const vol = Math.min(1.0, (dMin * dMin) / (distance * distance));
            hornGain.gain.setValueAtTime(vol * 0.5, audioCtx.currentTime);
            engineGain.gain.setValueAtTime(vol * 0.3, audioCtx.currentTime);

            // Pan: -1 (left) to +1 (right)
            const pan = Math.max(-1, Math.min(1, xCar / (v * halfDuration)));
            panner.pan.setValueAtTime(pan, audioCtx.currentTime);

            // Record for visualization
            freqHistory.push({ t: elapsed, f: fObserved, vol });

            // Draw
            drawFrequencyPlot(freqHistory, totalDuration, f0, v);

            animationId = requestAnimationFrame(animate);
        }

        animate();
    }

    function drawFrequencyPlot(history, totalDuration, f0, v) {
        const w = canvas.width;
        const h = canvas.height;
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);

        // Compute axis range
        const fMax = f0 * (SPEED_OF_SOUND / (SPEED_OF_SOUND - v)) * 1.05;
        const fMin = f0 * (SPEED_OF_SOUND / (SPEED_OF_SOUND + v)) * 0.95;

        // Grid
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = (i / 5) * h;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            const freqLabel = (fMax - (i / 5) * (fMax - fMin)).toFixed(0);
            ctx.fillStyle = '#666';
            ctx.font = '11px monospace';
            ctx.fillText(freqLabel + ' Hz', 5, y - 3);
        }

        // Source frequency line
        const ySource = h - ((f0 - fMin) / (fMax - fMin)) * h;
        ctx.strokeStyle = '#444';
        ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.moveTo(0, ySource); ctx.lineTo(w, ySource); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#666';
        ctx.fillText('f₀ = ' + f0 + ' Hz', w - 120, ySource - 5);

        // Frequency curve
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < history.length; i++) {
            const x = (history[i].t / totalDuration) * w;
            const y = h - ((history[i].f - fMin) / (fMax - fMin)) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Volume bar at bottom
        if (history.length > 0) {
            const last = history[history.length - 1];
            const barW = 100;
            const barH = 10;
            ctx.fillStyle = '#333';
            ctx.fillRect(w - barW - 10, h - 25, barW, barH);
            ctx.fillStyle = '#0f0';
            ctx.fillRect(w - barW - 10, h - 25, barW * last.vol, barH);
            ctx.fillStyle = '#999';
            ctx.fillText('Volume', w - barW - 10, h - 28);

            ctx.fillStyle = '#00d4ff';
            ctx.font = '13px monospace';
            ctx.fillText('f = ' + last.f.toFixed(1) + ' Hz', w / 2 - 60, 20);
        }

        // Labels
        ctx.fillStyle = '#666';
        ctx.font = '11px monospace';
        ctx.fillText('Time →', w / 2 - 20, h - 5);
    }
})();