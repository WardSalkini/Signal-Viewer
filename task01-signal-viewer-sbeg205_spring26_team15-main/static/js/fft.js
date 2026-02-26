/**
 * Minimal FFT implementation for audio signal processing.
 * Cooley-Tukey radix-2 decimation-in-time.
 */
class FFTProcessor {
    constructor(size) {
        this.size = size;
        this.halfSize = size / 2;

        // Precompute twiddle factors
        this.cosTable = new Float64Array(this.halfSize);
        this.sinTable = new Float64Array(this.halfSize);
        for (let i = 0; i < this.halfSize; i++) {
            this.cosTable[i] = Math.cos((2 * Math.PI * i) / size);
            this.sinTable[i] = Math.sin((2 * Math.PI * i) / size);
        }

        // Bit-reversal permutation
        this.reverseTable = new Uint32Array(size);
        let limit = 1;
        let bit = size >> 1;
        while (limit < size) {
            for (let i = 0; i < limit; i++) {
                this.reverseTable[i + limit] = this.reverseTable[i] + bit;
            }
            limit <<= 1;
            bit >>= 1;
        }
    }

    /**
     * Perform in-place FFT.
     * @param {Float64Array} real - Real part (length = size)
     * @param {Float64Array} imag - Imaginary part (length = size)
     */
    forward(real, imag) {
        const n = this.size;

        // Bit-reversal
        for (let i = 0; i < n; i++) {
            const j = this.reverseTable[i];
            if (i < j) {
                let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
                tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
            }
        }

        // Cooley-Tukey
        for (let len = 2; len <= n; len <<= 1) {
            const halfLen = len >> 1;
            const step = n / len;
            for (let i = 0; i < n; i += len) {
                for (let j = 0; j < halfLen; j++) {
                    const idx = j * step;
                    const tR = real[i + j + halfLen] * this.cosTable[idx] +
                        imag[i + j + halfLen] * this.sinTable[idx];
                    const tI = imag[i + j + halfLen] * this.cosTable[idx] -
                        real[i + j + halfLen] * this.sinTable[idx];
                    real[i + j + halfLen] = real[i + j] - tR;
                    imag[i + j + halfLen] = imag[i + j] - tI;
                    real[i + j] += tR;
                    imag[i + j] += tI;
                }
            }
        }
    }

    /**
     * Compute magnitude spectrum from time-domain signal.
     * @param {Float32Array} signal - Input signal (length = size)
     * @returns {Float64Array} Magnitude spectrum (length = size/2)
     */
    magnitudeSpectrum(signal) {
        const real = new Float64Array(this.size);
        const imag = new Float64Array(this.size);

        // Apply Hann window and copy
        for (let i = 0; i < this.size; i++) {
            const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this.size - 1)));
            real[i] = (signal[i] || 0) * window;
            imag[i] = 0;
        }

        this.forward(real, imag);

        const magnitudes = new Float64Array(this.halfSize);
        for (let i = 0; i < this.halfSize; i++) {
            magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }
        return magnitudes;
    }

    /**
     * Compute Short-Time Fourier Transform (spectrogram).
     * @param {Float32Array} signal - Input signal
     * @param {number} sampleRate - Sample rate in Hz
     * @param {number} hopSize - Hop size in samples
     * @returns {{ times: number[], freqs: number[], data: Float64Array[] }}
     */
    stft(signal, sampleRate, hopSize) {
        hopSize = hopSize || this.size / 2;
        const frames = [];
        const times = [];
        const freqs = [];

        // Build frequency axis
        for (let i = 0; i < this.halfSize; i++) {
            freqs.push((i * sampleRate) / this.size);
        }

        for (let start = 0; start + this.size <= signal.length; start += hopSize) {
            const segment = signal.subarray(start, start + this.size);
            frames.push(this.magnitudeSpectrum(segment));
            times.push((start + this.size / 2) / sampleRate);
        }

        return { times, freqs, data: frames };
    }
}

// Export for use in other modules
window.FFTProcessor = FFTProcessor;