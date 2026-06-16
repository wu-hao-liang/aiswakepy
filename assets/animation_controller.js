(function (root) {
    'use strict';

    class VesselWaveAnimationController {
        constructor(options = {}) {
            this.realTimeScale = options.realTimeScale || 10;
            this.now = options.now || (() => performance.now());
            this.requestFrame = options.requestFrame || (fn => requestAnimationFrame(fn));
            this.cancelFrame = options.cancelFrame || (id => cancelAnimationFrame(id));
            this.onChange = options.onChange || (() => {});
            this.selection = null;
            this.playing = false;
            this.progress = 0;
            this.startedAt = null;
            this.frameId = null;
            this._tick = this._tick.bind(this);
        }

        _displayDurationMs() {
            const loopDurationS = this.selection?.loopDurationS || 1;
            return Math.max(1, loopDurationS / Math.max(1e-9, this.realTimeScale) * 1000);
        }

        select(selection) {
            this.pause();
            this.selection = selection || null;
            this.progress = 0;
            this.startedAt = null;
            this.onChange(this.getState());
        }

        clear() {
            this.select(null);
        }

        toggle() {
            if (!this.selection) return false;
            if (this.playing) this.pause();
            else this.play();
            return this.playing;
        }

        play() {
            if (!this.selection || this.playing) return;
            this.playing = true;
            this.startedAt = this.now() - this.progress * this._displayDurationMs();
            this.frameId = this.requestFrame(this._tick);
            this.onChange(this.getState());
        }

        pause() {
            if (!this.playing) return;
            this.playing = false;
            if (this.frameId != null) this.cancelFrame(this.frameId);
            this.frameId = null;
            this.onChange(this.getState());
        }

        setProgress(progress) {
            this.progress = Math.max(0, Math.min(1, Number(progress) || 0));
            this.onChange(this.getState());
        }

        _tick(timestamp) {
            if (!this.playing) return;
            const elapsed = Math.max(0, timestamp - this.startedAt);
            const durationMs = this._displayDurationMs();
            this.progress = (elapsed % durationMs) / durationMs;
            this.onChange(this.getState());
            this.frameId = this.requestFrame(this._tick);
        }

        getState() {
            const loopDurationS = this.selection?.loopDurationS || 1;
            const trackDurationS = this.selection?.trackDurationS || loopDurationS;
            const simElapsedS = this.progress * loopDurationS;
            return {
                selection: this.selection,
                playing: this.playing,
                progress: this.progress,
                simElapsedS,
                loopDurationS,
                trackDurationS,
                realTimeScale: this.realTimeScale,
                trackProgress: Math.min(1, simElapsedS / Math.max(1e-9, trackDurationS)),
            };
        }

        frontProgress(sourceOffsetS, speedMps, distanceM) {
            const state = this.getState();
            const elapsed = state.simElapsedS - Math.max(0, Number(sourceOffsetS) || 0);
            if (elapsed <= 0) return 0;
            const speed = Math.max(0, Number(speedMps) || 0);
            const distance = Math.max(1e-9, Number(distanceM) || 0);
            return Math.min(1, (elapsed * speed) / distance);
        }

        transverseRadius(sourceOffsetS, speedMps) {
            const state = this.getState();
            const elapsed = state.simElapsedS - Math.max(0, Number(sourceOffsetS) || 0);
            if (elapsed <= 0) return 0;
            return elapsed * Math.max(0, Number(speedMps) || 0);
        }
    }

    root.VesselWaveAnimationController = VesselWaveAnimationController;
})(typeof window !== 'undefined' ? window : globalThis);
