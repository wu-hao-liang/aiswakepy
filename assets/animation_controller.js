(function (root) {
    'use strict';

    class VesselWaveAnimationController {
        constructor(options = {}) {
            this.durationMs = options.durationMs || 12000;
            this.trackFraction = options.trackFraction || 0.7;
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
            this.startedAt = this.now() - this.progress * this.durationMs;
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
            this.progress = (elapsed % this.durationMs) / this.durationMs;
            this.onChange(this.getState());
            this.frameId = this.requestFrame(this._tick);
        }

        getState() {
            return {
                selection: this.selection,
                playing: this.playing,
                progress: this.progress,
                trackProgress: Math.min(1, this.progress / this.trackFraction),
                trackFraction: this.trackFraction,
            };
        }

        rayProgress(sourceFraction) {
            const start = Math.max(
                0, Math.min(1, Number(sourceFraction) || 0)
            ) * this.trackFraction;
            if (this.progress <= start) return 0;
            return Math.min(1, (this.progress - start) / Math.max(1e-9, 1 - start));
        }
    }

    root.VesselWaveAnimationController = VesselWaveAnimationController;
})(typeof window !== 'undefined' ? window : globalThis);
