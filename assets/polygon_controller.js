(function () {
    'use strict';

    class PolygonDrawingController {
        constructor(options) {
            this.container = options.container;
            this.canvas = options.canvas;
            this.button = options.button;
            this.getViewport = options.getViewport;
            this.onComplete = options.onComplete;
            this.onStateChange = options.onStateChange || function () {};
            this.closeRadius = options.closeRadius || 14;
            this.dragThreshold = options.dragThreshold || 5;
            this.vertices = [];
            this.hoverPoint = null;
            this.pointerStart = null;
            this.armed = false;
            this.drawing = false;
            this.ctrlHeld = false;

            this._onPointerDown = this._onPointerDown.bind(this);
            this._onPointerMove = this._onPointerMove.bind(this);
            this._onPointerUp = this._onPointerUp.bind(this);
            this._onContextMenu = this._onContextMenu.bind(this);
            this._onKeyDown = this._onKeyDown.bind(this);
        }

        arm() {
            if (this.armed || this.drawing) {
                this.cancel();
                return false;
            }
            this.armed = true;
            this._setButton('Hold Ctrl to add polygon vertices', '0.75');
            this.onStateChange('armed');
            if (this.ctrlHeld) this.activate();
            return true;
        }

        activate() {
            if (!this.armed || this.drawing) return false;
            this.drawing = true;
            this.canvas.style.display = 'block';
            this._resizeCanvas();
            this._setButton('Click vertices; click start to finish', '0.75');
            this.container.addEventListener('pointerdown', this._onPointerDown);
            this.container.addEventListener('pointermove', this._onPointerMove);
            this.container.addEventListener('pointerup', this._onPointerUp);
            this.container.addEventListener('contextmenu', this._onContextMenu);
            window.addEventListener('keydown', this._onKeyDown);
            this.onStateChange('drawing');
            this.redraw();
            return true;
        }

        setCtrlHeld(held) {
            this.ctrlHeld = Boolean(held);
            if (this.ctrlHeld && this.armed && !this.drawing) this.activate();
            if (!this.ctrlHeld) {
                this.hoverPoint = null;
                this.redraw();
            }
        }

        undo() {
            if (!this.drawing || this.vertices.length === 0) return false;
            this.vertices.pop();
            this.hoverPoint = null;
            this.redraw();
            return true;
        }

        cancel() {
            if (!this.armed && !this.drawing) return false;
            this._cleanup();
            this.onStateChange('cancelled');
            return true;
        }

        redraw() {
            if (!this.drawing) return;
            this._resizeCanvas();
            const ctx = this.canvas.getContext('2d');
            ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            if (this.vertices.length === 0) return;

            const viewport = this.getViewport();
            const points = this.vertices.map((point) => viewport.project(point));
            const hover = this.hoverPoint ? viewport.project(this.hoverPoint) : null;
            ctx.strokeStyle = 'rgba(80,200,255,0.95)';
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i][0], points[i][1]);
            }
            if (hover) ctx.lineTo(hover[0], hover[1]);
            ctx.stroke();

            ctx.setLineDash([]);
            for (let i = 0; i < points.length; i++) {
                ctx.beginPath();
                ctx.arc(points[i][0], points[i][1], i === 0 ? 6 : 4, 0, Math.PI * 2);
                ctx.fillStyle = i === 0
                    ? 'rgba(255,220,80,0.95)'
                    : 'rgba(80,200,255,0.95)';
                ctx.fill();
            }
        }

        getState() {
            return {
                armed: this.armed,
                drawing: this.drawing,
                ctrlHeld: this.ctrlHeld,
                vertices: this.vertices.map((point) => point.slice()),
            };
        }

        destroy() {
            this._cleanup();
        }

        _screenPoint(event) {
            const rect = this.container.getBoundingClientRect();
            return [event.clientX - rect.left, event.clientY - rect.top];
        }

        _onPointerDown(event) {
            if (event.button !== 0) return;
            this.pointerStart = this._screenPoint(event);
        }

        _onPointerMove(event) {
            if (!this.ctrlHeld && !event.ctrlKey) {
                this.hoverPoint = null;
                this.redraw();
                return;
            }
            this.hoverPoint = this.getViewport().unproject(this._screenPoint(event));
            this.redraw();
        }

        _onPointerUp(event) {
            if (event.button !== 0 || !this.pointerStart) return;
            const point = this._screenPoint(event);
            const moved = Math.hypot(
                point[0] - this.pointerStart[0],
                point[1] - this.pointerStart[1],
            );
            this.pointerStart = null;
            if (moved > this.dragThreshold) {
                this.redraw();
                return;
            }
            if (!this.ctrlHeld && !event.ctrlKey) return;

            const viewport = this.getViewport();
            if (this.vertices.length >= 3) {
                const first = viewport.project(this.vertices[0]);
                if (Math.hypot(point[0] - first[0], point[1] - first[1])
                        <= this.closeRadius) {
                    const polygon = this.vertices.map((vertex) => vertex.slice());
                    this._cleanup();
                    this.onComplete(polygon);
                    this.onStateChange('completed');
                    return;
                }
            }
            this.vertices.push(viewport.unproject(point));
            this.hoverPoint = null;
            this.redraw();
        }

        _onContextMenu(event) {
            event.preventDefault();
            this.cancel();
        }

        _onKeyDown(event) {
            const target = event.target;
            const tag = target && target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || (target && target.isContentEditable)) {
                return;
            }
            if (event.key === 'Escape') {
                event.preventDefault();
                this.cancel();
            } else if (event.key === 'Backspace') {
                event.preventDefault();
                this.undo();
            }
        }

        _resizeCanvas() {
            const rect = this.container.getBoundingClientRect();
            const width = Math.max(1, Math.round(rect.width));
            const height = Math.max(1, Math.round(rect.height));
            if (this.canvas.width !== width) this.canvas.width = width;
            if (this.canvas.height !== height) this.canvas.height = height;
        }

        _setButton(text, opacity) {
            if (!this.button) return;
            this.button.textContent = text;
            this.button.style.opacity = opacity;
        }

        _removeListeners() {
            this.container.removeEventListener('pointerdown', this._onPointerDown);
            this.container.removeEventListener('pointermove', this._onPointerMove);
            this.container.removeEventListener('pointerup', this._onPointerUp);
            this.container.removeEventListener('contextmenu', this._onContextMenu);
            window.removeEventListener('keydown', this._onKeyDown);
        }

        _cleanup() {
            this._removeListeners();
            this.armed = false;
            this.drawing = false;
            this.vertices = [];
            this.hoverPoint = null;
            this.pointerStart = null;
            this.canvas.style.display = 'none';
            const ctx = this.canvas.getContext('2d');
            ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this._setButton('Draw polygon on the map', '');
        }
    }

    window.AiswakePolygonController = PolygonDrawingController;
}());
