import './Controls.css'

function logScale(value, min, max) {
  const minLog = Math.log10(min)
  const maxLog = Math.log10(max)
  return Math.pow(10, minLog + (maxLog - minLog) * value)
}

function logScaleInverse(value, min, max) {
  const minLog = Math.log10(min)
  const maxLog = Math.log10(max)
  return (Math.log10(value) - minLog) / (maxLog - minLog)
}

export default function Controls({
  playing,
  onTogglePlay,
  viscosity,
  onViscosityChange,
  diffusion,
  onDiffusionChange,
  onReset,
  onClearDye,
  fps,
}) {
  return (
    <div className="controls">
      <div className="fps-display">
        <span className="fps-value">{fps}</span>
        <span className="fps-label">FPS</span>
      </div>

      <div className="control-group">
        <div className="button-row">
          <button
            className={`btn ${playing ? 'btn-pause' : 'btn-play'}`}
            onClick={onTogglePlay}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button className="btn btn-secondary" onClick={onReset}>
            Reset
          </button>
          <button className="btn btn-secondary" onClick={onClearDye}>
            Clear Dye
          </button>
        </div>
      </div>

      <div className="control-group">
        <label className="slider-label">
          <span>Viscosity</span>
          <span className="slider-value">{viscosity.toExponential(1)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={logScaleInverse(viscosity, 0.000001, 0.01)}
          onChange={(e) =>
            onViscosityChange(logScale(parseFloat(e.target.value), 0.000001, 0.01))
          }
          className="slider"
        />
      </div>

      <div className="control-group">
        <label className="slider-label">
          <span>Diffusion</span>
          <span className="slider-value">{diffusion.toExponential(1)}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={logScaleInverse(diffusion, 0.0000001, 0.001)}
          onChange={(e) =>
            onDiffusionChange(logScale(parseFloat(e.target.value), 0.0000001, 0.001))
          }
          className="slider"
        />
      </div>

      <div className="controls-hint">
        <p>Drag on the canvas to add colorful dye and swirl the fluid.</p>
        <p>Adjust viscosity for thicker/thinner fluid.</p>
        <p>Adjust diffusion for how fast dye spreads.</p>
      </div>
    </div>
  )
}
