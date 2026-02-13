import { useState, useRef, useCallback } from 'react'
import FluidCanvas from './FluidCanvas'
import Controls from './Controls'
import './App.css'

const GRID_SIZE = 256

function App() {
  const [playing, setPlaying] = useState(true)
  const [viscosity, setViscosity] = useState(0.0001)
  const [diffusion, setDiffusion] = useState(0.00001)
  const [fps, setFps] = useState(0)
  const simRef = useRef(null)

  const handleReset = useCallback(() => {
    if (simRef.current) {
      simRef.current.reset()
    }
  }, [])

  const handleClearDye = useCallback(() => {
    if (simRef.current) {
      simRef.current.clearDye()
    }
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>Turbulent Flow</h1>
        <span className="subtitle">2D Navier-Stokes Fluid Simulation &middot; Rust/WASM</span>
      </header>
      <div className="main-layout">
        <FluidCanvas
          gridSize={GRID_SIZE}
          playing={playing}
          viscosity={viscosity}
          diffusion={diffusion}
          onFpsUpdate={setFps}
          simRef={simRef}
        />
        <Controls
          playing={playing}
          onTogglePlay={() => setPlaying(p => !p)}
          viscosity={viscosity}
          onViscosityChange={setViscosity}
          diffusion={diffusion}
          onDiffusionChange={setDiffusion}
          onReset={handleReset}
          onClearDye={handleClearDye}
          fps={fps}
        />
      </div>
      <footer className="footer">
        Click and drag to inject dye and forces &middot; Colors follow mouse direction
      </footer>
    </div>
  )
}

export default App
