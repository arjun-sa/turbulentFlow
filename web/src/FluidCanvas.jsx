import { useEffect, useRef, useCallback } from 'react'
import init, { FluidSim, wasm_memory, initThreadPool } from 'fluid-sim-wasm'
import './FluidCanvas.css'

// Hue-based color palette for dye injection
function hueToRGB(hue) {
  const h = ((hue % 360) + 360) % 360
  const s = 1, l = 0.55
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs((h / 60) % 2 - 1))
  const m = l - c / 2
  let r, g, b
  if (h < 60) { r = c; g = x; b = 0 }
  else if (h < 120) { r = x; g = c; b = 0 }
  else if (h < 180) { r = 0; g = c; b = x }
  else if (h < 240) { r = 0; g = x; b = c }
  else if (h < 300) { r = x; g = 0; b = c }
  else { r = c; g = 0; b = x }
  return [(r + m), (g + m), (b + m)]
}

export default function FluidCanvas({
  gridSize,
  playing,
  viscosity,
  diffusion,
  onFpsUpdate,
  simRef,
}) {
  const canvasRef = useRef(null)
  const wasmReady = useRef(false)
  const simInstance = useRef(null)
  const animFrameRef = useRef(null)
  const mouseRef = useRef({ down: false, x: 0, y: 0, prevX: 0, prevY: 0 })
  const hueRef = useRef(0)
  const fpsFrames = useRef([])
  const playingRef = useRef(playing)
  const viscRef = useRef(viscosity)
  const diffRef = useRef(diffusion)

  // Keep refs in sync with props
  useEffect(() => { playingRef.current = playing }, [playing])
  useEffect(() => {
    viscRef.current = viscosity
    if (simInstance.current) simInstance.current.set_viscosity(viscosity)
  }, [viscosity])
  useEffect(() => {
    diffRef.current = diffusion
    if (simInstance.current) simInstance.current.set_diffusion(diffusion)
  }, [diffusion])

  // Initialize WASM, create sim, start render loop
  useEffect(() => {
    let cancelled = false
    let running = true

    async function setup() {
      await init()
      await initThreadPool(navigator.hardwareConcurrency)
      if (cancelled) return

      const sim = new FluidSim(gridSize)
      simInstance.current = sim
      simRef.current = sim
      wasmReady.current = true

      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const memory = wasm_memory()

      function loop(timestamp) {
        if (!running) return

        // FPS tracking
        fpsFrames.current.push(timestamp)
        while (fpsFrames.current.length > 0 && fpsFrames.current[0] < timestamp - 1000) {
          fpsFrames.current.shift()
        }
        onFpsUpdate(fpsFrames.current.length)

        if (playingRef.current) {
          // Process mouse interaction
          const m = mouseRef.current
          if (m.down) {
            const rect = canvas.getBoundingClientRect()
            const scaleX = gridSize / rect.width
            const scaleY = gridSize / rect.height
            const gx = (m.x - rect.left) * scaleX
            const gy = (m.y - rect.top) * scaleY
            const dx = (m.x - m.prevX) * scaleX
            const dy = (m.y - m.prevY) * scaleY

            // Add velocity force
            const forceMult = 5.0
            sim.add_velocity(gx, gy, dx * forceMult, dy * forceMult)

            // Add colorful dye
            hueRef.current = (hueRef.current + 1) % 360
            const [r, g, b] = hueToRGB(hueRef.current)
            sim.add_dye(gx, gy, r, g, b, 4)

            m.prevX = m.x
            m.prevY = m.y
          }

          sim.step()
        }

        // Render: get pointer into WASM memory and create ImageData
        sim.render_velocity_colored()
        const ptr = sim.pixels_ptr()
        const len = sim.pixels_len()

        // Create a copy from WASM linear memory (buffer can detach on grow)
        const wasmBytes = new Uint8ClampedArray(memory.buffer, ptr, len)
        const pixelsCopy = new Uint8ClampedArray(len)
        pixelsCopy.set(wasmBytes)
        const imageData = new ImageData(pixelsCopy, gridSize, gridSize)
        ctx.putImageData(imageData, 0, 0)

        animFrameRef.current = requestAnimationFrame(loop)
      }

      animFrameRef.current = requestAnimationFrame(loop)
    }

    setup()

    return () => {
      cancelled = true
      running = false
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
      if (simInstance.current) {
        simInstance.current.free()
        simInstance.current = null
        simRef.current = null
      }
    }
  }, [gridSize, simRef, onFpsUpdate])

  // Mouse/touch handlers
  const getPos = useCallback((e) => {
    if (e.touches) {
      return { x: e.touches[0].clientX, y: e.touches[0].clientY }
    }
    return { x: e.clientX, y: e.clientY }
  }, [])

  const handlePointerDown = useCallback((e) => {
    const pos = getPos(e)
    mouseRef.current = { down: true, x: pos.x, y: pos.y, prevX: pos.x, prevY: pos.y }
  }, [getPos])

  const handlePointerMove = useCallback((e) => {
    if (!mouseRef.current.down) return
    const pos = getPos(e)
    mouseRef.current.prevX = mouseRef.current.x
    mouseRef.current.prevY = mouseRef.current.y
    mouseRef.current.x = pos.x
    mouseRef.current.y = pos.y
  }, [getPos])

  const handlePointerUp = useCallback(() => {
    mouseRef.current.down = false
  }, [])

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        width={gridSize}
        height={gridSize}
        className="fluid-canvas"
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={handlePointerUp}
        onTouchStart={handlePointerDown}
        onTouchMove={handlePointerMove}
        onTouchEnd={handlePointerUp}
      />
    </div>
  )
}
