# Turbulent Flow

A real-time 2D fluid simulation running in the browser, powered by **Rust compiled to WebAssembly** with a **React** frontend.

## How It Works

The simulation implements Jos Stam's **Stable Fluids** algorithm, a Navier-Stokes solver that is unconditionally stable and fast enough for real-time use. The solver runs entirely in WASM for performance, while React handles the UI and Canvas2D renders the output.

**Key features:**
- 256x256 grid fluid simulation at 60 FPS
- RGB dye injection with mouse/touch interaction
- Adjustable viscosity and diffusion parameters
- Play/pause, reset, and clear controls
- FPS counter

## Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/) (v18+)

Install wasm-pack if you don't have it:

```bash
cargo install wasm-pack
rustup target add wasm32-unknown-unknown
```

## Setup & Build

### 1. Build the WASM module

```bash
cd fluid-sim-wasm
wasm-pack build --target web --release
```

### 2. Install frontend dependencies

```bash
cd web
npm install
```

### 3. Run the development server

```bash
cd web
npm run dev
```

Open the URL shown in the terminal (typically http://localhost:5173).

### Production build

```bash
cd web
npm run build
npm run preview
```

## Project Structure

```
turbulentFlow/
├── fluid-sim-wasm/          # Rust/WASM fluid simulation
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs           # Navier-Stokes solver + WASM bindings
│   └── pkg/                  # wasm-pack build output
├── web/                      # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx           # Main app layout
│   │   ├── FluidCanvas.jsx   # Canvas rendering + WASM integration
│   │   ├── Controls.jsx      # UI controls (sliders, buttons)
│   │   └── *.css
│   └── package.json
└── README.md
```

## Usage

- **Click and drag** on the canvas to inject colorful dye and apply forces
- **Viscosity** slider: controls how thick/thin the fluid is (higher = more viscous)
- **Diffusion** slider: controls how fast dye spreads (higher = faster spread)
- **Pause/Play**: stop or resume the simulation
- **Reset**: clear everything and restart
- **Clear Dye**: remove dye but keep velocity field

## Technical Details

- **Solver**: Jos Stam's Stable Fluids (advection-projection method)
- **Grid**: 256x256 with 1-cell boundary padding
- **Relaxation**: 20 Gauss-Seidel iterations for diffusion and pressure
- **Rendering**: RGBA pixel buffer written in Rust, transferred via shared WASM memory
- **Interop**: wasm-bindgen for Rust↔JS, vite-plugin-wasm for bundling
