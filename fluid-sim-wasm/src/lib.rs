use wasm_bindgen::prelude::*;

/// Export WASM memory so JS can create views into the pixel buffer.
#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// 2D Fluid Simulation using Jos Stam's "Stable Fluids" (Navier-Stokes based).
///
/// Fields are stored in row-major order with a 1-cell boundary padding,
/// so the total array size is (N+2)*(N+2) for an NxN simulation grid.
#[wasm_bindgen]
pub struct FluidSim {
    n: usize,        // grid resolution (interior cells per side)
    dt: f32,         // timestep
    visc: f32,       // viscosity
    diff: f32,       // diffusion rate for dye

    // velocity components
    vx: Vec<f32>,
    vy: Vec<f32>,
    vx0: Vec<f32>,
    vy0: Vec<f32>,

    // dye density (3 channels for RGB)
    density_r: Vec<f32>,
    density_g: Vec<f32>,
    density_b: Vec<f32>,
    density_r0: Vec<f32>,
    density_g0: Vec<f32>,
    density_b0: Vec<f32>,

    // output pixel buffer (RGBA)
    pixels: Vec<u8>,
}

#[wasm_bindgen]
impl FluidSim {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> FluidSim {
        let size = (n + 2) * (n + 2);
        FluidSim {
            n,
            dt: 0.1,
            visc: 0.0001,
            diff: 0.00001,
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            vx0: vec![0.0; size],
            vy0: vec![0.0; size],
            density_r: vec![0.0; size],
            density_g: vec![0.0; size],
            density_b: vec![0.0; size],
            density_r0: vec![0.0; size],
            density_g0: vec![0.0; size],
            density_b0: vec![0.0; size],
            pixels: vec![0; n * n * 4],
        }
    }

    pub fn grid_size(&self) -> usize {
        self.n
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    pub fn set_viscosity(&mut self, v: f32) {
        self.visc = v;
    }

    pub fn set_diffusion(&mut self, d: f32) {
        self.diff = d;
    }

    /// Add velocity at a grid cell (x, y in 0..N range, mapped to interior).
    pub fn add_velocity(&mut self, x: f32, y: f32, fx: f32, fy: f32) {
        let n = self.n;
        let i = (x.clamp(0.0, (n - 1) as f32) as usize) + 1;
        let j = (y.clamp(0.0, (n - 1) as f32) as usize) + 1;
        let idx = ix(i, j, n);
        self.vx0[idx] += fx;
        self.vy0[idx] += fy;
    }

    /// Add dye at a grid cell with given RGB color and radius.
    pub fn add_dye(&mut self, x: f32, y: f32, r: f32, g: f32, b: f32, radius: f32) {
        let n = self.n;
        let rad = radius as i32;
        let cx = x as i32;
        let cy = y as i32;
        for di in -rad..=rad {
            for dj in -rad..=rad {
                let dist_sq = (di * di + dj * dj) as f32;
                if dist_sq > radius * radius {
                    continue;
                }
                let pi = cx + di;
                let pj = cy + dj;
                if pi >= 0 && pi < n as i32 && pj >= 0 && pj < n as i32 {
                    let idx = ix((pi as usize) + 1, (pj as usize) + 1, n);
                    let falloff = 1.0 - (dist_sq / (radius * radius));
                    self.density_r0[idx] += r * falloff * 1000.0;
                    self.density_g0[idx] += g * falloff * 1000.0;
                    self.density_b0[idx] += b * falloff * 1000.0;
                }
            }
        }
    }

    /// Advance simulation by one timestep.
    pub fn step(&mut self) {
        let n = self.n;
        let dt = self.dt;
        let visc = self.visc;
        let diff = self.diff;

        // Velocity step
        add_source(n, &mut self.vx, &self.vx0, dt);
        add_source(n, &mut self.vy, &self.vy0, dt);

        std::mem::swap(&mut self.vx0, &mut self.vx);
        diffuse(n, 1, &mut self.vx, &self.vx0, visc, dt);
        std::mem::swap(&mut self.vy0, &mut self.vy);
        diffuse(n, 2, &mut self.vy, &self.vy0, visc, dt);

        project(n, &mut self.vx, &mut self.vy, &mut self.vx0, &mut self.vy0);

        std::mem::swap(&mut self.vx0, &mut self.vx);
        std::mem::swap(&mut self.vy0, &mut self.vy);
        advect(n, 1, &mut self.vx, &self.vx0, &self.vx0, &self.vy0, dt);
        advect(n, 2, &mut self.vy, &self.vy0, &self.vx0, &self.vy0, dt);

        project(n, &mut self.vx, &mut self.vy, &mut self.vx0, &mut self.vy0);

        // Density step (RGB channels)
        add_source(n, &mut self.density_r, &self.density_r0, dt);
        add_source(n, &mut self.density_g, &self.density_g0, dt);
        add_source(n, &mut self.density_b, &self.density_b0, dt);

        std::mem::swap(&mut self.density_r0, &mut self.density_r);
        diffuse(n, 0, &mut self.density_r, &self.density_r0, diff, dt);
        std::mem::swap(&mut self.density_g0, &mut self.density_g);
        diffuse(n, 0, &mut self.density_g, &self.density_g0, diff, dt);
        std::mem::swap(&mut self.density_b0, &mut self.density_b);
        diffuse(n, 0, &mut self.density_b, &self.density_b0, diff, dt);

        std::mem::swap(&mut self.density_r0, &mut self.density_r);
        advect(n, 0, &mut self.density_r, &self.density_r0, &self.vx, &self.vy, dt);
        std::mem::swap(&mut self.density_g0, &mut self.density_g);
        advect(n, 0, &mut self.density_g, &self.density_g0, &self.vx, &self.vy, dt);
        std::mem::swap(&mut self.density_b0, &mut self.density_b);
        advect(n, 0, &mut self.density_b, &self.density_b0, &self.vx, &self.vy, dt);

        // Clear source arrays for next frame
        for v in self.vx0.iter_mut() { *v = 0.0; }
        for v in self.vy0.iter_mut() { *v = 0.0; }
        for v in self.density_r0.iter_mut() { *v = 0.0; }
        for v in self.density_g0.iter_mut() { *v = 0.0; }
        for v in self.density_b0.iter_mut() { *v = 0.0; }

        // Gentle decay to prevent dye accumulation
        for v in self.density_r.iter_mut() { *v *= 0.999; }
        for v in self.density_g.iter_mut() { *v *= 0.999; }
        for v in self.density_b.iter_mut() { *v *= 0.999; }
    }

    /// Render dye density to RGBA pixel buffer.
    pub fn render(&mut self) -> *const u8 {
        let n = self.n;
        for j in 0..n {
            for i in 0..n {
                let idx = ix(i + 1, j + 1, n);
                let pi = (j * n + i) * 4;

                let r = (self.density_r[idx]).clamp(0.0, 255.0) as u8;
                let g = (self.density_g[idx]).clamp(0.0, 255.0) as u8;
                let b = (self.density_b[idx]).clamp(0.0, 255.0) as u8;

                self.pixels[pi] = r;
                self.pixels[pi + 1] = g;
                self.pixels[pi + 2] = b;
                self.pixels[pi + 3] = 255;
            }
        }
        self.pixels.as_ptr()
    }

    /// Render with velocity-based color mapping for a more vivid visualization.
    pub fn render_velocity_colored(&mut self) -> *const u8 {
        let n = self.n;
        for j in 0..n {
            for i in 0..n {
                let idx = ix(i + 1, j + 1, n);
                let pi = (j * n + i) * 4;

                // Combine dye density with velocity-based coloring
                let dr = self.density_r[idx];
                let dg = self.density_g[idx];
                let db = self.density_b[idx];

                let vx = self.vx[idx];
                let vy = self.vy[idx];
                let speed = (vx * vx + vy * vy).sqrt();

                // Velocity adds a subtle glow
                let vel_brightness = (speed * 50.0).clamp(0.0, 40.0);

                let r = (dr + vel_brightness).clamp(0.0, 255.0) as u8;
                let g = (dg + vel_brightness * 0.5).clamp(0.0, 255.0) as u8;
                let b = (db + vel_brightness * 0.8).clamp(0.0, 255.0) as u8;

                self.pixels[pi] = r;
                self.pixels[pi + 1] = g;
                self.pixels[pi + 2] = b;
                self.pixels[pi + 3] = 255;
            }
        }
        self.pixels.as_ptr()
    }

    pub fn pixels_ptr(&self) -> *const u8 {
        self.pixels.as_ptr()
    }

    pub fn pixels_len(&self) -> usize {
        self.pixels.len()
    }

    pub fn reset(&mut self) {
        let size = (self.n + 2) * (self.n + 2);
        self.vx = vec![0.0; size];
        self.vy = vec![0.0; size];
        self.vx0 = vec![0.0; size];
        self.vy0 = vec![0.0; size];
        self.density_r = vec![0.0; size];
        self.density_g = vec![0.0; size];
        self.density_b = vec![0.0; size];
        self.density_r0 = vec![0.0; size];
        self.density_g0 = vec![0.0; size];
        self.density_b0 = vec![0.0; size];
        self.pixels = vec![0; self.n * self.n * 4];
    }

    pub fn clear_dye(&mut self) {
        let size = (self.n + 2) * (self.n + 2);
        self.density_r = vec![0.0; size];
        self.density_g = vec![0.0; size];
        self.density_b = vec![0.0; size];
        self.density_r0 = vec![0.0; size];
        self.density_g0 = vec![0.0; size];
        self.density_b0 = vec![0.0; size];
    }
}

// ── Helper functions for the Stable Fluids solver ──

#[inline(always)]
fn ix(i: usize, j: usize, n: usize) -> usize {
    i + (n + 2) * j
}

fn add_source(n: usize, x: &mut [f32], s: &[f32], dt: f32) {
    let size = (n + 2) * (n + 2);
    for i in 0..size {
        x[i] += dt * s[i];
    }
}

fn set_bnd(n: usize, b: usize, x: &mut [f32]) {
    for i in 1..=n {
        x[ix(0, i, n)] = if b == 1 { -x[ix(1, i, n)] } else { x[ix(1, i, n)] };
        x[ix(n + 1, i, n)] = if b == 1 { -x[ix(n, i, n)] } else { x[ix(n, i, n)] };
        x[ix(i, 0, n)] = if b == 2 { -x[ix(i, 1, n)] } else { x[ix(i, 1, n)] };
        x[ix(i, n + 1, n)] = if b == 2 { -x[ix(i, n, n)] } else { x[ix(i, n, n)] };
    }
    x[ix(0, 0, n)] = 0.5 * (x[ix(1, 0, n)] + x[ix(0, 1, n)]);
    x[ix(0, n + 1, n)] = 0.5 * (x[ix(1, n + 1, n)] + x[ix(0, n, n)]);
    x[ix(n + 1, 0, n)] = 0.5 * (x[ix(n, 0, n)] + x[ix(n + 1, 1, n)]);
    x[ix(n + 1, n + 1, n)] = 0.5 * (x[ix(n, n + 1, n)] + x[ix(n + 1, n, n)]);
}

fn diffuse(n: usize, b: usize, x: &mut [f32], x0: &[f32], diff: f32, dt: f32) {
    let a = dt * diff * (n as f32) * (n as f32);
    let c = 1.0 + 4.0 * a;
    // Gauss-Seidel relaxation
    for _ in 0..20 {
        for j in 1..=n {
            for i in 1..=n {
                x[ix(i, j, n)] = (x0[ix(i, j, n)]
                    + a * (x[ix(i - 1, j, n)]
                        + x[ix(i + 1, j, n)]
                        + x[ix(i, j - 1, n)]
                        + x[ix(i, j + 1, n)]))
                    / c;
            }
        }
        set_bnd(n, b, x);
    }
}

fn advect(n: usize, b: usize, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32], dt: f32) {
    let dt0 = dt * n as f32;
    let nf = n as f32;
    for j in 1..=n {
        for i in 1..=n {
            let mut x = i as f32 - dt0 * u[ix(i, j, n)];
            let mut y = j as f32 - dt0 * v[ix(i, j, n)];

            x = x.clamp(0.5, nf + 0.5);
            y = y.clamp(0.5, nf + 0.5);

            let i0 = x as usize;
            let i1 = i0 + 1;
            let j0 = y as usize;
            let j1 = j0 + 1;

            let s1 = x - i0 as f32;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f32;
            let t0 = 1.0 - t1;

            d[ix(i, j, n)] = s0 * (t0 * d0[ix(i0, j0, n)] + t1 * d0[ix(i0, j1, n)])
                + s1 * (t0 * d0[ix(i1, j0, n)] + t1 * d0[ix(i1, j1, n)]);
        }
    }
    set_bnd(n, b, d);
}

fn project(n: usize, u: &mut [f32], v: &mut [f32], p: &mut [f32], div: &mut [f32]) {
    let h = 1.0 / n as f32;
    for j in 1..=n {
        for i in 1..=n {
            div[ix(i, j, n)] =
                -0.5 * h * (u[ix(i + 1, j, n)] - u[ix(i - 1, j, n)] + v[ix(i, j + 1, n)] - v[ix(i, j - 1, n)]);
            p[ix(i, j, n)] = 0.0;
        }
    }
    set_bnd(n, 0, div);
    set_bnd(n, 0, p);

    // Gauss-Seidel
    for _ in 0..20 {
        for j in 1..=n {
            for i in 1..=n {
                p[ix(i, j, n)] = (div[ix(i, j, n)]
                    + p[ix(i - 1, j, n)]
                    + p[ix(i + 1, j, n)]
                    + p[ix(i, j - 1, n)]
                    + p[ix(i, j + 1, n)])
                    / 4.0;
            }
        }
        set_bnd(n, 0, p);
    }

    for j in 1..=n {
        for i in 1..=n {
            u[ix(i, j, n)] -= 0.5 * (p[ix(i + 1, j, n)] - p[ix(i - 1, j, n)]) * n as f32;
            v[ix(i, j, n)] -= 0.5 * (p[ix(i, j + 1, n)] - p[ix(i, j - 1, n)]) * n as f32;
        }
    }
    set_bnd(n, 1, u);
    set_bnd(n, 2, v);
}
