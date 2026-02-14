#include "fluid_sim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten.h>

// ── Helpers ──

#define IX(i, j, n) ((i) + ((n) + 2) * (j))

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline void swap_ptrs(float **a, float **b) {
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

// ── Lifecycle ──

EMSCRIPTEN_KEEPALIVE
FluidSim *fluid_sim_new(int n) {
    FluidSim *sim = (FluidSim *)calloc(1, sizeof(FluidSim));
    if (!sim) return NULL;

    int size = (n + 2) * (n + 2);
    sim->n    = n;
    sim->dt   = 0.1f;
    sim->visc = 0.0001f;
    sim->diff = 0.00001f;

    sim->vx  = (float *)calloc(size, sizeof(float));
    sim->vy  = (float *)calloc(size, sizeof(float));
    sim->vx0 = (float *)calloc(size, sizeof(float));
    sim->vy0 = (float *)calloc(size, sizeof(float));

    sim->density_r  = (float *)calloc(size, sizeof(float));
    sim->density_g  = (float *)calloc(size, sizeof(float));
    sim->density_b  = (float *)calloc(size, sizeof(float));
    sim->density_r0 = (float *)calloc(size, sizeof(float));
    sim->density_g0 = (float *)calloc(size, sizeof(float));
    sim->density_b0 = (float *)calloc(size, sizeof(float));

    sim->pixels  = (uint8_t *)calloc(n * n * 4, sizeof(uint8_t));
    sim->scratch = (float *)calloc(size, sizeof(float));

    return sim;
}

EMSCRIPTEN_KEEPALIVE
void fluid_sim_free(FluidSim *sim) {
    if (!sim) return;
    free(sim->vx);   free(sim->vy);
    free(sim->vx0);  free(sim->vy0);
    free(sim->density_r);  free(sim->density_g);  free(sim->density_b);
    free(sim->density_r0); free(sim->density_g0); free(sim->density_b0);
    free(sim->pixels);
    free(sim->scratch);
    free(sim);
}

// ── Parameter setters ──

EMSCRIPTEN_KEEPALIVE
void fluid_sim_set_dt(FluidSim *sim, float dt) { sim->dt = dt; }

EMSCRIPTEN_KEEPALIVE
void fluid_sim_set_viscosity(FluidSim *sim, float v) { sim->visc = v; }

EMSCRIPTEN_KEEPALIVE
void fluid_sim_set_diffusion(FluidSim *sim, float d) { sim->diff = d; }

EMSCRIPTEN_KEEPALIVE
int fluid_sim_grid_size(FluidSim *sim) { return sim->n; }

// ── Interaction ──

EMSCRIPTEN_KEEPALIVE
void fluid_sim_add_velocity(FluidSim *sim, float x, float y, float fx, float fy) {
    int n = sim->n;
    int i = (int)clampf(x, 0.0f, (float)(n - 1)) + 1;
    int j = (int)clampf(y, 0.0f, (float)(n - 1)) + 1;
    int idx = IX(i, j, n);
    sim->vx0[idx] += fx;
    sim->vy0[idx] += fy;
}

EMSCRIPTEN_KEEPALIVE
void fluid_sim_add_dye(FluidSim *sim, float x, float y,
                        float r, float g, float b, float radius) {
    int n = sim->n;
    int rad = (int)radius;
    int cx = (int)x;
    int cy = (int)y;

    for (int di = -rad; di <= rad; di++) {
        for (int dj = -rad; dj <= rad; dj++) {
            float dist_sq = (float)(di * di + dj * dj);
            if (dist_sq > radius * radius) continue;

            int pi = cx + di;
            int pj = cy + dj;
            if (pi >= 0 && pi < n && pj >= 0 && pj < n) {
                int idx = IX(pi + 1, pj + 1, n);
                float falloff = 1.0f - (dist_sq / (radius * radius));
                sim->density_r0[idx] += r * falloff * 1000.0f;
                sim->density_g0[idx] += g * falloff * 1000.0f;
                sim->density_b0[idx] += b * falloff * 1000.0f;
            }
        }
    }
}

// ── Solver internals ──

static void add_source(int n, float *x, const float *s, float dt) {
    int size = (n + 2) * (n + 2);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(int n, int b, float *x) {
    for (int i = 1; i <= n; i++) {
        x[IX(0,     i, n)] = (b == 1) ? -x[IX(1, i, n)] : x[IX(1, i, n)];
        x[IX(n + 1, i, n)] = (b == 1) ? -x[IX(n, i, n)] : x[IX(n, i, n)];
        x[IX(i,     0, n)] = (b == 2) ? -x[IX(i, 1, n)] : x[IX(i, 1, n)];
        x[IX(i, n + 1, n)] = (b == 2) ? -x[IX(i, n, n)] : x[IX(i, n, n)];
    }
    x[IX(0,     0,     n)] = 0.5f * (x[IX(1, 0, n)]     + x[IX(0, 1, n)]);
    x[IX(0,     n + 1, n)] = 0.5f * (x[IX(1, n + 1, n)] + x[IX(0, n, n)]);
    x[IX(n + 1, 0,     n)] = 0.5f * (x[IX(n, 0, n)]     + x[IX(n + 1, 1, n)]);
    x[IX(n + 1, n + 1, n)] = 0.5f * (x[IX(n, n + 1, n)] + x[IX(n + 1, n, n)]);
}

static void diffuse(int n, int b, float *x, const float *x0,
                     float diff, float dt, float *x_new) {
    float a = dt * diff * (float)n * (float)n;
    float c = 1.0f + 4.0f * a;
    int size = (n + 2) * (n + 2);

    for (int iter = 0; iter < 28; iter++) {
        #pragma omp parallel for
        for (int idx = 0; idx < size; idx++) {
            int i = idx % (n + 2);
            int j = idx / (n + 2);
            if (i >= 1 && i <= n && j >= 1 && j <= n) {
                x_new[idx] = (x0[IX(i, j, n)]
                    + a * (x[IX(i - 1, j, n)]
                         + x[IX(i + 1, j, n)]
                         + x[IX(i, j - 1, n)]
                         + x[IX(i, j + 1, n)])) / c;
            }
        }
        memcpy(x, x_new, size * sizeof(float));
        set_bnd(n, b, x);
    }
}

static void advect(int n, int b, float *d, const float *d0,
                    const float *u, const float *v, float dt, float *d_new) {
    float dt0 = dt * (float)n;
    float nf = (float)n;
    int stride = n + 2;
    int size = stride * stride;

    #pragma omp parallel for
    for (int idx = 0; idx < size; idx++) {
        int i = idx % stride;
        int j = idx / stride;
        if (i >= 1 && i <= n && j >= 1 && j <= n) {
            float x = (float)i - dt0 * u[IX(i, j, n)];
            float y = (float)j - dt0 * v[IX(i, j, n)];

            x = clampf(x, 0.5f, nf + 0.5f);
            y = clampf(y, 0.5f, nf + 0.5f);

            int i0 = (int)x;
            int i1 = i0 + 1;
            int j0 = (int)y;
            int j1 = j0 + 1;

            float s1 = x - (float)i0;
            float s0 = 1.0f - s1;
            float t1 = y - (float)j0;
            float t0 = 1.0f - t1;

            d_new[idx] = s0 * (t0 * d0[IX(i0, j0, n)] + t1 * d0[IX(i0, j1, n)])
                       + s1 * (t0 * d0[IX(i1, j0, n)] + t1 * d0[IX(i1, j1, n)]);
        }
    }

    memcpy(d, d_new, size * sizeof(float));
    set_bnd(n, b, d);
}

static void project(int n, float *u, float *v, float *p, float *div, float *p_new) {
    float h = 1.0f / (float)n;
    int stride = n + 2;
    int size = stride * stride;

    // Compute divergence
    #pragma omp parallel for
    for (int idx = 0; idx < size; idx++) {
        int i = idx % stride;
        int j = idx / stride;
        if (i >= 1 && i <= n && j >= 1 && j <= n) {
            div[idx] = -0.5f * h *
                (u[IX(i + 1, j, n)] - u[IX(i - 1, j, n)]
               + v[IX(i, j + 1, n)] - v[IX(i, j - 1, n)]);
        }
    }

    // Zero pressure
    memset(p, 0, size * sizeof(float));

    set_bnd(n, 0, div);
    set_bnd(n, 0, p);

    // Jacobi iteration for pressure
    for (int iter = 0; iter < 28; iter++) {
        #pragma omp parallel for
        for (int idx = 0; idx < size; idx++) {
            int i = idx % stride;
            int j = idx / stride;
            if (i >= 1 && i <= n && j >= 1 && j <= n) {
                p_new[idx] = (div[IX(i, j, n)]
                    + p[IX(i - 1, j, n)]
                    + p[IX(i + 1, j, n)]
                    + p[IX(i, j - 1, n)]
                    + p[IX(i, j + 1, n)]) / 4.0f;
            }
        }
        memcpy(p, p_new, size * sizeof(float));
        set_bnd(n, 0, p);
    }

    // Subtract pressure gradient
    float nf = (float)n;
    #pragma omp parallel for
    for (int idx = 0; idx < size; idx++) {
        int i = idx % stride;
        int j = idx / stride;
        if (i >= 1 && i <= n && j >= 1 && j <= n) {
            u[idx] -= 0.5f * (p[IX(i + 1, j, n)] - p[IX(i - 1, j, n)]) * nf;
            v[idx] -= 0.5f * (p[IX(i, j + 1, n)] - p[IX(i, j - 1, n)]) * nf;
        }
    }
    set_bnd(n, 1, u);
    set_bnd(n, 2, v);
}

// ── Main simulation step ──

EMSCRIPTEN_KEEPALIVE
void fluid_sim_step(FluidSim *sim) {
    int n    = sim->n;
    float dt   = sim->dt;
    float visc = sim->visc;
    float diff = sim->diff;
    float *scratch = sim->scratch;

    // Velocity step
    add_source(n, sim->vx, sim->vx0, dt);
    add_source(n, sim->vy, sim->vy0, dt);

    swap_ptrs(&sim->vx0, &sim->vx);
    diffuse(n, 1, sim->vx, sim->vx0, visc, dt, scratch);
    swap_ptrs(&sim->vy0, &sim->vy);
    diffuse(n, 2, sim->vy, sim->vy0, visc, dt, scratch);

    project(n, sim->vx, sim->vy, sim->vx0, sim->vy0, scratch);

    swap_ptrs(&sim->vx0, &sim->vx);
    swap_ptrs(&sim->vy0, &sim->vy);
    advect(n, 1, sim->vx, sim->vx0, sim->vx0, sim->vy0, dt, scratch);
    advect(n, 2, sim->vy, sim->vy0, sim->vx0, sim->vy0, dt, scratch);

    project(n, sim->vx, sim->vy, sim->vx0, sim->vy0, scratch);

    // Density step (RGB channels)
    add_source(n, sim->density_r, sim->density_r0, dt);
    add_source(n, sim->density_g, sim->density_g0, dt);
    add_source(n, sim->density_b, sim->density_b0, dt);

    swap_ptrs(&sim->density_r0, &sim->density_r);
    diffuse(n, 0, sim->density_r, sim->density_r0, diff, dt, scratch);
    swap_ptrs(&sim->density_g0, &sim->density_g);
    diffuse(n, 0, sim->density_g, sim->density_g0, diff, dt, scratch);
    swap_ptrs(&sim->density_b0, &sim->density_b);
    diffuse(n, 0, sim->density_b, sim->density_b0, diff, dt, scratch);

    swap_ptrs(&sim->density_r0, &sim->density_r);
    advect(n, 0, sim->density_r, sim->density_r0, sim->vx, sim->vy, dt, scratch);
    swap_ptrs(&sim->density_g0, &sim->density_g);
    advect(n, 0, sim->density_g, sim->density_g0, sim->vx, sim->vy, dt, scratch);
    swap_ptrs(&sim->density_b0, &sim->density_b);
    advect(n, 0, sim->density_b, sim->density_b0, sim->vx, sim->vy, dt, scratch);

    // Clear source arrays for next frame
    int size = (n + 2) * (n + 2);
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        sim->vx0[i] = 0.0f;
        sim->vy0[i] = 0.0f;
        sim->density_r0[i] = 0.0f;
        sim->density_g0[i] = 0.0f;
        sim->density_b0[i] = 0.0f;
    }

    // Gentle decay to prevent dye accumulation
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        sim->density_r[i] *= 0.999f;
        sim->density_g[i] *= 0.999f;
        sim->density_b[i] *= 0.999f;
    }
}

// ── Rendering ──

EMSCRIPTEN_KEEPALIVE
void fluid_sim_render(FluidSim *sim) {
    int n = sim->n;
    const float *dr = sim->density_r;
    const float *dg = sim->density_g;
    const float *db = sim->density_b;
    uint8_t *pixels = sim->pixels;

    #pragma omp parallel for
    for (int px = 0; px < n * n; px++) {
        int i = px % n;
        int j = px / n;
        int idx = IX(i + 1, j + 1, n);

        int base = px * 4;
        pixels[base + 0] = (uint8_t)clampf(dr[idx], 0.0f, 255.0f);
        pixels[base + 1] = (uint8_t)clampf(dg[idx], 0.0f, 255.0f);
        pixels[base + 2] = (uint8_t)clampf(db[idx], 0.0f, 255.0f);
        pixels[base + 3] = 255;
    }
}

EMSCRIPTEN_KEEPALIVE
void fluid_sim_render_velocity_colored(FluidSim *sim) {
    int n = sim->n;
    const float *dr = sim->density_r;
    const float *dg = sim->density_g;
    const float *db = sim->density_b;
    const float *vx = sim->vx;
    const float *vy = sim->vy;
    uint8_t *pixels = sim->pixels;

    #pragma omp parallel for
    for (int px = 0; px < n * n; px++) {
        int i = px % n;
        int j = px / n;
        int idx = IX(i + 1, j + 1, n);

        float vxi = vx[idx];
        float vyi = vy[idx];
        float speed = sqrtf(vxi * vxi + vyi * vyi);
        float vel_brightness = clampf(speed * 50.0f, 0.0f, 40.0f);

        int base = px * 4;
        pixels[base + 0] = (uint8_t)clampf(dr[idx] + vel_brightness,       0.0f, 255.0f);
        pixels[base + 1] = (uint8_t)clampf(dg[idx] + vel_brightness * 0.5f, 0.0f, 255.0f);
        pixels[base + 2] = (uint8_t)clampf(db[idx] + vel_brightness * 0.8f, 0.0f, 255.0f);
        pixels[base + 3] = 255;
    }
}

// ── Memory access ──

EMSCRIPTEN_KEEPALIVE
uint8_t *fluid_sim_pixels_ptr(FluidSim *sim) {
    return sim->pixels;
}

EMSCRIPTEN_KEEPALIVE
int fluid_sim_pixels_len(FluidSim *sim) {
    return sim->n * sim->n * 4;
}

// ── Reset ──

EMSCRIPTEN_KEEPALIVE
void fluid_sim_reset(FluidSim *sim) {
    int size = (sim->n + 2) * (sim->n + 2);
    memset(sim->vx,  0, size * sizeof(float));
    memset(sim->vy,  0, size * sizeof(float));
    memset(sim->vx0, 0, size * sizeof(float));
    memset(sim->vy0, 0, size * sizeof(float));
    memset(sim->density_r,  0, size * sizeof(float));
    memset(sim->density_g,  0, size * sizeof(float));
    memset(sim->density_b,  0, size * sizeof(float));
    memset(sim->density_r0, 0, size * sizeof(float));
    memset(sim->density_g0, 0, size * sizeof(float));
    memset(sim->density_b0, 0, size * sizeof(float));
    memset(sim->pixels, 0, sim->n * sim->n * 4);
}

EMSCRIPTEN_KEEPALIVE
void fluid_sim_clear_dye(FluidSim *sim) {
    int size = (sim->n + 2) * (sim->n + 2);
    memset(sim->density_r,  0, size * sizeof(float));
    memset(sim->density_g,  0, size * sizeof(float));
    memset(sim->density_b,  0, size * sizeof(float));
    memset(sim->density_r0, 0, size * sizeof(float));
    memset(sim->density_g0, 0, size * sizeof(float));
    memset(sim->density_b0, 0, size * sizeof(float));
}
