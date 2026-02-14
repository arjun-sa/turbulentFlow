#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int n;           // grid resolution (interior cells per side)
    float dt;        // timestep
    float visc;      // viscosity
    float diff;      // diffusion rate for dye

    // velocity components
    float *vx, *vy, *vx0, *vy0;

    // dye density (RGB channels)
    float *density_r, *density_g, *density_b;
    float *density_r0, *density_g0, *density_b0;

    // output pixel buffer (RGBA)
    uint8_t *pixels;

    // scratch buffer for Jacobi iteration
    float *scratch;
} FluidSim;

// Lifecycle
FluidSim *fluid_sim_new(int n);
void fluid_sim_free(FluidSim *sim);

// Parameter setters
void fluid_sim_set_dt(FluidSim *sim, float dt);
void fluid_sim_set_viscosity(FluidSim *sim, float v);
void fluid_sim_set_diffusion(FluidSim *sim, float d);
int  fluid_sim_grid_size(FluidSim *sim);

// Interaction
void fluid_sim_add_velocity(FluidSim *sim, float x, float y, float fx, float fy);
void fluid_sim_add_dye(FluidSim *sim, float x, float y, float r, float g, float b, float radius);

// Simulation
void fluid_sim_step(FluidSim *sim);

// Rendering
void fluid_sim_render(FluidSim *sim);
void fluid_sim_render_velocity_colored(FluidSim *sim);

// Memory access
uint8_t *fluid_sim_pixels_ptr(FluidSim *sim);
int fluid_sim_pixels_len(FluidSim *sim);

// Reset
void fluid_sim_reset(FluidSim *sim);
void fluid_sim_clear_dye(FluidSim *sim);

#endif
