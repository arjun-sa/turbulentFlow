CC = emcc
SRC = src/fluid_sim.c
OUT = pkg/fluid_sim.js

EXPORTED_FUNCS = '["_fluid_sim_new","_fluid_sim_free","_fluid_sim_set_dt", \
  "_fluid_sim_set_viscosity","_fluid_sim_set_diffusion","_fluid_sim_add_velocity", \
  "_fluid_sim_add_dye","_fluid_sim_step","_fluid_sim_render", \
  "_fluid_sim_render_velocity_colored","_fluid_sim_pixels_ptr","_fluid_sim_pixels_len", \
  "_fluid_sim_grid_size","_fluid_sim_reset","_fluid_sim_clear_dye","_malloc","_free"]'

CFLAGS = -O3 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MAXIMUM_MEMORY=1073741824 \
  -s EXPORTED_FUNCTIONS=$(EXPORTED_FUNCS) \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]' \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=FluidSimModule \
  -s EXPORT_ES6=1 \
  -s ENVIRONMENT='web,worker'

all: $(OUT)

$(OUT): $(SRC) src/fluid_sim.h
	mkdir -p pkg
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f pkg/fluid_sim.js pkg/fluid_sim.wasm pkg/fluid_sim.worker.js

.PHONY: all clean
