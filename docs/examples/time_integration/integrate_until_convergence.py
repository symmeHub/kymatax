"""
Batch integration example: find periodic attractors of the H46 problem
over a small frequency sweep and a set of initial conditions, then
assemble the results into a Polars DataFrame and write them to Parquet.

Key steps
- Configure JAX precision and ODE solver/controller.
- Define the frequency sweep and attractor finder configuration.
- Build the H46 problem and randomized initial conditions.
- Vectorize (vmap) and JIT-compile the attractor search across both
  initial conditions and frequencies.
- Collect, tidy, and export results.
"""

import os
from jax import numpy as jnp
import jax
from jax import config, vmap
from diffrax import Tsit5, PIDController
import numpy as np
import polars as pl
import equinox as eqx
from kinamax.core import H46Problem, AttractorFinder, AttractorFinderConfig

config.update("jax_enable_x64", True)  # Use double precision for improved accuracy


# Frequency sweep (Hz): here just 2 points between 30 and 50 Hz
fd = jnp.linspace(50, 51.0, 1)
# Attractor finder configuration; only target_frequency varies with the sweep
finder_config = AttractorFinderConfig(
    convergence_tol=1.0e-10,
    target_frequency=fd,
    init_time=0.0,
    init_time_step=1.0e-3,
    subharmonic_factor=10,
)
# ODE solver and adaptive time-step controller
solver = Tsit5()
controller = PIDController(rtol=1e-8, atol=1e-9)
# Subharmonics we intend to target/detect in the orbits
target_subharmonics = np.array([1, 2, 3, 4, 5], int)
# Attractor finder driver; stop after at most `max_periods` periods
attractor_finder = AttractorFinder(
    residuals_per_period=10,
    targetted_subharmonics=target_subharmonics,
    max_periods=5000,
    controller=controller,
    solver=solver,
)
# H46 problem definition; `Ad` is the drive amplitude
problem = H46Problem(fd=fd, Ad=2.5)
# PRNG for randomized initial conditions
key = jax.random.PRNGKey(758493)
# Build a batch of initial conditions (20 samples, [x0, xdot0]).
# Samples are drawn uniformly in [-1, 1]^2 and scaled by problem-specific
# length/time scales so we explore a reasonable state-space region.
init_conditions = (
    (jax.random.uniform(key, shape=(20, 2)) - 0.5)
    * 2
    * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0])
)
find_attractors = attractor_finder.find_attractors

# Vectorize and JIT the attractor search:
# - inner vmap: over initial conditions (axis 0 of the 2D array)
# - outer vmap: over the frequency sweep by mapping the `fd` field in the
#   H46Problem and the `target_frequency` field in AttractorFinderConfig
jvfind_attractors = eqx.filter_jit(
    vmap(
        vmap(find_attractors, in_axes=(None, 0, None)),
        in_axes=(
            H46Problem(fd=0, xw=None, Q=None, Ad=None, w0=None),
            None,
            AttractorFinderConfig(
                init_time=None,
                init_time_step=None,
                convergence_tol=None,
                target_frequency=0,
                subharmonic_factor=None,
            ),
        ),
    )
)

# Run the batched attractor search across all frequencies and initial conditions
problems, finder_configs, vmaped_init_conditions, solutions = jvfind_attractors(
    problem,
    init_conditions,
    finder_config,
)

# POST PROCESSING
# Flatten the initial condition batch to 2 columns [x0, xdot0]
state_space_dim = len(H46Problem.state_vector_labels)
vmaped_init_conditions = np.array(vmaped_init_conditions).reshape(-1, state_space_dim)

# Prepare a DataFrame with the initial conditions, repeated so that each
# row can align with multiple subharmonic results per simulation.
max_attractors = target_subharmonics.max()
df_init_conditions = pl.DataFrame(
    {
        "x0": vmaped_init_conditions[:, 0].flatten().repeat(max_attractors),
        "dotx0": vmaped_init_conditions[:, 1].flatten().repeat(max_attractors),
    }
)
raw_data = pl.concat(
    [
        problems.as_polars(repeat=max_attractors),
        finder_configs.as_polars(repeat=max_attractors),
        df_init_conditions,
        solutions.as_polars(state_vector_labels=["xa", "dotxa"]),
    ],
    how="horizontal",
)
# Balance the number of rows per simulation across detected subharmonics.
# For a given detected subharmonic `sh`, keep the first `sh` rows within
# each `sim_label`. Special-case `sh == 0` (e.g., no subharmonic detected 
# which can mean chaos or aperiodic orbits) by taking the same number of 
# rows as the maximum detected subharmonic.
data_list = []
unique_detected_subharmonics = raw_data["detected_subharmonic"].unique()
for sh in unique_detected_subharmonics:
    if sh != 0:
        data_list.append(
            raw_data.filter(pl.col("detected_subharmonic") == sh)
            .group_by("sim_label")
            .head(sh)
        )
    else:
        data_list.append(
            raw_data.filter(pl.col("detected_subharmonic") == sh)
            .group_by("sim_label")
            .head(unique_detected_subharmonics.max())
        )
data = pl.concat(data_list, how="vertical")
data = data.sort(["sim_label", "attractor_label"])
# Write the consolidated results to disk

output_dir = "outputs"
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
data.write_parquet(f"{output_dir}/simulations.parquet")
