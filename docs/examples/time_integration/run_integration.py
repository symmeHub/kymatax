import os
from jax import numpy as jnp
import jax
from jax import config, vmap
from diffrax import Tsit5, PIDController
import numpy as np
import polars as pl
import equinox as eqx
from kymatax.core import H46Problem, OrbitFinder, OrbitFinderConfig

config.update("jax_enable_x64", True)  # DOUBLE PRECISION


fd = jnp.linspace(30.0, 50.0, 2)  # Frequency range from 10 to 100 Hz
finder_config = OrbitFinderConfig(
    convergence_tol=1.0e-10,
    target_frequency=fd,
    init_time=0.0,
    init_time_step=1.0e-3,
    subharmonic_factor=100,
)
solver = Tsit5()
controller = PIDController(rtol=1e-8, atol=1e-9)
target_subharmonics = np.array([1, 2, 3, 4, 5], int)  # Target subharmonics
orbit_finder = OrbitFinder(
    residuals_per_period=10,
    targetted_subharmonics=target_subharmonics,
    max_periods=5000,
    controller=controller,
    solver=solver,
)
problem = H46Problem(fd=fd, Ad=2.5)
key = jax.random.PRNGKey(758493)
init_conditions = (
    (jax.random.uniform(key, shape=(20, 2)) - 0.5)
    * 2
    * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0])
)  # Example initial conditions
find_orbits = orbit_finder.find_orbits

jvfind_orbits = eqx.filter_jit(
    vmap(
        vmap(find_orbits, in_axes=(None, 0, None)),
        in_axes=(
            H46Problem(fd=0, xw=None, Q=None, Ad=None, w0=None),
            None,
            OrbitFinderConfig(
                init_time=None,
                init_time_step=None,
                convergence_tol=None,
                target_frequency=0,
                subharmonic_factor=None,
            ),
        ),
    )
)

problems, finder_configs, vmaped_init_conditions, solutions = jvfind_orbits(
    problem,
    init_conditions,
    finder_config,
)

max_attractors = target_subharmonics.max()
vmaped_init_conditions = np.array(vmaped_init_conditions).reshape(-1, 2)
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


data_list= []
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
if os.path.exists("data") is False:
    os.mkdir("data")
data.write_parquet("data/simulations.parquet")