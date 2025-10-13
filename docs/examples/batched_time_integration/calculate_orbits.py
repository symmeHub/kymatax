# Calculate orbits from attractors.

import jax
import polars as pl
from kinamax.problems import H46Problem
import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, diffeqsolve, Tsit5, PIDController
from typing import NamedTuple
import numpy as np


working_dir = "outputs"
simulations = pl.read_parquet(f"{working_dir}/simulations.parquet")
orbits = pl.read_parquet(f"{working_dir}/orbits.parquet")
sim_orbit = pl.read_parquet(f"{working_dir}/sim_orbit.parquet")
state_vec_labels = H46Problem.state_vector_labels
attractor_state_vec_labels = [f"{k}a" for k in state_vec_labels]
ode_params_labels = H46Problem.params_labels


def tutu():
    attractor_orbit_map = dict(zip(orbits["attractor_label"].to_list(), orbits["orbit_label"].to_list()))
    orbit_to_sim = sim_orbit.group_by("orbit_label").first()
    orbit_sim_map = dict(zip(orbit_to_sim["orbit_label"].to_list(), orbit_to_sim["sim_label"].to_list()))
    attractor_sim_map = {k: orbit_sim_map[v] for k, v in attractor_orbit_map.items()}

    order_df = (
        pl.DataFrame({"sim_label": list(attractor_sim_map.values())})
        .with_row_index("order")
    )

    unique_sims = simulations.unique(subset=["sim_label"], keep="first")

    stacked = (
        order_df
        .join(unique_sims, on="sim_label", how="left")
        .sort("order")
        .drop("order")
    )
    return stacked

stacked_sims = tutu()
ode_params = {k: jnp.array(stacked_sims[k].to_numpy()) for k in ode_params_labels}
sim_labels = jnp.array(stacked_sims["sim_label"].to_numpy())
target_frequencies = jnp.array(stacked_sims["target_frequency"].to_numpy())
init_times = jnp.array(stacked_sims["init_time"].to_numpy())
init_time_steps = jnp.array(stacked_sims["init_time_step"].to_numpy())
Xa = jnp.array(orbits[state_vec_labels].to_numpy())


aid = 0
problem = H46Problem(**{k: ode_params[k][aid] for k in ode_params_labels})
problems = H46Problem(**ode_params)

class OrbitCalculator(NamedTuple):
    samples_per_period : int =np.array(60)

    
    def calculate_orbit(self, problem,Xa, init_time, target_frequency, init_time_step=1e-4):
        """
        Calculate orbits from attractors.
        Args:
            Xa (jnp.ndarray): Attractor states.
            init_time (jnp.ndarray): Initial times.
            problem (H46Problem): Problem instance.
        Returns:
            jnp.ndarray: Orbits.
        """
        # problems = H46Problem(**ode_params)
        Ns = self.samples_per_period
        solver = Tsit5()
        controller = PIDController(rtol=1e-8, atol=1e-9)
        term = ODETerm(problem.rhs)
        t0 = init_time
        fd = target_frequency
        Td = 1.0 / fd
        t1 = t0 + Td
        dt0 = init_time_step
        # print(f"Calculating orbit from t={t0} to t={t1} with dt0={dt0} and N={Ns}" )
        tg = jnp.arange(0, np.array(Ns))
        # tg[0] = 0.
        # tg[1] = 1.0
        ts = t0 + tg * (Td / Ns)
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0=Xa,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )
        return sol.ys




calculator = OrbitCalculator()
calculator_fn = jax.jit(jax.vmap(calculator.calculate_orbit))
# orbit = calculator_fn(problem, Xa=Xa[aid], init_time=init_times[aid], target_frequency=target_frequencies[aid], init_time_step=init_time_steps[aid])
orbits = calculator_fn(problems, Xa=Xa, init_time=init_times, target_frequency=target_frequencies, init_time_step=init_time_steps)