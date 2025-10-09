import polars as pl
from kinamax.core import H46Problem, detect_orbits

working_dir = "outputs"
simulations = pl.read_parquet(f"{working_dir}/simulations.parquet")
state_vec_labels = H46Problem.state_vector_labels
attractor_state_vec_labels = [f"{k}a" for k in state_vec_labels]
ode_params_labels = H46Problem.params_labels

attractors, sim_orbit = detect_orbits(
    problem_class=H46Problem,
    simulations=simulations,
    ode_params_labels=ode_params_labels,
    attractor_state_vec_labels=attractor_state_vec_labels,
    state_vec_labels=state_vec_labels,
)
attractors.write_parquet(f"{working_dir}/orbits.parquet")
sim_orbit.write_parquet(f"{working_dir}/sim_orbit.parquet")
