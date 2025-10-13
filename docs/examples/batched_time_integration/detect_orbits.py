import polars as pl
from kinamax.core import detect_orbits
from kinamax.problems import H46Problem

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

# Set Eh to zero for orbits since it's not relevant for the attractor detection.
attractors = attractors.with_columns(Eh = attractors["Eh"] * 0.)
attractors.write_parquet(f"{working_dir}/orbits.parquet")
sim_orbit.write_parquet(f"{working_dir}/sim_orbit.parquet")
