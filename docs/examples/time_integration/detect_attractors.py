import polars as pl
from kymatax.core import H46Problem, detect_attractors_orbits

simulations = pl.read_parquet("data/simulations.parquet")
state_vec_labels = H46Problem.state_vector_labels
attractor_state_vec_labels = [f"{k}a" for k in state_vec_labels]
ode_params_labels = H46Problem.params_labels




attractors, sim_orbit = detect_attractors_orbits(simulations, 
                                                  ode_params_labels, 
                                                  attractor_state_vec_labels, 
                                                  state_vec_labels)
attractors.write_parquet("data/attractors.parquet")
sim_orbit.write_parquet("data/sim_orbit.parquet")
