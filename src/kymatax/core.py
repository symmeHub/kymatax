import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from dataclasses import dataclass, field
from jax.tree_util import register_dataclass
from jax import lax, vmap
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5
from diffrax import PIDController
from typing import NamedTuple, ClassVar
from collections import namedtuple

try:
    import cuml
except:
    pass
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from jax import jit
import networkx as nx


@dataclass
class Container:
    """
    Container class for holding various parameters.
    """

    def as_dict(self, flatten=True):
        """
        Converts the problem to a dictionary.
        Returns:
            dict: Dictionary representation of the problem.
        """
        raw_dic = self.__dict__
        dic = {}
        for key, value in raw_dic.items():
            if flatten:
                v = value.flatten()
            else:
                v = value
            dic[key] = np.array(v)
        return dic

    def as_polars(self, repeat=0):
        """
        Converts the problem to a Polars DataFrame.
        Returns:
            pl.DataFrame: Polars DataFrame representation of the problem.
        """
        dic = self.as_dict()
        # keys = list(dic.keys())
        # s = len(dic[keys[0]])
        # label = np.arange(s)
        # label_name = self.label_col if hasattr(self, "label_col") else "label"
        # dic[label_name] = label
        if repeat > 0:
            for key, value in dic.items():
                dic[key] = value.repeat(repeat)
        df = pl.DataFrame(dic)
        return df


@register_dataclass
@dataclass
class H46Problem(Container):
    """
    H46 Problem
    """

    xw: jax.Array = 0.5e-3
    fd: jax.Array = 50.0
    w0: jax.Array = 121.0
    Q: jax.Array = 87.0
    Ad: jax.Array = 2.5
    state_vector_labels: ClassVar = ["x", "dotx"]
    params_labels: ClassVar = ["xw", "w0", "Ad", "Q", "fd"]
    # label_col: ClassVar = "problem_id"

    def state_weights(self):
        """
        Returns the state weights for the system.
        Returns:
            jnp.ndarray: State weights.
        """
        xw = self.xw
        w0 = self.w0
        return jnp.array([1.0 / xw, 1.0 / (w0 * xw)])

    def rhs(self, t, X, args=None):
        """
        Right-hand side of the ODE.
        Args:
            t (float): Time.
            X (jnp.ndarray): State vector.
            args (tuple, optional): Additional arguments. Defaults to None.
        Returns:
            jnp.ndarray: Derivative of the state vector.
        """
        xw = self.xw
        w0 = self.w0
        Q = self.Q
        fd = self.fd
        Ad = self.Ad
        x, dotx = X
        wd = 2.0 * jnp.pi * fd
        ddotx = (
            -(jnp.pow(w0, 2)) / 2.0 * (jnp.pow(x / xw, 2) - 1.0) * x
            - w0 / Q * dotx
            + Ad * jnp.sin(wd * t)
        )
        Xout = jnp.array([dotx, ddotx])
        return Xout


@register_dataclass
@dataclass
class OrbitFinderConfig(Container):
    init_time: jax.Array = field(default_factory=lambda: jnp.array(0.0, dtype=float))
    init_time_step: jax.Array = field(
        default_factory=lambda: jnp.array(1.0e-3, dtype=float)
    )
    convergence_tol: jax.Array = field(
        default_factory=lambda: jnp.array(1.0e-6, dtype=float)
    )
    target_frequency: float = 50.0
    subharmonic_factor: float = 10.0
    # label_col: ClassVar = "config_id"


def convert_subharmonics_flags(subharmonics_flags, final_flags, targetted_subharmonics):
    """
    Convert the detected subharmonics into a boolean array.
    """
    detected_subharmonics = np.zeros(final_flags.shape, dtype=np.int32)
    for i in range(len(subharmonics_flags)):
        sh = subharmonics_flags[i]
        if final_flags[i] == 1:
            detected_subharmonics[i] = targetted_subharmonics[i, sh == 1][0]
    return detected_subharmonics


@register_dataclass
@dataclass
class OrbitFinderSolution(Container):
    """
    Solution of the orbit finder.
    """

    attractors: jax.Array
    detected_subharmonic: jax.Array
    subharmonic_residual: jax.Array
    minimum_residual: jax.Array
    simulated_periods: jax.Array
    simulated_time: jax.Array
    final_flag: jax.Array
    simulated_iterations: jax.Array
    converged: jax.Array

    def as_dict(self, state_vector_labels=None):
        """
        Converts the problem to a dictionary.
        Returns:
            dict: Dictionary representation of the problem.
        """
        raw_dic = self.__dict__
        dic = {}
        s = self.detected_subharmonic.shape
        orbit = np.arange(np.prod(s[:-1]))[..., None].repeat(s[-1]).flatten()
        attractor = (
            np.arange(s[-1])[:, None].repeat(np.prod(s[:-1]), axis=1).T.flatten()
        )
        dic["sim_label"] = orbit
        dic["attractor_label"] = attractor
        for key, value in raw_dic.items():
            value = np.array(value)
            if key == "attractors":
                Nv = value.shape[-1]
                value = value.reshape(-1, Nv)
                if state_vector_labels is not None:
                    for i in range(Nv):
                        dic[state_vector_labels[i]] = value[:, i]
                else:
                    for i in range(Nv):
                        dic[f"Xa_{i}"] = value[:, i]
            else:
                dic[key] = value.flatten()
        return dic

    def get_subharmonics(self):
        """
        Returns the detected subharmonics.
        Returns:
            jnp.ndarray: Detected subharmonics.
        """
        subharmonics_flags = self.subharmonics_flags
        final_flags = self.final_flags
        targetted_subharmonics = self.targetted_subharmonics
        Nts = targetted_subharmonics.shape[-1]
        subharmonics_flags = subharmonics_flags.reshape(-1, Nts)
        final_flags = final_flags.flatten()
        targetted_subharmonics = targetted_subharmonics.reshape(-1, Nts)
        return convert_subharmonics_flags(
            subharmonics_flags, final_flags, targetted_subharmonics
        )

    def as_polars(self, *args, **kwargs):
        """
        Converts the problem to a Polars DataFrame.
        Returns:
            pl.DataFrame: Polars DataFrame representation of the problem.
        """
        dic = self.as_dict(*args, **kwargs)
        # orbit = dic.pop("orbit_label", None)

        df = pl.DataFrame(dic)
        # df.insert_column(0, pl.Series("sim_label", orbit))
        return df


class OrbitFinder(NamedTuple):
    """
    Container class for finding orbits in a given ODE problem.
    """

    residuals_per_period: jax.Array = np.array(10, int)
    targetted_subharmonics: jax.Array = np.array([1, 3, 5], int)
    max_periods: int = 1000
    controller: PIDController = PIDController(rtol=1e-7, atol=1e-9)
    solver: Tsit5 = Tsit5()

    def get_max_subharmonic(self):
        """
        Returns the maximum subharmonic.
        Returns:
            int: Maximum subharmonic.
        """
        return np.max(self.targetted_subharmonics)

    def get_time_steps_number(self):
        """
        Returns the number of time steps.
        Returns:
            int: Number of time steps.
        """
        max_subharmonic = self.get_max_subharmonic()
        return 2 * max_subharmonic * self.residuals_per_period + 1

    def get_max_shooting_iterations(self):
        """
        Returns the maximum number of shooting iterations.
        Returns:
            int: Maximum number of shooting iterations.
        """
        max_subharmonic = self.get_max_subharmonic()
        return self.max_periods // (2 * max_subharmonic)

    def find_orbits(
        self,
        problem: NamedTuple,
        init_conditions: jax.Array,
        finder_config: OrbitFinderConfig,
    ):
        """
        Find orbits for the given problem.
        Args:
            problem (H46Problem): The problem instance.
            term (ODETerm): The ODE term.
            solver (Tsit5): The solver.
            controller (PIDController): The controller.
            init_conditions (jax.Array): Initial conditions for the system.
            target_frequency (float): Target frequency.
            finder_config (OrbitFinderConfig): Configuration for finding orbits.
        Returns:
            jax.Array: Found orbits.
        """

        def body_fun(carry):
            """
            Body of the while loop that integrates the ODE and checks for convergence.
            """

            start_time = carry.start_time
            end_time = carry.end_time
            init_conditions = carry.init_conditions
            iteration = carry.iteration
            finder_config = carry.finder_config
            convergence_tol = finder_config.convergence_tol
            flag = carry.flag
            # max_shooting_iterations = finder_config.get_max_shooting_iterations()
            (
                Xout,
                res,
                start_time,
                end_time,
            ) = OrbitFinder.integrate_and_check_convergence(
                init_conditions=init_conditions,
                start_time=start_time,
                end_time=end_time,
                steps_number=time_steps_number,
                problem=carry.problem,
                solver=solver,
                controller=controller,
                init_time_step=carry.finder_config.init_time_step,
                # target_frequency=carry.target_frequency,
                targetted_subharmonics=targetted_subharmonics,
                residuals_per_period=residuals_per_period,
            )
            iteration += 1
            flag = jnp.where(
                jnp.any(res <= convergence_tol), 1, flag
            )  # CONVERGENCE ACHIEVED
            flag = jnp.where(
                iteration >= max_shooting_iterations, 2, flag
            )  # EXCEED MAX ITERATIONS
            carry = Carry(
                iteration=iteration,
                flag=flag,
                residuals=res,
                start_time=start_time,
                end_time=end_time,
                init_conditions=Xout,
                problem=carry.problem,
                target_frequency=carry.target_frequency,
                finder_config=carry.finder_config,
            )
            return carry

        def condition_fun(carry):
            """
            Stop condition for the while loop. The loop continues until convergence is achieved
            or the maximum number of iterations is reached.
            """

            flag = carry.flag
            return jnp.all(flag == 0)

        Carry = namedtuple(
            "Carry",
            [
                "iteration",
                "residuals",
                "start_time",
                "end_time",
                "init_conditions",
                "problem",
                "target_frequency",
                "finder_config",
                "flag",
            ],
        )
        solver = self.solver
        controller = self.controller
        residuals_per_period = self.residuals_per_period
        targetted_subharmonics = self.targetted_subharmonics
        target_frequency = finder_config.target_frequency
        max_subharmonic = self.get_max_subharmonic()
        time_steps_number = self.get_time_steps_number()
        max_shooting_iterations = self.get_max_shooting_iterations()
        init_time = finder_config.init_time
        subharmonic_factor = finder_config.subharmonic_factor
        target_period = 1.0 / target_frequency
        duration = 2.0 * max_subharmonic * target_period
        start_time = finder_config.init_time
        end_time = start_time + duration
        iteration = jnp.array(0)
        flag = jnp.array(0)
        residuals = jnp.zeros(targetted_subharmonics.shape)

        carry_in = Carry(
            iteration=iteration,
            flag=flag,
            residuals=residuals,
            start_time=start_time,
            end_time=end_time,
            init_conditions=init_conditions,
            problem=problem,
            target_frequency=target_frequency,
            finder_config=finder_config,
        )
        carry_out = lax.while_loop(
            body_fun=body_fun,
            cond_fun=condition_fun,
            init_val=carry_in,
        )
        final_flag = carry_out.flag
        end_time = carry_out.start_time
        final_conditions = carry_out.init_conditions
        simulated_time = end_time - init_time
        iterations = carry_out.iteration
        simulated_periods = 2 * max_subharmonic * iterations
        residuals = carry_out.residuals
        subharmonic_flag = (
            (final_flag == 1) & (residuals <= residuals.min() * subharmonic_factor)
        ) * 1
        # subharmonic_flag = ((final_flag == 1) & (residuals == residuals.min())) * 1
        # subharmonic_mask = targetted_subharmonics.max() + 1
        detected_subharmonic = jnp.where(
            subharmonic_flag == 1, targetted_subharmonics, jnp.inf
        ).min()
        detected_subharmonic = jnp.where(
            detected_subharmonic == jnp.inf, 0, detected_subharmonic
        ).astype(int)
        subharmonic_residual = jnp.where(
            detected_subharmonic == targetted_subharmonics, residuals, jnp.inf
        ).min()
        min_residual = residuals.min()
        # CALCULATE ATTRACTORS
        term = ODETerm(problem.rhs)
        tg = jnp.arange(1, max_subharmonic + 1)
        t0 = finder_config.init_time
        t1 = t0 + max_subharmonic * target_period
        dt0 = finder_config.init_time_step
        ts = tg * target_period + t0
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            final_conditions,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )

        attractors = sol.ys

        solution = OrbitFinderSolution(
            attractors=attractors,
            detected_subharmonic=detected_subharmonic
            * np.ones(max_subharmonic, dtype=int),
            subharmonic_residual=subharmonic_residual
            * np.ones(max_subharmonic, dtype=float),
            minimum_residual=min_residual * np.ones(max_subharmonic, dtype=float),
            simulated_periods=simulated_periods * np.ones(max_subharmonic, dtype=int),
            simulated_time=simulated_time * np.ones(max_subharmonic, dtype=float),
            converged= (final_flag == 1) * np.ones(max_subharmonic, dtype= bool),
            final_flag=final_flag * np.ones(max_subharmonic, dtype=int),
            simulated_iterations=iterations * np.ones(max_subharmonic, dtype=int),
        )
        return problem, finder_config, init_conditions, solution

    @staticmethod
    def calculate_subharmonic_atomic_residual(pos, args):
        """
        Calculates the shooting residual for a given subharmonic and a given time step.
        """
        norm, offset, X, state_weights = args
        residual = (X[-pos - offset - 1] - X[-pos - 1]) * state_weights
        norm += (residual * residual).sum()
        return norm, offset, X, state_weights

    @staticmethod
    def calculate_subharmonic_residual(
        subharmonic, X, residuals_per_period, state_weights
    ):
        """
        Calculates the shooting residual for a given subharmonic and at all time steps.
        """
        calculate_residuals = OrbitFinder.calculate_subharmonic_atomic_residual
        offset = residuals_per_period * subharmonic
        args_in = (0.0, offset, X, state_weights)
        out = lax.fori_loop(0, offset, calculate_residuals, args_in)
        return out[0] * subharmonic**2 / residuals_per_period

    @staticmethod
    def integrate_and_check_convergence(
        init_conditions,
        start_time,
        end_time,
        steps_number,
        problem,
        solver,
        controller,
        init_time_step,
        targetted_subharmonics,
        residuals_per_period,
    ):
        """
        Integrate the ODE between t0 and t1, and check the convergence of the subharmonic residuals.
        """
        t0 = start_time
        t1 = end_time
        dt0 = init_time_step
        X0 = init_conditions
        term = ODETerm(problem.rhs)
        # Vectorized version of calculate_subharmonic_residual
        batched_calculate_subharmonic_residual = vmap(
            OrbitFinder.calculate_subharmonic_residual, in_axes=(0, None, None, None)
        )
        tg = jnp.arange(steps_number)
        ts = tg * (t1 - t0) / (steps_number - 1) + t0
        ts = ts.at[0].set(t0)  # Ensure the first time step
        ts = ts.at[-1].set(t1)  # Ensure the last time step
        saveat = SaveAt(ts=ts)
        sol = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            X0,
            saveat=saveat,
            args=None,
            stepsize_controller=controller,
            max_steps=None,
        )

        Xs = sol.ys
        state_weights = problem.state_weights()
        res = batched_calculate_subharmonic_residual(
            targetted_subharmonics, Xs, residuals_per_period, state_weights
        )
        X2 = Xs[-1]
        t2 = t1 + (t1 - t0)
        return X2, res, t1, t2


def cluster_points(points, weights, distance_threshold=0.01, method="dbscan"):
    """
    xxx
    """
    if len(points) > 1:
        X = points * weights  # Array CPU
        if method == "agglomerative-clustering":
            db = AgglomerativeClustering(
                distance_threshold=distance_threshold, n_clusters=None, linkage="ward"
            ).fit(X)
        if method == "dbscan":
            db = DBSCAN(eps=distance_threshold, min_samples=1).fit(
                X
            )  # applique la methode fit() de l'objet DBSAN de scikit-learn; fit(X) effectue
            # le clustering DBSCAN a partir des caracteristiques ou de la matrice de distance
        if method == "dbscan_cuml":
            db = cuml.DBSCAN(eps=distance_threshold, min_samples=1).fit(
                X
            )  # CUDA version
        labels = (
            db.labels_
        )  # etiquettes des clusters pour chaque point de l'ensemble de donnees donne a fit(X). etiquette -1 si bruite.
        nclusters = np.unique(labels).size
        centroids = []
        for c in range(nclusters):
            centroids.append(points[labels == c].mean(axis=0))
        centroids = np.array(centroids)
        return nclusters, labels, centroids

    else:
        centroids = points.copy()
        return 1, np.zeros(1, dtype=np.int32), centroids


def detect_attractors_orbits(simulations, ode_params_labels, attractor_state_vec_labels, state_vec_labels):
    """
    Detects attractors and orbits from the simulations data.
    """
    next_attractor_label = 0
    group_labels = ode_params_labels + ["detected_subharmonic", "target_frequency"]
    attractors = {
        k: []
        for k in state_vec_labels
        + ode_params_labels
        + ["detected_subharmonic", "attractor_label", "orbit_label", "target_frequency"]
    }
    sim_attractor_map = {}
    attractor_sim_map = {}

    for keys, group in simulations.group_by(group_labels):
        group = group.sort(["sim_label", "attractor_label"])
        params = dict(zip(group_labels, keys))
        ode_params = {k: params[k] for k in ode_params_labels}
        sh = params["detected_subharmonic"]
        if sh > 0:
            points = group.select(attractor_state_vec_labels).to_numpy()
            problem = H46Problem(**ode_params)
            weights = problem.state_weights()
            nclusters, labels, centroids = cluster_points(
                points, weights, distance_threshold=0.01, method="dbscan"
            )
            labels += next_attractor_label
            attractor_labels = np.arange(nclusters) + next_attractor_label
            
            for i in range(nclusters):
                for k in ode_params_labels:
                    attractors[k].append(params[k])
                for j, k in enumerate(state_vec_labels):
                    attractors[k].append(centroids[i, j])
                attractors["detected_subharmonic"].append(sh)
                attractors["attractor_label"].append(attractor_labels[i])
                attractors["target_frequency"].append(params["target_frequency"])
            next_attractor_label = next_attractor_label + nclusters
            for k, v in zip(group["sim_label"][::sh].to_numpy(), labels.reshape(-1,sh)):
                while v[0] != v.min():
                    v = np.roll(v, -1)
                v = tuple(v)
                sim_attractor_map[k] = v
                if v not in attractor_sim_map:
                    attractor_sim_map[v] = []
                attractor_sim_map[v].append(k)

    orbits_attractors_list = list(attractor_sim_map.keys())
    orbit_attractor_map = {}
    attractor_orbit_map = {}
    for oid in range(len(orbits_attractors_list)):
        orbit_attractor_map[oid] = orbits_attractors_list[oid]
    for k, v in orbit_attractor_map.items():
        for vv in v:
            attractor_orbit_map[vv] = k
    for aid in attractors["attractor_label"]:
        attractors["orbit_label"].append(attractor_orbit_map[aid])  
    sim_orbit = []
    for k, v in sim_attractor_map.items():
        sim_orbit.append((k, attractor_orbit_map[v[0]]))
    sim_orbit = np.array(sim_orbit)
    sim_orbit = pl.DataFrame({"sim_label": sim_orbit[:, 0], "orbit_label": sim_orbit[:, 1]})
    attractors = pl.DataFrame(attractors)
    return attractors, sim_orbit