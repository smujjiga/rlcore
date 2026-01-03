"""Markov Process implementation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rlcore.tabular.exceptions import InvalidStateError
from rlcore.tabular.type_aliases import Probability
from rlcore.tabular.validation import (
    validate_probability_distribution,
    validate_transition_matrix,
)


class MarkovProcess[State]:
    """Markov Process defined by (S, P).

    Represents a memoryless stochastic process where the probability of
    transitioning to the next state depends only on the current state.

    Attributes:
        states: Sequence of all possible states.
        transition_matrix: Square matrix P where P[i,j] = p(s'=j | s=i).
        num_states: Number of states in the process.
    """

    def __init__(
        self,
        states: Sequence[State],
        transition_matrix: NDArray[np.float64],
    ) -> None:
        self._states = tuple(states)
        self._num_states = len(self._states)

        self._state_to_index = {s: i for i, s in enumerate(self._states)}
        self._index_to_state = dict(enumerate(self._states))

        if len(self._state_to_index) != self._num_states:
            raise ValueError("States must be unique and hashable")

        validate_transition_matrix(transition_matrix, self._num_states)
        self._transition_matrix = transition_matrix.copy()

    @property
    def states(self) -> tuple[State, ...]:
        return self._states

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        return self._transition_matrix.copy()

    @property
    def num_states(self) -> int:
        return self._num_states

    def get_state_index(self, state: State) -> int:
        if state not in self._state_to_index:
            raise InvalidStateError(f"State {state} not in process")
        return self._state_to_index[state]

    def get_state_from_index(self, index: int) -> State:
        if index not in self._index_to_state:
            raise InvalidStateError(f"Index {index} out of range")
        return self._index_to_state[index]

    def get_transition_probability(
        self, from_state: State, to_state: State
    ) -> Probability:
        from_idx = self.get_state_index(from_state)
        to_idx = self.get_state_index(to_state)
        return float(self._transition_matrix[from_idx, to_idx])

    def sample_next_state(
        self, current_state: State, rng: np.random.Generator | None = None
    ) -> State:
        if rng is None:
            rng = np.random.default_rng()

        current_idx = self.get_state_index(current_state)
        transition_probs = self._transition_matrix[current_idx]
        next_idx = rng.choice(self._num_states, p=transition_probs)
        return self.get_state_from_index(next_idx)

    def episode(
        self,
        initial_state: State,
        num_steps: int,
        rng: np.random.Generator | None = None,
    ) -> list[State]:
        trajectory = [initial_state]
        current_state = initial_state

        for _ in range(num_steps):
            next_state = self.sample_next_state(current_state, rng)
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def compute_state_distribution(
        self,
        initial_distribution: NDArray[np.float64],
        num_steps: int,
    ) -> NDArray[np.float64]:
        validate_probability_distribution(initial_distribution)
        transition_power = np.linalg.matrix_power(self._transition_matrix, num_steps)
        # d_t = d_0 @ P^t
        return initial_distribution @ transition_power

    def compute_stationary_distribution(
        self,
        method: str = "analytical",
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ) -> NDArray[np.float64] | None:
        """Compute stationary distribution if it exists.

        Finds π such that π = πP (left eigenvector with eigenvalue 1).

        For method='analytical': Computes left eigenvector of P with eigenvalue 1.

        For method='iterative': Repeatedly applies π_{k+1} = π_k @ P until
        convergence. Equivalent to computing π_0 @ P^N for large N.
        """
        if method == "analytical":
            return self._compute_stationary_analytical(tolerance)
        elif method == "iterative":
            return self._compute_stationary_iterative(tolerance, max_iterations)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'analytical' or 'iterative'."
            )

    def _compute_stationary_analytical(
        self,
        tolerance: float,
    ) -> NDArray[np.float64] | None:
        """Compute stationary distribution via eigendecomposition."""
        # Transpose to get right eigenvectors of P^T, then transpose back
        eigenvalues, eigenvectors = np.linalg.eig(self._transition_matrix.T)

        # Find eigenvector corresponding to eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))

        # Check if eigenvalue is actually close to 1
        if not np.isclose(eigenvalues[idx], 1.0, atol=tolerance):
            return None  # No stationary distribution found

        # Get the eigenvector and ensure it's real
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1 (handle negative values by taking absolute)
        stationary = np.abs(stationary)
        stationary = stationary / np.sum(stationary)

        return stationary

    def _compute_stationary_iterative(
        self,
        tolerance: float,
        max_iterations: int,
    ) -> NDArray[np.float64] | None:
        """Compute stationary distribution via power iteration.

        Iteratively computes π_{k+1} = π_k @ P until convergence.
        This is equivalent to computing π_0 @ P^N for large N.
        """
        # Start with uniform distribution
        distribution = np.ones(self._num_states) / self._num_states

        for _ in range(max_iterations):
            # π_{k+1} = π_k @ P
            new_distribution = distribution @ self._transition_matrix

            if np.allclose(distribution, new_distribution, atol=tolerance):
                return new_distribution

            distribution = new_distribution

        # Return current estimate even if not fully converged
        return distribution

    def plot(self, **kwargs):
        from rlcore.tabular.visualization import plot_transition_graph

        return plot_transition_graph(self, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        states_preview = f"{self._states[:3]}{'...' if self._num_states > 3 else ''}"
        return f"MarkovProcess(num_states={self._num_states}, states={states_preview})"
