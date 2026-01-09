"""Markov Reward Process implementation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rlcore.tabular.exceptions import SingularMatrixError
from rlcore.tabular.markov_process import MarkovProcess
from rlcore.tabular.type_aliases import DiscountFactor
from rlcore.tabular.validation import validate_discount_factor, validate_reward_function


class MarkovRewardProcess[State](MarkovProcess[State]):
    """Markov Reward Process defined by (S, P, R, γ).

    Represent MRP by extending MP by adding rewards and discount factor.

    Attributes:
        states: Sequence of all possible states.
        transition_matrix: Square matrix P where P[i,j] = p(s'=j | s=i).
        reward_function: Vector R where R[i] is reward for being in state i.
        discount_factor: γ ∈ [0,1] for discounting future rewards.
        num_states: Number of states in the process.
    """

    def __init__(
        self,
        states: Sequence[State],
        transition_matrix: NDArray[np.float64],
        reward_function: NDArray[np.float64],
        discount_factor: DiscountFactor,
    ) -> None:
        # Initialize parent MarkovProcess
        super().__init__(states, transition_matrix)

        # Validate and store reward function
        validate_reward_function(reward_function, self._num_states)
        self._reward_function = reward_function.copy()

        # Validate and store discount factor
        validate_discount_factor(discount_factor)
        self._discount_factor = discount_factor

    @property
    def reward_function(self) -> NDArray[np.float64]:
        return self._reward_function.copy()

    @property
    def discount_factor(self) -> DiscountFactor:
        return self._discount_factor

    def get_reward(self, state: State) -> float:
        state_idx = self.get_state_index(state)
        return float(self._reward_function[state_idx])

    def episode(
        self,
        initial_state: State,
        num_steps: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[State], list[float]]:
        trajectory = super().episode(initial_state, num_steps, rng)
        rewards = [self.get_reward(state) for state in trajectory]
        return trajectory, rewards

    def compute_return(
        self,
        rewards: Sequence[float],
        start_index: int = 0,
    ) -> float:
        """Compute discounted return from a reward sequence.

        G_t = R_t + γR_{t+1} + γ²R_{t+2} + ...

        Args:
            rewards: Sequence of reward values.
            start_index: Index to start computing return from.

        Returns:
            Discounted return starting from start_index.

        Raises:
            ValueError: If start_index is out of bounds.
        """
        if start_index < 0 or start_index > len(rewards):
            raise ValueError(
                f"start_index must be in [0, {len(rewards)}], got {start_index}"
            )
        discounted_return = 0.0
        for i, reward in enumerate(rewards[start_index:]):
            discounted_return += (self._discount_factor**i) * reward
        return discounted_return

    def compute_value_function(
        self,
        method: str = "analytical",
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ) -> NDArray[np.float64]:
        """Compute state value function V(s).

        For method='analytical': Solves V = R + γPV directly via matrix inversion.
        V = (I - γP)^{-1} R

        For method='iterative': Uses iterative Bellman evaluation.
        Repeatedly applies V_{k+1} = R + γPV_k until convergence.
        """
        if method == "analytical":
            return self._compute_value_function_analytical()
        elif method == "iterative":
            return self._compute_value_function_iterative(tolerance, max_iterations)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'analytical' or 'iterative'."
            )

    def _compute_value_function_analytical(self) -> NDArray[np.float64]:
        """Compute value function via matrix inversion.

        Solves V = (I - γP)^{-1} R directly.

        Raises:
            SingularMatrixError: If (I - γP) is singular (e.g., when γ=1.0
                and P has eigenvalue 1).
        """
        identity = np.eye(self._num_states)
        matrix = identity - self._discount_factor * self._transition_matrix
        # NOTE: Solves Ax = b for x
        # Using np.linalg.solve(A, b) is numerically more stable and efficient than
        # computing np.linalg.inv(A) @ b. It uses LU decomposition internally rather
        # than explicitly computing the inverse.
        try:
            value_function = np.linalg.solve(matrix, self._reward_function)
        except np.linalg.LinAlgError as e:
            raise SingularMatrixError(
                f"Cannot compute value function: matrix (I - γP) is singular. "
                f"This often occurs when γ=1.0. Try using method='iterative' or "
                f"a discount factor < 1.0. Original error: {e}"
            ) from e
        return value_function

    def _compute_value_function_iterative(
        self,
        tolerance: float,
        max_iterations: int,
    ) -> NDArray[np.float64]:
        """Compute value function via iterative Bellman evaluation.

        Repeatedly applies V_{k+1} = R + γPV_k until convergence.
        """
        value = np.zeros(self._num_states)

        for _ in range(max_iterations):
            # V_{k+1} = R + γPV_k
            new_value = self._reward_function + self._discount_factor * (
                self._transition_matrix @ value
            )

            if np.allclose(value, new_value, atol=tolerance):
                return new_value

            value = new_value

        # Return current estimate even if not fully converged
        return value

    def __repr__(self) -> str:
        """String representation."""
        states_preview = f"{self._states[:3]}{'...' if self._num_states > 3 else ''}"
        return (
            f"MarkovRewardProcess(num_states={self._num_states}, "
            f"gamma={self._discount_factor:.2f}, "
            f"states={states_preview})"
        )
