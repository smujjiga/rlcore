"""Validation utilities for tabular RL."""

import numpy as np
from numpy.typing import NDArray

from rlcore.tabular.exceptions import (
    InvalidDiscountFactorError,
    InvalidProbabilityDistributionError,
    InvalidTransitionMatrixError,
)


def validate_probability_distribution(
    probabilities: NDArray[np.float64],
    tolerance: float = 1e-6,
    axis: int | None = None,
) -> None:
    """Validate that probabilities sum to 1.0.

    Args:
        probabilities: Array of probability values.
        tolerance: Acceptable deviation from 1.0.
        axis: Axis along which to sum. If None, checks entire array sums to 1.0.

    Raises:
        InvalidProbabilityDistributionError: If probabilities don't sum to 1.0
            or contain negative values.
    """
    # Check for negative probabilities
    if np.any(probabilities < 0):
        raise InvalidProbabilityDistributionError("Probabilities must be non-negative")

    # Check sum
    prob_sum = np.sum(probabilities, axis=axis)

    if axis is None:
        # Scalar sum
        if not np.isclose(prob_sum, 1.0, atol=tolerance):
            raise InvalidProbabilityDistributionError(
                f"Probabilities must sum to 1.0, got {prob_sum}"
            )
    else:
        # Array of sums
        if not np.allclose(prob_sum, 1.0, atol=tolerance):
            invalid_indices = np.where(~np.isclose(prob_sum, 1.0, atol=tolerance))[0]
            invalid_sums = prob_sum[invalid_indices]
            raise InvalidProbabilityDistributionError(
                f"Probabilities must sum to 1.0 along axis {axis}. "
                f"Invalid at indices: {invalid_indices}, sums: {invalid_sums}"
            )


def validate_transition_matrix(
    transition_matrix: NDArray[np.float64], num_states: int
) -> None:
    """Validate transition matrix shape and probabilities.

    Args:
        transition_matrix: Square matrix of transition probabilities.
        num_states: Expected number of states.

    Raises:
        InvalidTransitionMatrixError: If matrix is invalid.
    """
    # Check shape
    if transition_matrix.shape != (num_states, num_states):
        raise InvalidTransitionMatrixError(
            f"Transition matrix must have shape ({num_states}, {num_states}), "
            f"got {transition_matrix.shape}"
        )

    # Check if each row is a valid probability distribution
    try:
        validate_probability_distribution(transition_matrix, axis=1)
    except InvalidProbabilityDistributionError as e:
        raise InvalidTransitionMatrixError(
            f"Each row of transition matrix must be a valid probability "
            f"distribution: {e}"
        ) from e


def validate_discount_factor(gamma: float) -> None:
    """Validate discount factor is in [0, 1].

    Args:
        gamma: Discount factor.

    Raises:
        InvalidDiscountFactorError: If gamma not in [0, 1].
    """
    if not 0 <= gamma <= 1:
        raise InvalidDiscountFactorError(
            f"Discount factor must be in [0, 1], got {gamma}"
        )
