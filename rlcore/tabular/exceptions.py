"""Custom exceptions for tabular RL modules."""


class InvalidProbabilityDistributionError(ValueError):
    """Raised when probability distribution doesn't sum to 1.0."""

    pass


class InvalidStateError(ValueError):
    """Raised when referencing a non-existent state."""

    pass


class InvalidTransitionMatrixError(ValueError):
    """Raised when transition matrix has invalid dimensions or values."""

    pass


class InvalidDiscountFactorError(ValueError):
    """Raised when discount factor is not in [0, 1]."""

    pass
