"""Tests for MarkovProcess class."""

from pathlib import Path

import numpy as np
import pytest

from rlcore.tabular import MarkovProcess
from rlcore.tabular.exceptions import InvalidTransitionMatrixError
from rlcore.tabular.visualization import plot_markov_process

OUTPUT_DIR = Path(__file__).parent.parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Fixtures ---


@pytest.fixture
def weather_mp():
    """Create a weather Markov process."""
    states = ["sunny", "rainy"]
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    return MarkovProcess(states, P)


@pytest.fixture
def student_mp():
    """Create a student Markov process with absorbing states."""
    states = ["Class1", "Class2", "Class3", "Pass", "Sleep"]
    P = np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 0.5],  # Class1
            [0.0, 0.0, 0.8, 0.0, 0.2],  # Class2
            [0.0, 0.0, 0.0, 0.6, 0.4],  # Class3
            [0.0, 0.0, 0.0, 1.0, 0.0],  # Pass (absorbing)
            [0.0, 0.0, 0.0, 0.0, 1.0],  # Sleep (absorbing)
        ]
    )
    return MarkovProcess(states, P)


# --- Weather Markov Process Tests ---


def test_weather_basic_properties(weather_mp):
    """Test basic properties of the weather Markov process."""
    assert weather_mp.num_states == 2
    assert weather_mp.states == ("sunny", "rainy")


def test_weather_transition_matrix(weather_mp):
    """Test transition matrix is correctly stored."""
    expected_P = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert np.allclose(weather_mp.transition_matrix, expected_P)


def test_weather_trajectory_properties(weather_mp):
    """Test trajectory generation."""
    trajectory = weather_mp.episode(
        "sunny", num_steps=10, rng=np.random.default_rng(42)
    )

    assert len(trajectory) == 11  # initial state + 10 steps
    assert all(state in ["sunny", "rainy"] for state in trajectory)
    assert trajectory[0] == "sunny"


def test_weather_state_distribution(weather_mp):
    """Test state distribution computation."""
    initial_dist = np.array([1.0, 0.0])  # Start in sunny
    dist_after_5 = weather_mp.compute_state_distribution(initial_dist, num_steps=5)

    assert np.isclose(np.sum(dist_after_5), 1.0)
    assert np.all(dist_after_5 >= 0)
    assert np.all(dist_after_5 <= 1)


def test_weather_stationary_distribution_eigenvector(weather_mp):
    """Test stationary distribution via eigenvector method."""
    stationary = weather_mp.compute_stationary_distribution(method="analytical")

    assert stationary is not None
    assert np.isclose(np.sum(stationary), 1.0)
    assert np.all(stationary >= 0)

    # Check π = πP
    stationary_result = stationary @ weather_mp.transition_matrix
    assert np.allclose(stationary, stationary_result)

    # For this matrix, stationary distribution should be [2/3, 1/3]
    expected_stationary = np.array([2 / 3, 1 / 3])
    assert np.allclose(stationary, expected_stationary, atol=1e-6)


def test_weather_stationary_distribution_iterative(weather_mp):
    """Test stationary distribution via iterative method."""
    stationary = weather_mp.compute_stationary_distribution(method="iterative")

    assert stationary is not None
    expected_stationary = np.array([2 / 3, 1 / 3])
    assert np.allclose(stationary, expected_stationary, atol=1e-6)


def test_weather_visualization(weather_mp):
    """Test that visualization runs without error."""
    info = plot_markov_process(
        weather_mp, filename=str(OUTPUT_DIR / "weather_mp.graphml")
    )
    assert info.num_nodes == 2
    assert info.num_edges > 0


# --- Student Markov Process Tests ---


def test_student_basic_properties(student_mp):
    """Test basic properties."""
    assert student_mp.num_states == 5
    assert student_mp.states == ("Class1", "Class2", "Class3", "Pass", "Sleep")


def test_student_trajectory_reaches_absorbing(student_mp):
    """Test that trajectory eventually reaches an absorbing state."""
    trajectory = student_mp.episode(
        "Class1", num_steps=20, rng=np.random.default_rng(123)
    )

    assert len(trajectory) == 21
    assert trajectory[0] == "Class1"
    assert trajectory[-1] in ["Pass", "Sleep"]


def test_student_state_distribution(student_mp):
    """Test state distribution computation."""
    initial_dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    dist_after_10 = student_mp.compute_state_distribution(initial_dist, num_steps=10)

    assert np.isclose(np.sum(dist_after_10), 1.0)
    assert np.all(dist_after_10 >= 0)
    # Should have positive probability in absorbing states
    assert dist_after_10[3] > 0  # Pass
    assert dist_after_10[4] > 0  # Sleep


def test_student_stationary_distribution_with_absorbing(student_mp):
    """Test stationary distribution with absorbing states."""
    stationary = student_mp.compute_stationary_distribution()

    if stationary is not None:
        assert np.isclose(np.sum(stationary), 1.0)
        assert np.all(stationary >= 0)
        # Check π = πP
        stationary_result = stationary @ student_mp.transition_matrix
        assert np.allclose(stationary, stationary_result)


def test_student_visualization(student_mp):
    """Test that visualization runs without error."""
    info = plot_markov_process(
        student_mp, filename=str(OUTPUT_DIR / "student_mp.graphml")
    )
    assert info.num_nodes == 5


# --- Validation Tests ---


def test_negative_num_steps_raises():
    """Test that negative num_steps raises ValueError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    mp = MarkovProcess(states, P)

    with pytest.raises(ValueError, match="num_steps must be non-negative"):
        mp.episode("A", num_steps=-1)


def test_duplicate_states_raises():
    """Test that duplicate states raise ValueError."""
    states = ["A", "A"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(ValueError, match="unique"):
        MarkovProcess(states, P)


def test_invalid_transition_matrix_shape():
    """Test that invalid matrix shape raises error."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])

    with pytest.raises(InvalidTransitionMatrixError):
        MarkovProcess(states, P)


def test_invalid_probability_distribution():
    """Test that rows not summing to 1 raise error."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.4], [0.5, 0.5]])  # First row sums to 0.9

    with pytest.raises(InvalidTransitionMatrixError):
        MarkovProcess(states, P)
