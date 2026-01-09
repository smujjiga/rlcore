"""Tests for MarkovRewardProcess class."""

from pathlib import Path

import numpy as np
import pytest

from rlcore.tabular import MarkovRewardProcess, SingularMatrixError
from rlcore.tabular.exceptions import InvalidDiscountFactorError
from rlcore.tabular.visualization import (
    plot_reward_transition_graph,
    plot_value_function,
)

OUTPUT_DIR = Path(__file__).parent.parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Fixtures ---


@pytest.fixture
def weather_mrp():
    """Create a weather MRP."""
    states = ["sunny", "rainy"]
    P = np.array([[0.8, 0.2], [0.4, 0.6]])
    R = np.array([1.0, -1.0])  # +1 for sunny, -1 for rainy
    gamma = 0.9
    return MarkovRewardProcess(states, P, R, gamma)


@pytest.fixture
def student_mrp():
    """Create a student MRP with absorbing states."""
    states = ["Class1", "Class2", "Class3", "Pass", "Sleep"]
    P = np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.8, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.6, 0.4],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    R = np.array([-2.0, -2.0, -2.0, 10.0, 0.0])
    gamma = 0.99
    return MarkovRewardProcess(states, P, R, gamma)


# --- Weather MRP Tests ---


def test_weather_basic_properties(weather_mrp):
    """Test basic properties."""
    assert weather_mrp.num_states == 2
    assert weather_mrp.states == ("sunny", "rainy")
    assert weather_mrp.discount_factor == 0.9


def test_weather_reward_function(weather_mrp):
    """Test reward function."""
    assert np.allclose(weather_mrp.reward_function, np.array([1.0, -1.0]))
    assert weather_mrp.get_reward("sunny") == 1.0
    assert weather_mrp.get_reward("rainy") == -1.0


def test_weather_value_function_analytical(weather_mrp):
    """Test analytical value function computation."""
    v = weather_mrp.compute_value_function(method="analytical")

    assert v.shape == (2,)
    # Sunny state should have higher value
    assert v[0] > v[1]

    # Check Bellman equation: V = R + γPV
    R = weather_mrp.reward_function
    P = weather_mrp.transition_matrix
    gamma = weather_mrp.discount_factor
    bellman_check = R + gamma * (P @ v)
    assert np.allclose(v, bellman_check, atol=1e-6)


def test_weather_value_function_iterative(weather_mrp):
    """Test iterative value function computation."""
    v_analytical = weather_mrp.compute_value_function(method="analytical")
    v_iterative = weather_mrp.compute_value_function(method="iterative")

    assert np.allclose(v_analytical, v_iterative, atol=1e-3)


def test_weather_compute_return(weather_mrp):
    """Test discounted return computation."""
    rewards = [1.0, 1.0, -1.0, 1.0, -1.0]

    discounted_return = weather_mrp.compute_return(rewards)

    # G = 1 + 0.9*1 + 0.81*(-1) + 0.729*1 + 0.6561*(-1)
    expected = 1.0 + 0.9 * 1.0 + 0.81 * (-1.0) + 0.729 * 1.0 + 0.6561 * (-1.0)
    assert np.isclose(discounted_return, expected)


def test_weather_episode(weather_mrp):
    """Test episode generation with rewards."""
    trajectory, rewards = weather_mrp.episode(
        "sunny", num_steps=5, rng=np.random.default_rng(42)
    )

    assert len(trajectory) == 6  # initial + 5 steps
    assert len(rewards) == 6
    assert trajectory[0] == "sunny"
    # Check rewards match states
    for state, reward in zip(trajectory, rewards, strict=True):
        assert reward == weather_mrp.get_reward(state)


def test_weather_visualization_value_function(weather_mrp):
    """Test value function visualization."""
    fig = plot_value_function(weather_mrp, method="analytical")
    fig.savefig(OUTPUT_DIR / "weather_mrp_value_function.png")
    assert fig is not None


def test_weather_visualization_reward_graph(weather_mrp):
    """Test reward graph visualization."""
    info = plot_reward_transition_graph(
        weather_mrp, filename=str(OUTPUT_DIR / "weather_mrp.graphml")
    )
    assert info.num_nodes == 2
    assert info.num_edges > 0


# --- Student MRP Tests ---


def test_student_basic_properties(student_mrp):
    """Test basic properties."""
    assert student_mrp.num_states == 5
    assert student_mrp.discount_factor == 0.99


def test_student_value_function(student_mrp):
    """Test value function computation."""
    v = student_mrp.compute_value_function(method="analytical")

    assert v.shape == (5,)

    # Absorbing states: V(s) = R(s) / (1 - γ) for γ < 1
    gamma = student_mrp.discount_factor
    assert np.isclose(v[3], 10.0 / (1 - gamma), atol=1e-2)  # Pass
    assert np.isclose(v[4], 0.0, atol=1e-6)  # Sleep

    # Check Bellman equation
    R = student_mrp.reward_function
    P = student_mrp.transition_matrix
    bellman_check = R + gamma * (P @ v)
    assert np.allclose(v, bellman_check, atol=1e-5)


def test_student_value_function_iterative_converges(student_mrp):
    """Test that iterative method converges close to analytical."""
    v_analytical = student_mrp.compute_value_function(method="analytical")
    v_iterative = student_mrp.compute_value_function(
        method="iterative", max_iterations=5000
    )

    # Iterative converges slowly with high gamma
    assert np.allclose(v_analytical, v_iterative, atol=1.0)


def test_student_class_states_have_positive_value(student_mrp):
    """Test that class states have positive expected value."""
    v = student_mrp.compute_value_function()

    assert v[0] > 0  # Class1
    assert v[1] > 0  # Class2
    assert v[2] > 0  # Class3


def test_student_visualization(student_mrp):
    """Test visualizations."""
    fig = plot_value_function(student_mrp, color="forestgreen")
    fig.savefig(OUTPUT_DIR / "student_mrp_value_function.png")

    info = plot_reward_transition_graph(
        student_mrp, filename=str(OUTPUT_DIR / "student_mrp.graphml")
    )
    assert info.num_nodes == 5


# --- Validation Tests ---


def test_invalid_reward_shape_raises():
    """Test that invalid reward shape raises ValueError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, 2.0, 3.0])  # Wrong shape

    with pytest.raises(ValueError):
        MarkovRewardProcess(states, P, R, 0.9)


def test_invalid_discount_factor_raises():
    """Test that invalid discount factor raises error."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, 2.0])

    with pytest.raises(InvalidDiscountFactorError):
        MarkovRewardProcess(states, P, R, 1.5)

    with pytest.raises(InvalidDiscountFactorError):
        MarkovRewardProcess(states, P, R, -0.1)


def test_start_index_out_of_bounds_raises():
    """Test that out-of-bounds start_index raises ValueError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, 2.0])
    mrp = MarkovRewardProcess(states, P, R, 0.9)

    rewards = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="start_index"):
        mrp.compute_return(rewards, start_index=-1)

    with pytest.raises(ValueError, match="start_index"):
        mrp.compute_return(rewards, start_index=10)


def test_nan_in_rewards_raises():
    """Test that NaN in rewards raises ValueError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, np.nan])

    with pytest.raises(ValueError, match="NaN"):
        MarkovRewardProcess(states, P, R, 0.9)


def test_inf_in_rewards_raises():
    """Test that Inf in rewards raises ValueError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, np.inf])

    with pytest.raises(ValueError, match="Inf"):
        MarkovRewardProcess(states, P, R, 0.9)


# --- Singular Matrix Handling Tests ---


def test_gamma_one_raises_singular_matrix_error():
    """Test that gamma=1.0 with regular chain raises SingularMatrixError."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, 2.0])
    mrp = MarkovRewardProcess(states, P, R, 1.0)

    with pytest.raises(SingularMatrixError):
        mrp.compute_value_function(method="analytical")


def test_gamma_one_iterative_still_works():
    """Test that iterative method works even with gamma=1.0."""
    states = ["A", "B"]
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([1.0, 2.0])
    mrp = MarkovRewardProcess(states, P, R, 1.0)

    # Iterative should not raise, but may not converge
    v = mrp.compute_value_function(method="iterative", max_iterations=100)
    # Result may be None if not converged, or a value if it did
    # Just check it doesn't crash
    assert v is None or v.shape == (2,)
