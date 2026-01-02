"""Tests for MarkovProcess class."""

import numpy as np

from rlcore.tabular import MarkovProcess
from rlcore.tabular.visualization import plot_markov_process


# Example 1: Simple Weather Markov Process
def test_simple_weather():
    print("=" * 60)
    print("Example 1: Weather Markov Process")
    print("=" * 60)

    states = ["sunny", "rainy"]
    P = np.array([[0.8, 0.2], [0.4, 0.6]])  # noqa: N806
    mp = MarkovProcess(states, P)
    print(f"Created: {mp}")
    print(f"Number of states: {mp.num_states}")
    print(f"States: {mp.states}")

    # Sample a trajectory
    trajectory = mp.episode("sunny", num_steps=10, rng=np.random.default_rng(42))
    print(f"\nTrajectory (10 steps from 'sunny'): {trajectory}")

    # Compute state distribution after 5 steps
    initial_dist = np.array([1.0, 0.0])  # Start in sunny
    dist_after_5 = mp.compute_state_distribution(initial_dist, num_steps=5)
    print(
        f"\nDistribution after 5 steps: {dict(zip(states, dist_after_5, strict=True))}"
    )

    # Find stationary distribution using eigenvector method (default)
    stationary_eigen = mp.compute_stationary_distribution()
    if stationary_eigen is not None:
        print(
            f"Stationary distribution (eigenvector): "
            f"{dict(zip(states, stationary_eigen, strict=True))}"
        )

    # Visualization - save PNG
    plot_markov_process(mp, "weather_markov_process.png")

    # Test cases - Basic properties
    assert mp.num_states == 2
    assert mp.states == ("sunny", "rainy")
    assert np.allclose(mp.transition_matrix, P)

    # Test trajectory properties
    assert len(trajectory) == 11  # initial state + 10 steps
    assert all(state in states for state in trajectory)  # all states are valid
    assert trajectory[0] == "sunny"  # starts from sunny

    # Test state distribution properties
    assert np.isclose(np.sum(dist_after_5), 1.0)  # probabilities sum to 1
    assert np.all(dist_after_5 >= 0)  # all probabilities non-negative
    assert np.all(dist_after_5 <= 1)  # all probabilities <= 1

    # Test stationary distribution properties
    assert stationary_eigen is not None
    assert np.isclose(np.sum(stationary_eigen), 1.0)  # sums to 1
    assert np.all(stationary_eigen >= 0)  # all non-negative
    # Check that stationary distribution satisfies π = πP
    stationary_result = stationary_eigen @ mp.transition_matrix
    assert np.allclose(stationary_eigen, stationary_result)
    # For this specific matrix, stationary distribution should be [2/3, 1/3]
    expected_stationary = np.array([2/3, 1/3])
    assert np.allclose(stationary_eigen, expected_stationary, atol=1e-6)


# Example 2: Student Markov Process
def test_student():
    print("=" * 60)
    print("Example 2: Student Markov Process")
    print("=" * 60)

    states = ["Class1", "Class2", "Class3", "Pass", "Sleep"]
    P = np.array(  # noqa: N806
        [
            [0.0, 0.5, 0.0, 0.0, 0.5],  # Class1
            [0.0, 0.0, 0.8, 0.0, 0.2],  # Class2
            [0.0, 0.0, 0.0, 0.6, 0.4],  # Class3
            [0.0, 0.0, 0.0, 1.0, 0.0],  # Pass (absorbing)
            [0.0, 0.0, 0.0, 0.0, 1.0],  # Sleep (absorbing)
        ]
    )

    mp = MarkovProcess(states, P)
    print(f"Created: {mp}")
    print(f"Number of states: {mp.num_states}")
    print(f"States: {mp.states}")

    # Sample a trajectory from Class1
    trajectory = mp.episode("Class1", num_steps=20, rng=np.random.default_rng(123))
    print(f"\nTrajectory (20 steps from 'Class1'): {trajectory}")

    # Compute state distribution after 10 steps from Class1
    initial_dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Start in Class1
    dist_after_10 = mp.compute_state_distribution(initial_dist, num_steps=10)
    state_dist_dict = dict(zip(states, dist_after_10, strict=True))
    print(f"\nDistribution after 10 steps: {state_dist_dict}")

    # Find stationary distribution
    stationary = mp.compute_stationary_distribution()
    if stationary is not None:
        print(
            f"Stationary distribution: "
            f"{dict(zip(states, stationary, strict=True))}"
        )

    # Visualization - save PNG
    plot_markov_process(mp, "student_markov_process.png")

    # Test cases - Basic properties
    assert mp.num_states == 5
    assert mp.states == ("Class1", "Class2", "Class3", "Pass", "Sleep")
    assert np.allclose(mp.transition_matrix, P)

    # Test trajectory properties
    assert len(trajectory) == 21  # initial state + 20 steps
    assert all(state in states for state in trajectory)  # all states are valid
    assert trajectory[0] == "Class1"  # starts from Class1
    # Check that trajectory eventually reaches an absorbing state
    assert trajectory[-1] in ["Pass", "Sleep"]

    # Test state distribution properties
    assert np.isclose(np.sum(dist_after_10), 1.0)  # probabilities sum to 1
    assert np.all(dist_after_10 >= 0)  # all probabilities non-negative
    assert np.all(dist_after_10 <= 1)  # all probabilities <= 1
    # After many steps, should have non-zero probability only in absorbing states
    assert dist_after_10[0] >= 0  # Class1
    assert dist_after_10[1] >= 0  # Class2
    assert dist_after_10[2] >= 0  # Class3
    assert dist_after_10[3] > 0  # Pass (should have positive probability)
    assert dist_after_10[4] > 0  # Sleep (should have positive probability)

    # Test stationary distribution properties
    # For Markov chains with absorbing states, stationary distribution may not be unique
    if stationary is not None:
        assert np.isclose(np.sum(stationary), 1.0)  # sums to 1
        assert np.all(stationary >= 0)  # all non-negative
        # Check that stationary distribution satisfies π = πP
        stationary_result = stationary @ mp.transition_matrix
        assert np.allclose(stationary, stationary_result)
