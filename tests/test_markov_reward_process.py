"""Tests for MarkovRewardProcess class."""

from pathlib import Path

import numpy as np

from rlcore.tabular import MarkovRewardProcess
from rlcore.tabular.visualization import (
    plot_reward_transition_graph,
    plot_value_function,
)

OUTPUT_DIR = Path(__file__).parent.parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# Example 1: Simple Weather MRP with rewards
def test_weather_mrp():
    print("=" * 60)
    print("Example 1: Weather Markov Reward Process")
    print("=" * 60)

    states = ["sunny", "rainy"]
    P = np.array([[0.8, 0.2], [0.4, 0.6]])  # noqa: N806
    R = np.array([1.0, -1.0])  # noqa: N806  # +1 for sunny, -1 for rainy
    gamma = 0.9

    mrp = MarkovRewardProcess(states, P, R, gamma)
    print(f"Created: {mrp}")
    print(f"Number of states: {mrp.num_states}")
    print(f"States: {mrp.states}")
    print(f"Discount factor: {mrp.discount_factor}")
    print(f"Reward function: {mrp.reward_function}")

    # Compute value function using both methods
    v_analytical = mrp.compute_value_function(method="analytical")
    v_iterative = mrp.compute_value_function(method="iterative")
    v_dict_a = dict(zip(states, v_analytical, strict=True))
    v_dict_i = dict(zip(states, v_iterative, strict=True))
    print(f"\nValue function (analytical): {v_dict_a}")
    print(f"Value function (iterative): {v_dict_i}")

    # Test return computation
    rewards = [1.0, 1.0, -1.0, 1.0, -1.0]
    discounted_return = mrp.compute_return(rewards)
    print(f"\nRewards sequence: {rewards}")
    print(f"Discounted return (γ={gamma}): {discounted_return:.4f}")

    # Visualization - save to test_output/
    fig = plot_value_function(mrp, method="analytical")
    fig.savefig(OUTPUT_DIR / "weather_mrp_value_function.png")
    print(f"\nSaved: {OUTPUT_DIR / 'weather_mrp_value_function.png'}")

    plot_reward_transition_graph(
        mrp, filename=str(OUTPUT_DIR / "weather_mrp_graph.png")
    )

    # Test cases - Basic properties
    assert mrp.num_states == 2
    assert mrp.states == ("sunny", "rainy")
    assert np.allclose(mrp.transition_matrix, P)
    assert np.allclose(mrp.reward_function, R)
    assert mrp.discount_factor == gamma

    # Test get_reward
    assert mrp.get_reward("sunny") == 1.0
    assert mrp.get_reward("rainy") == -1.0

    # Test value function properties
    assert v_analytical.shape == (2,)
    assert v_iterative.shape == (2,)
    assert np.allclose(v_analytical, v_iterative, atol=1e-3)  # methods should agree
    # Sunny state should have higher value than rainy
    assert v_analytical[0] > v_analytical[1]

    # Test Bellman equation: V = R + γPV
    bellman_check = R + gamma * (P @ v_analytical)
    assert np.allclose(v_analytical, bellman_check, atol=1e-6)

    # Test return computation
    # G = 1 + 0.9*1 + 0.81*(-1) + 0.729*1 + 0.6561*(-1)
    expected_return = 1.0 + 0.9 * 1.0 + 0.81 * (-1.0) + 0.729 * 1.0 + 0.6561 * (-1.0)
    assert np.isclose(discounted_return, expected_return)


# Example 2: Student MRP with rewards
def test_student_mrp():
    print("=" * 60)
    print("Example 2: Student Markov Reward Process")
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
    R = np.array([-2.0, -2.0, -2.0, 10.0, 0.0])  # noqa: N806
    gamma = 0.99

    mrp = MarkovRewardProcess(states, P, R, gamma)
    print(f"Created: {mrp}")
    print(f"Number of states: {mrp.num_states}")
    print(f"Discount factor: {mrp.discount_factor}")
    print(f"Rewards: {dict(zip(states, R, strict=True))}")

    # Compute value function
    v_analytical = mrp.compute_value_function(method="analytical")
    v_iterative = mrp.compute_value_function(method="iterative", max_iterations=5000)
    print("\nValue function (analytical):")
    for s, v in zip(states, v_analytical, strict=True):
        print(f"  V({s}) = {v:.4f}")

    # Visualization - save to test_output/
    fig = plot_value_function(mrp, method="analytical", color="forestgreen")
    fig.savefig(OUTPUT_DIR / "student_mrp_value_function.png")
    print(f"\nSaved: {OUTPUT_DIR / 'student_mrp_value_function.png'}")

    plot_reward_transition_graph(
        mrp, filename=str(OUTPUT_DIR / "student_mrp_graph.png"), layout="dot"
    )

    # Test cases - Basic properties
    assert mrp.num_states == 5
    assert mrp.states == ("Class1", "Class2", "Class3", "Pass", "Sleep")
    assert np.allclose(mrp.transition_matrix, P)
    assert np.allclose(mrp.reward_function, R)
    assert mrp.discount_factor == gamma

    # Test value function properties
    assert v_analytical.shape == (5,)
    assert np.allclose(v_analytical, v_iterative, atol=1.0)  # converges slowly

    # Absorbing states should have value = R / (1 - γ) for γ < 1
    # V(Pass) = 10 / (1 - 0.99) = 1000
    # V(Sleep) = 0 / (1 - 0.99) = 0
    assert np.isclose(v_analytical[3], 10.0 / (1 - gamma), atol=1e-2)  # Pass
    assert np.isclose(v_analytical[4], 0.0, atol=1e-6)  # Sleep

    # Test Bellman equation: V = R + γPV
    bellman_check = R + gamma * (P @ v_analytical)
    assert np.allclose(v_analytical, bellman_check, atol=1e-5)

    # Value of Class states should be positive (can reach Pass with +10 reward)
    assert v_analytical[0] > 0  # Class1
    assert v_analytical[1] > 0  # Class2
    assert v_analytical[2] > 0  # Class3
