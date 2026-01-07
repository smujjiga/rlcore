"""Visualization utilities for Markov processes."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
from matplotlib.figure import Figure
from tqdm import tqdm

from rlcore.tabular.markov_process import MarkovProcess
from rlcore.tabular.markov_reward_process import MarkovRewardProcess


def plot_markov_process(
    markov_process: MarkovProcess[Any],
    filename: str = "markov_process.png",
    layout: str = "dot",
    dpi: int = 300,
) -> None:
    """
    Plot a Markov process and save as PNG.

    Args:
        markov_process: The Markov process to visualize.
        filename: Output PNG filename.
        layout: Graph layout algorithm ('dot', 'neato', 'circo', 'fdp', 'sfdp').
        dpi: Resolution of the output image.
    """
    # Create directed graph
    dot = Digraph(format="png", engine=layout)
    dot.attr(dpi=str(dpi))
    dot.attr(
        "node",
        shape="circle",
        style="filled",
        fillcolor="lightblue",
        fontsize="12",
        width="0.75",
    )
    dot.attr("edge", fontsize="10")

    # Add nodes
    state_list = list(markov_process.states)
    for state in state_list:
        dot.node(str(state))

    # Add edges with transition probabilities
    transition_matrix = markov_process.transition_matrix
    for i, from_state in enumerate(state_list):
        for j, to_state in enumerate(state_list):
            prob = transition_matrix[i, j]
            if prob > 0:
                dot.edge(str(from_state), str(to_state), label=f"{prob:.3f}")

    # Render to file (graphviz adds extension, so we remove it if present)
    output_path = filename.removesuffix(".png")
    dot.render(output_path, cleanup=True)

    num_edges = sum(1 for p in transition_matrix.flat if p > 0)
    print(f"Saved visualization: {filename}")
    print(f"  Nodes: {len(state_list)}, Edges: {num_edges}")


def plot_value_function(
    mrp: MarkovRewardProcess[Any],
    method: str = "analytical",
    figsize: tuple[float, float] = (10, 6),
    color: str = "steelblue",
    title: str | None = None,
) -> Figure:
    """Plot value function as a bar chart.
    """
    # Compute value function
    value_function = mrp.compute_value_function(method=method)
    states = list(mrp.states)
    state_labels = [str(s) for s in states]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    x = np.arange(len(states))
    ax.bar(x, value_function, color=color, edgecolor="black", linewidth=0.5)

    # Customize plot
    ax.set_xlabel("State")
    ax.set_ylabel("Value V(s)")
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)

    if title is None:
        title = f"State Value Function (γ={mrp.discount_factor:.2f}, method={method})"
    ax.set_title(title)

    # Add value labels on bars
    for i, v in enumerate(value_function):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")

    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()

    return fig


def plot_reward_transition_graph(
    mrp: MarkovRewardProcess[Any],
    filename: str = "mrp_reward_graph.png",
    layout: str = "dot",
    dpi: int = 300,
    cmap: str = "RdYlGn",
    node_size: str = "0.75",
    edge_labels: bool = True,
    title: str | None = None,
) -> None:
    """Plot transition graph with reward-colored nodes.
    """
    # Get rewards and normalize to [0, 1] for colormap
    rewards = mrp.reward_function
    r_min, r_max = rewards.min(), rewards.max()

    if r_max - r_min > 0:
        normalized_rewards = (rewards - r_min) / (r_max - r_min)
    else:
        normalized_rewards = np.full_like(rewards, 0.5)

    # Get colormap
    colormap = plt.get_cmap(cmap)

    def reward_to_hex(normalized_value: float) -> str:
        """Convert normalized reward to hex color."""
        rgba = colormap(normalized_value)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Create directed graph
    dot = Digraph(format="png", engine=layout)
    dot.attr(dpi=str(dpi))

    if title is None:
        title = "MRP Reward Transition Graph"
    dot.attr(label=title, labelloc="t", fontsize="14")

    dot.attr("edge", fontsize="10")

    # Add nodes with reward-based coloring
    state_list = list(mrp.states)
    for i, state in tqdm(enumerate(state_list), total=len(state_list), desc="Nodes"):
        reward = rewards[i]
        fill_color = reward_to_hex(normalized_rewards[i])
        # Use black text for light colors, white for dark
        font_color = "black" if normalized_rewards[i] > 0.4 else "white"

        dot.node(
            str(state),
            label=f"{state}\nR={reward:.2f}",
            shape="circle",
            style="filled",
            fillcolor=fill_color,
            fontcolor=font_color,
            fontsize="11",
            width=node_size,
        )

    # Add edges with transition probabilities
    transition_matrix = mrp.transition_matrix
    n = len(state_list)
    for i, from_state in tqdm(enumerate(state_list), total=n, desc="Edges"):
        for j, to_state in enumerate(state_list):
            prob = transition_matrix[i, j]
            if prob > 0:
                label = f"{prob:.2f}" if edge_labels else ""
                dot.edge(str(from_state), str(to_state), label=label)

    # Render to file
    output_path = filename.removesuffix(".png")
    dot.render(output_path, cleanup=True)

    num_edges = sum(1 for p in transition_matrix.flat if p > 0)
    print(f"Saved reward graph: {filename}")
    print(f"  Nodes: {len(state_list)}, Edges: {num_edges}")
