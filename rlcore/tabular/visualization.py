"""Visualization utilities for Markov processes."""

from typing import Any

from graphviz import Digraph

from rlcore.tabular.markov_process import MarkovProcess


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
