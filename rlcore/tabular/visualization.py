"""Visualization utilities for Markov processes using yEd-compatible GraphML.

Uses duck typing - functions accept any object with the required attributes:
- MarkovProcess-like: .states, .transition_matrix
- MRP-like: above + .reward_function, .discount_factor, .compute_value_function()
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# Default edge colors for transitions
DEFAULT_EDGE_COLOR = "#1f77b4"
SELF_LOOP_COLOR = "#9467bd"


@dataclass
class GraphInfo:
    """Information about a rendered graph."""

    filename: str
    num_nodes: int
    num_edges: int


def _graphml_header() -> str:
    """Generate GraphML header for yEd compatibility."""
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:y="http://www.yworks.com/xml/graphml"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">

  <key id="d0" for="node" yfiles.type="nodegraphics"/>
  <key id="d1" for="edge" yfiles.type="edgegraphics"/>
  <key id="d2" for="graph" yfiles.type="resources"/>

  <graph id="G" edgedefault="directed">
    <data key="d2">
      <y:HierarchicLayout/>
    </data>
"""


def _graphml_footer() -> str:
    """Generate GraphML footer."""
    return """
  </graph>
</graphml>
"""


def _graphml_node(
    node_id: str,
    label: str,
    fill_color: str = "#FFCC00",
    font_color: str = "black",
    width: float = 70.0,
    height: float = 40.0,
    shape: str = "ellipse",
) -> str:
    """Generate a yEd-compatible GraphML node."""
    return f"""
    <node id="{node_id}">
      <data key="d0">
        <y:ShapeNode>
          <y:Geometry width="{width}" height="{height}"/>
          <y:Fill color="{fill_color}"/>
          <y:BorderStyle color="#000000" width="1.0"/>
          <y:NodeLabel
              autoSizePolicy="node_size"
              modelName="internal"
              modelPosition="c"
              alignment="center"
              textAlignment="center"
              fontFamily="Dialog"
              fontSize="12"
              fontColor="{font_color}"
              visible="true">{label}</y:NodeLabel>
          <y:Insets top="0" bottom="0" left="0" right="0"/>
          <y:Shape type="{shape}"/>
        </y:ShapeNode>
      </data>
    </node>
"""


def _graphml_edge(
    edge_id: str,
    source: str,
    target: str,
    label: str,
    color: str = DEFAULT_EDGE_COLOR,
    is_self_loop: bool = False,
) -> str:
    """Generate a yEd-compatible GraphML edge."""
    if is_self_loop:
        graphics = f"""
        <y:PolyLineEdge>
          <y:LineStyle color="{color}" width="1.0"/>
          <y:Arrows source="none" target="standard"/>
          <y:Path>
            <y:Point x="20" y="-20"/>
            <y:Point x="-20" y="-20"/>
          </y:Path>
          <y:EdgeStyle type="loop"/>
          <y:EdgeLabel
              modelName="loop"
              modelPosition="c"
              fontSize="10">{label}</y:EdgeLabel>
        </y:PolyLineEdge>
"""
    else:
        graphics = f"""
        <y:PolyLineEdge>
          <y:LineStyle color="{color}" width="1.0"/>
          <y:Arrows source="none" target="standard"/>
          <y:EdgeLabel
              autoSizePolicy="content"
              modelName="edge_relative"
              modelPosition="center"
              preferredPlacement="above"
              offset="0.0 30.0"
              alignment="center"
              textAlignment="center"
              fontSize="10">{label}</y:EdgeLabel>
        </y:PolyLineEdge>
"""
    return f"""
    <edge id="{edge_id}" source="{source}" target="{target}">
      <data key="d1">
{graphics}
      </data>
    </edge>
"""


def plot_markov_process(
    markov_process: Any,
    filename: str = "markov_process.graphml",
    node_color: str = "#FFCC00",
    edge_color: str = DEFAULT_EDGE_COLOR,
    self_loop_color: str = SELF_LOOP_COLOR,
) -> GraphInfo:
    """
    Export a Markov process to yEd-compatible GraphML format.

    Args:
        markov_process: The Markov process to export.
        filename: Output GraphML filename.
        node_color: Fill color for nodes (hex format).
        edge_color: Color for regular edges (hex format).
        self_loop_color: Color for self-loop edges (hex format).

    Returns:
        GraphInfo with filename, node count, and edge count.
    """
    out = [_graphml_header()]

    state_list = list(markov_process.states)

    # Add nodes
    for state in state_list:
        out.append(_graphml_node(str(state), str(state), fill_color=node_color))

    # Add edges with transition probabilities
    transition_matrix = markov_process.transition_matrix
    edge_idx = 0
    num_edges = 0

    for i, from_state in enumerate(state_list):
        for j, to_state in enumerate(state_list):
            prob = transition_matrix[i, j]
            if prob > 0:
                is_self_loop = from_state == to_state
                color = self_loop_color if is_self_loop else edge_color
                label = f"p={prob:.3f}"
                out.append(
                    _graphml_edge(
                        f"e{edge_idx}",
                        str(from_state),
                        str(to_state),
                        label,
                        color=color,
                        is_self_loop=is_self_loop,
                    )
                )
                edge_idx += 1
                num_edges += 1

    out.append(_graphml_footer())

    Path(filename).write_text("".join(out))

    logger.info(
        "Exported GraphML: %s (nodes=%d, edges=%d)",
        filename,
        len(state_list),
        num_edges,
    )

    return GraphInfo(filename=filename, num_nodes=len(state_list), num_edges=num_edges)


def plot_reward_transition_graph(
    mrp: Any,
    filename: str = "mrp.graphml",
    cmap: str = "RdYlGn",
    edge_color: str = DEFAULT_EDGE_COLOR,
    self_loop_color: str = SELF_LOOP_COLOR,
    show_rewards_on_edges: bool = False,
) -> GraphInfo:
    """
    Export a Markov Reward Process to yEd-compatible GraphML format.

    Nodes are colored based on their reward values using the specified colormap.

    Args:
        mrp: The Markov Reward Process to export.
        filename: Output GraphML filename.
        cmap: Matplotlib colormap name for reward-based node coloring.
        edge_color: Color for regular edges (hex format).
        self_loop_color: Color for self-loop edges (hex format).
        show_rewards_on_edges: If True, show reward info on edges.

    Returns:
        GraphInfo with filename, node count, and edge count.
    """
    out = [_graphml_header()]

    state_list = list(mrp.states)
    rewards = mrp.reward_function
    r_min, r_max = rewards.min(), rewards.max()

    if r_max - r_min > 0:
        normalized_rewards = (rewards - r_min) / (r_max - r_min)
    else:
        normalized_rewards = np.full_like(rewards, 0.5)

    colormap = plt.get_cmap(cmap)

    def reward_to_hex(normalized_value: float) -> str:
        rgba = colormap(normalized_value)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Add nodes with reward-based coloring
    for i, state in enumerate(state_list):
        reward = rewards[i]
        fill_color = reward_to_hex(normalized_rewards[i])
        font_color = "black" if normalized_rewards[i] > 0.4 else "white"
        label = f"{state}\nR={reward:.2f}"
        out.append(
            _graphml_node(
                str(state),
                label,
                fill_color=fill_color,
                font_color=font_color,
                height=50.0,
            )
        )

    # Add edges with transition probabilities
    transition_matrix = mrp.transition_matrix
    edge_idx = 0
    num_edges = 0

    for i, from_state in enumerate(state_list):
        for j, to_state in enumerate(state_list):
            prob = transition_matrix[i, j]
            if prob > 0:
                is_self_loop = from_state == to_state
                color = self_loop_color if is_self_loop else edge_color

                if show_rewards_on_edges:
                    label = f"p={prob:.3f} r={rewards[j]:.2f}"
                else:
                    label = f"p={prob:.3f}"

                out.append(
                    _graphml_edge(
                        f"e{edge_idx}",
                        str(from_state),
                        str(to_state),
                        label,
                        color=color,
                        is_self_loop=is_self_loop,
                    )
                )
                edge_idx += 1
                num_edges += 1

    out.append(_graphml_footer())

    Path(filename).write_text("".join(out))

    logger.info(
        "Exported MRP GraphML: %s (nodes=%d, edges=%d)",
        filename,
        len(state_list),
        num_edges,
    )

    return GraphInfo(filename=filename, num_nodes=len(state_list), num_edges=num_edges)


def plot_value_function(
    mrp: Any,
    method: str = "analytical",
    figsize: tuple[float, float] = (10, 6),
    color: str = "steelblue",
    title: str | None = None,
) -> Figure:
    """Plot value function as a bar chart."""
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
