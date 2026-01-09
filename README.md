# rlcore

A Python library for reinforcement learning algorithms.

## Installation

```bash
uv sync
```

## Usage

### Tabular Methods

The `rlcore.tabular` module implements Markov Processes (MP) and Markov Reward Processes (MRP). See `tests/test_markov_process.py` and `tests/test_markov_reward_process.py` for examples.

## Visualization

The library exports graphs to GraphML format for visualization with [yEd Live](https://www.yworks.com/yed-live/).

```python
from rlcore.tabular.visualization import plot_markov_process, plot_reward_transition_graph

# Export Markov Process
plot_markov_process(mp, filename="my_mp.graphml")

# Export MRP with reward-colored nodes
plot_reward_transition_graph(mrp, filename="my_mrp.graphml")
```

### Viewing in yEd Live

1. Go to [yEd Live](https://www.yworks.com/yed-live/)
2. Click **File → Open** and select your `.graphml` file
3. Apply automatic layout: **Layout → Hierarchic**

## Development

```bash
uv run pytest tests/
uv run ruff check rlcore/ tests/
uv run ruff format rlcore/ tests/
```
