from graphviz import Digraph

def plot_interval_tree(intervals):
    dot = Digraph()

    for i, interval in enumerate(intervals):
        dot.node(str(i), str(interval))

    # Manually defining the tree structure for the provided intervals
    # This is a simplified example and does not represent the actual logic of an interval tree
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    for start, end in edges:
        dot.edge(str(start), str(end))

    return dot

# Intervals for the demonstration
intervals = [[5, 20], [10, 30], [15, 40], [20, 60], [30, 70]]

# Plotting the interval tree
plot_interval_tree(intervals)
