import random
import copy
import numpy as np

# ------------------------
# Graph representation
# ------------------------
class Graph:
    def __init__(self, nodes, edges, node_features=None):
        self.nodes = nodes              # list of node ids
        self.edges = edges              # list of (u, v)
        self.node_features = node_features or {}

    def copy(self):
        return copy.deepcopy(self)


# ------------------------
# Mutation functions
# ------------------------
def mutate_add_edge(graph):
    g = graph.copy()
    u, v = random.sample(g.nodes, 2)
    if (u, v) not in g.edges:
        g.edges.append((u, v))
    return g

def mutate_remove_edge(graph):
    g = graph.copy()
    if g.edges:
        g.edges.pop(random.randrange(len(g.edges)))
    return g

def mutate_node_feature(graph):
    g = graph.copy()
    node = random.choice(g.nodes)
    g.node_features[node] = np.random.randn()
    return g

def mutate(graph):
    ops = [
        mutate_add_edge,
        mutate_remove_edge,
        mutate_node_feature,
    ]
    return random.choice(ops)(graph)


# ------------------------
# Custom dynamic loss
# ------------------------
def compute_loss(graph):
    """
    Replace this with your real objective.
    Can depend on:
    - structure
    - features
    - external data
    """
    num_edges = len(graph.edges)
    feature_sum = sum(graph.node_features.values()) if graph.node_features else 0

    # Example: encourage sparse + meaningful features
    return num_edges - feature_sum


# ------------------------
# Evolution loop
# ------------------------
def evolve(
    population,
    generations=50,
    mutation_rate=0.5,
    elite_frac=0.2
):
    for gen in range(generations):
        # Evaluate
        scored = [(g, compute_loss(g)) for g in population]
        scored.sort(key=lambda x: x[1])  # minimize loss

        # Select elites
        k = int(len(population) * elite_frac)
        elites = [g for g, _ in scored[:k]]

        # Generate offspring
        new_population = elites.copy()

        while len(new_population) < len(population):
            parent = random.choice(elites)
            child = parent.copy()

            if random.random() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population

        print(f"Gen {gen}: best loss = {scored[0][1]:.4f}")

    return population