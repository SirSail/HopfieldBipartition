import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- Konfiguracja ---
DEFAULT_EDGE_WEIGHT_COEFF = 1.0
DEFAULT_RANDOM_SEED = 42
ITERATION_CHECK_INTERVAL = 100

# --- Narzędzia ---
def set_random_seed(seed: int):
    np.random.seed(seed)

def generate_initial_spins(n: int) -> np.ndarray:
    return np.random.choice([-1, 1], size=n)


# --- Hopfield Model ---
class HopfieldModel:
    def __init__(self, weight_matrix: np.ndarray, initial_spins: np.ndarray):
        self._weights = weight_matrix
        self._spins = initial_spins.copy()

    def local_field(self, index: int) -> float:
        return np.dot(self._weights[index], self._spins)

    def update_spin(self, index: int):
        local = self.local_field(index)
        new_spin = 1 if local > 0 else -1 if local < 0 else self._spins[index]
        if self._spins[index] != new_spin:
            self._spins[index] = new_spin

    def calculate_energy(self) -> float:
        return -0.5 * np.sum(self._weights * np.outer(self._spins, self._spins))

    def get_spins(self) -> np.ndarray:
        return self._spins.copy()

# --- Budowanie macierzy wag ---
class WeightMatrixBuilder:
    def __init__(self, graph: nx.Graph, edge_weight_coefficient: float, cut_penalty: float):
        self.graph = graph
        self.coeff = edge_weight_coefficient
        self.cut_penalty = cut_penalty

    def build(self, index_map: dict) -> np.ndarray:
        n = len(index_map)
        matrix = np.zeros((n, n))
        for u, v, data in self.graph.edges(data=True):
            i, j = index_map[u], index_map[v]
            weight = data.get('weight', 1.0)
            adjusted = 2 * (self.coeff * weight - self.cut_penalty)
            matrix[i, j] = matrix[j, i] = adjusted
        return matrix

# --- Optymalizacja ---
class PartitionOptimizer:
    def __init__(self, model: HopfieldModel, max_iter: int, check_interval: int, track_energy: bool = False):
        self.model = model
        self.max_iter = max_iter
        self.check_interval = check_interval
        self.track_energy = track_energy
        self.energy_history = [] if track_energy else None

    def run(self):
        for iteration in range(self.max_iter):
            index = np.random.randint(len(self.model.get_spins()))
            self.model.update_spin(index)
            if self.track_energy and iteration % self.check_interval == 0:
                self.energy_history.append(self.model.calculate_energy())

# --- Główna klasa ---
class GraphPartitioner:
    def __init__(self, graph: nx.Graph, edge_weight_coefficient=DEFAULT_EDGE_WEIGHT_COEFF,
                 cut_penalty=None, max_iter=10000, seed=DEFAULT_RANDOM_SEED,
                 check_interval=ITERATION_CHECK_INTERVAL, track_energy=False):

        np.random.seed(seed)
        if not graph or graph.number_of_nodes() == 0:
            raise ValueError("Graph must be non-empty and properly initialized.")

        self._graph = graph
        self._nodes = list(graph.nodes)
        self._n = len(self._nodes)
        self._index_map = {node: i for i, node in enumerate(self._nodes)}

        self._cut_penalty = cut_penalty or self._default_cut_penalty()
        spins = generate_initial_spins(self._n)

        weight_builder = WeightMatrixBuilder(graph, edge_weight_coefficient, self._cut_penalty)
        weights = weight_builder.build(self._index_map)

        self._model = HopfieldModel(weights, spins)
        self._optimizer = PartitionOptimizer(self._model, max_iter, check_interval, track_energy)

    def _default_cut_penalty(self) -> float:
        weights = [data.get('weight', 1.0) for _, _, data in self._graph.edges(data=True)]
        return DEFAULT_EDGE_WEIGHT_COEFF * np.mean(weights) * 2 if weights else 2.0

    def optimize_partition(self):
        self._optimizer.run()

    def get_partition(self):
        spins = self._model.get_spins()
        pos = [self._nodes[i] for i in range(self._n) if spins[i] == 1]
        neg = [self._nodes[i] for i in range(self._n) if spins[i] == -1]
        return pos, neg

    def get_energy_history(self):
        return self._optimizer.energy_history or []

    def get_energy(self) -> float:
        return self._model.calculate_energy()


def visualize_partition(graph: nx.Graph, positive_partition):
    colors = ['red' if node in positive_partition else 'blue' for node in graph.nodes()]
    nx.draw(graph, with_labels=True, node_color=colors, node_size=500)
    plt.title("Visualization of Graph Bipartition")
    plt.show()

# TESTY JEDNOSTKOWE 

def test_partition_sums_to_n():
    graph = nx.complete_graph(5)
    partitioner = GraphPartitioner(graph, max_iter=100)
    partitioner.optimize_partition()
    pos, neg = partitioner.get_partition()
    assert len(pos) + len(neg) == 5, "Partition does not cover all nodes"

def test_graph_with_no_edges():
    graph = nx.empty_graph(5)
    partitioner = GraphPartitioner(graph, max_iter=100)
    partitioner.optimize_partition()
    pos, neg = partitioner.get_partition()
    assert len(pos) + len(neg) == 5, "Even without edges, all nodes should be assigned"

def test_invalid_graph():
    try:
        GraphPartitioner(None)
        assert False, "Should have raised ValueError for None graph"
    except ValueError:
        pass

def test_energy_decreases_or_constant():
    graph = nx.cycle_graph(6)
    partitioner = GraphPartitioner(graph, max_iter=1000, track_energy=True)
    partitioner.optimize_partition()
    energies = partitioner.get_energy_history()
    assert all(earlier >= later for earlier, later in zip(energies, energies[1:])), "Energy should not increase"

if __name__ == "__main__":
    # Część główna
    test_graph = nx.erdos_renyi_graph(n=10, p=0.3)
    for u, v in test_graph.edges():
        test_graph[u][v]['weight'] = np.random.uniform(0.5, 2.0)

    partitioner = GraphPartitioner(test_graph, max_iter=5000, track_energy=True)
    print("Energy before:", partitioner.get_energy())
    partitioner.optimize_partition()
    print("Energy after:", partitioner.get_energy())
    pos, neg = partitioner.get_partition()
    print(f"Positive partition ({len(pos)}):", pos)
    print(f"Negative partition ({len(neg)}):", neg)
    visualize_partition(test_graph, pos)

    # Testy
    test_partition_sums_to_n()
    test_graph_with_no_edges()
    test_invalid_graph()
    test_energy_decreases_or_constant()
    print("All tests passed.")

