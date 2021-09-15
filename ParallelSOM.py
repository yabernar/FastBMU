import itertools

import numpy as np

from Common_Functions import *

class Parallel_Fast:
    def __init__(self, som, vector, vector_id):
        self.som = som
        self.vector = vector
        self.vector_id = vector_id
        self.dist_memory = np.zeros(self.som.neurons_nbr, dtype=float)
        self.positions = [(0, 0), (self.som.neurons_nbr[0]-1, 0), (0, self.som.neurons_nbr[1]-1), (self.som.neurons_nbr[0]-1, self.som.neurons_nbr[1]-1)]
        self.best = [0, 0, 0, 0]
        self.status = [True, True, True, True]

        for i in range(len(self.positions)):
            self.best[i] = quadratic_distance(self.som.neurons[self.positions[i]], self.vector)
            self.dist_memory[self.positions[i]] = self.best[i]

    def update(self):
        for i in range(len(self.status)):
            if self.status[i]:
                res = self.particle_tick(self.positions[i], self.best[i])
                self.positions[i] = res[0]
                self.best[i] = res[1]
                self.status[i] = res[2]
        if np.sum(self.status) == 0:
            i = np.argmin(self.best)
            return self.positions[i]
        else:
            return False

    def particle_tick(self, position, best):
        bmu = position
        new_bmu, new_best = self.som.search_smallest_neighbor(self.vector, bmu, best)
        if new_bmu == bmu:  # There is no better neuron in the direct neighbourhood, so we go one step further
            cont = False
            if self.som.topology == "Hex":
                neighbors = self.som.return_all_neighbors_hex(bmu)
            else:
                neighbors = self.som.return_all_neighbors(bmu)
            for i in neighbors:
                neighbors_bmu, neighbors_best = self.som.search_smallest_neighbor(self.vector, i, np.inf)
                if neighbors_bmu != bmu: # A better BMU has been found in the neighbours, so we continue the descent
                    best = neighbors_best
                    bmu = neighbors_bmu
                    cont = True
                    break
        else:
            cont = True
            best = new_best
            bmu = new_bmu
        return bmu, best, cont

class ParallelSOM:
    def __init__(self, parameters):
        # Parameters
        self.alpha = parameters["alpha"]
        self.sigma = parameters["sigma"]
        self.data = parameters["data"]
        self.neurons_nbr = parameters["neurons_nbr"]
        self.epochs_nbr = parameters["epochs_nbr"]
        self.bmu_search = parameters["bmu_search"]
        self.topology = parameters["topology"]
        self.metrics = {}

        # Measuring metrics
        self.dist_memory = np.zeros(self.neurons_nbr)
        self.visited = np.zeros(self.neurons_nbr)
        self.last_bmu = np.full(self.data.shape[0], None)
        if self.topology == "Hex":
            self.distance_to_last_bmu = np.zeros(hexagonal_distance((0, 0), self.neurons_nbr))
        else:
            self.distance_to_last_bmu = np.zeros(manhattan_distance((0, 0), self.neurons_nbr))
        self.cycle_count = np.zeros(np.prod(self.neurons_nbr))

        # Computing variables
        self.neurons = np.random.random(self.neurons_nbr + self.data.shape[1:])
        self.vector_list = None
        self.distance_to_input = None
        if self.topology == "Hex":
            self.distance_vector = np.empty(hexagonal_distance((0, 0), self.neurons_nbr))
        else:
            self.distance_vector = np.empty(manhattan_distance((0, 0), self.neurons_nbr))
        self.iteration = 0
        self.epoch = 0
        self.max_iterations = self.epochs_nbr * self.data.shape[0]

    def winner(self, vector, vector_id=None):
        dist = np.empty(self.neurons_nbr, dtype=float)
        for i in np.ndindex(dist.shape):
            dist[i] = quadratic_distance(self.neurons[i], vector)
        self.distance_to_input = dist
        bmu_index = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

        # Calculating distance with previous BMU
        if vector_id is not None :
            if self.last_bmu[vector_id] is not None:
                self.distance_to_last_bmu[manhattan_distance(bmu_index, self.last_bmu[vector_id])] += 1
                order = []
                for i in range(manhattan_distance((0, 0), self.neurons_nbr)):
                    order.append([])
                it = np.nditer(dist, flags=['multi_index'])
                for i in it:
                    order[manhattan_distance(self.last_bmu[vector_id], it.multi_index)].append(float(i))
                order = np.asarray(list(itertools.chain.from_iterable(order)))
                #cycle = 0
                #while len(order) > 0:
                #    cycle += 1
                #    order = order[order < order[0]]
                #self.cycle_count[cycle] += 1
                order = order[order < order[0]]
                if len(order) > 0:
                    self.cycle_count[1+int(np.ceil(np.log2(len(order))))] += 1
                else:
                    self.cycle_count[1] +=1

            else:
                self.last_bmu[vector_id] = bmu_index

        return bmu_index  # Returning the Best Matching Unit's index.

    def fast_winner(self, vector, vector_id=None):
        self.dist_memory = np.zeros(self.neurons_nbr, dtype=float)
        starting_positions = [(0, 0), (self.neurons_nbr[0]-1, 0), (0, self.neurons_nbr[1]-1), (self.neurons_nbr[0]-1, self.neurons_nbr[1]-1)]
        best = np.inf
        bmu = None
        # Starting the 4 particles and picking the smallest
        for i in starting_positions:
            current = self.particle(vector, i)
            value = quadratic_distance(self.neurons[current], vector)
            if value < best:
                best = value
                bmu = current

        self.visited += self.dist_memory > 0
        # Calculating distance with previous BMU
        if vector_id is not None :
            if self.last_bmu[vector_id] is not None:
                self.distance_to_last_bmu[manhattan_distance(bmu, self.last_bmu[vector_id])] += 1
            else:
                self.last_bmu[vector_id] = bmu

        return bmu

    def particle(self, vector, start):
        cont = True
        bmu = start
        best = quadratic_distance(self.neurons[bmu], vector)
        self.dist_memory[bmu] = best
        while cont:
            new_bmu, new_best = self.search_smallest_neighbor(vector, bmu, best)
            if new_bmu == bmu:  # There is no better neuron in the direct neighbourhood, so we go one step further
                cont = False
                if self.topology == "Hex":
                    neighbors = self.return_all_neighbors_hex(bmu)
                else:
                    neighbors = self.return_all_neighbors(bmu)
                for i in neighbors:
                    neighbors_bmu, neighbors_best = self.search_smallest_neighbor(vector, i, np.inf)
                    if neighbors_bmu != bmu: # A better BMU has been found in the neighbours, so we continue the descent
                        best = neighbors_best
                        bmu = neighbors_bmu
                        cont = True
                        break
            else:
                best = new_best
                bmu = new_bmu
        return bmu

    def search_smallest_neighbor(self, vector, bmu, best):
        if self.topology == "Hex":
            neighbors = self.return_all_neighbors_hex(bmu)
        else:
            neighbors = self.return_all_neighbors(bmu)
        for i in neighbors:
            value = quadratic_distance(self.neurons[i[0], i[1]], vector)
            self.dist_memory[i] = value
            if value < best:
                best = value
                bmu = i
        return bmu, best

    def return_all_neighbors(self, neuron):
        neighbors = []
        if neuron[0] < self.neurons_nbr[0] - 1:
            neighbors.append((neuron[0] + 1, neuron[1]))
        if neuron[0] > 0:
            neighbors.append((neuron[0] - 1, neuron[1]))
        if neuron[1] < self.neurons_nbr[1] - 1:
            neighbors.append((neuron[0], neuron[1] + 1))
        if neuron[1] > 0:
            neighbors.append((neuron[0], neuron[1] - 1))
        return neighbors

    def return_all_neighbors_hex(self, neuron):
        neighbors = self.return_all_neighbors(neuron)
        if neuron[1] % 2 == 0 and neuron[0] > 0:
            if neuron[1] > 0:
                neighbors.append((neuron[0] - 1, neuron[1] - 1))
            if neuron[1] < self.neurons_nbr[1] - 1:
                neighbors.append((neuron[0] - 1, neuron[1] + 1))
        if neuron[1] % 2 == 1 and neuron[0] < self.neurons_nbr[0]-1:
            if neuron[1] > 0:
                neighbors.append((neuron[0] + 1, neuron[1] - 1))
            if neuron[1] < self.neurons_nbr[1] - 1:
                neighbors.append((neuron[0] + 1, neuron[1] + 1))
        return neighbors

    def get_all_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.winner(self.data[i])
        return winners_list

    def get_all_fast_winners(self):
        winners_list = np.zeros(self.data.shape[0], dtype=tuple)  # list of BMU for each corresponding training vector
        for i in np.ndindex(winners_list.shape):
            winners_list[i] = self.fast_winner(self.data[i])
        return winners_list

    def run_iteration(self):
        if self.iteration >= self.max_iterations:
            return False

        # Start of an epoch
        if self.iteration % self.data.shape[0] == 0:
            self.generate_random_list()
            self.alpha.execute()
            self.sigma.execute()
            for i in range(len(self.distance_vector)):
                self.distance_vector[i] = normalized_gaussian(i / (len(self.distance_vector) - 1), self.sigma.get())

        self.iteration += 1
        vector_id = self.unique_random_vector()
        vector = self.data[vector_id]
        if self.bmu_search == "Fast":
            bmu = self.fast_winner(vector, vector_id)
        else:
            bmu = self.winner(vector, vector_id)
        self.updating_weights(bmu, vector)
        return bmu[0], bmu[1]

    def run_epoch(self):
        for i in range(self.data.shape[0]):
            self.run_iteration()

    def run_parallel_epoch(self):
        self.generate_random_list()
        self.alpha.execute()
        self.sigma.execute()
        for i in range(len(self.distance_vector)):
            self.distance_vector[i] = normalized_gaussian(i / (len(self.distance_vector) - 1), self.sigma.get())

        search_list = []
        while len(self.vector_list) > 0 or len(search_list) > 0:
            if len(self.vector_list) > 0:
                for i in range(4):
                    if len(self.vector_list) > 0:
                        search = Parallel_Fast(self, self.data[self.unique_random_vector()], self.current_vector_index)
                        search_list.append(search)
            for e in search_list:
                bmu = e.update()
                if bmu is not False:
                    self.updating_weights(bmu, e.vector)
                    search_list.remove(e)

    def run(self):
        for i in range(self.epochs_nbr):
            print("Epoch", i)
            self.run_epoch()

    def run_parallel(self):
        for i in range(self.epochs_nbr):
            print("Epoch", i)
            self.run_parallel_epoch()

    def updating_weights(self, bmu, vector):
        for i in np.ndindex(self.neurons_nbr):
            if self.topology == "Hex":
                dist = hexagonal_distance(np.asarray(i), np.asarray(bmu))
            else:
                dist = manhattan_distance(np.asarray(i), np.asarray(bmu))
            self.neurons[i] += self.alpha.get() * self.distance_vector[dist] * (vector - self.neurons[i])

    ### Random input vector selection section
    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        self.current_vector_index = self.vector_list.pop(0)
        return self.current_vector_index

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        np.random.shuffle(self.vector_list)

    ### Metrics section
    def mean_square_distance_to_neighbour(self):
        error = np.zeros(self.neurons.shape)
        for i in np.ndindex(self.neurons_nbr):
            if self.topology == "Hex":
                neighbors = self.return_all_neighbors_hex(i)
            else:
                neighbors = self.return_all_neighbors(i)
            for n in neighbors:
                error[i] += np.mean((self.neurons[i] - self.neurons[n])**2)
            error[i] /= len(neighbors)
        return np.mean(error)

    def mean_square_quantization_error(self, winners=None):
        if winners is not None:
            winners = self.get_all_winners()
        error = np.zeros(winners.shape)
        for i in np.ndindex(winners.shape):
            error[i] = np.mean((self.data[i] - self.neurons[winners[i]])**2)
        return np.mean(error)

    def compute_metrics(self):
        real_winners = np.array(self.get_all_winners())
        fast_winners = self.get_all_fast_winners()

        self.metrics["MSDtN"] = self.mean_square_distance_to_neighbour()
        self.metrics["MSQE_S"] = self.mean_square_quantization_error(real_winners)
        self.metrics["MSQE_F"] = self.mean_square_quantization_error(fast_winners)
        self.metrics["Mismatch"] = np.sum(real_winners == fast_winners) / np.prod(self.neurons_nbr)
        return self.metrics
