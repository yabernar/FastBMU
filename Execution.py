import codecs
import json

import numpy
import numpy as np

from Data import *
from ParallelSOM import ParallelSOM
from Parameters import Variable, Parameters
from SOM import SOM

# This class is here to simplify the launching of tests runs and logs all parameters and results in a json file
class Execution:
    def __init__(self):
        self.metadata = {}
        self.dataset = {}
        self.model = {}
        self.metrics = {}

        self.data = None
        self.map = None

    def open(self, path):
        txt = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(txt)
        self.metadata = data["metadata"]
        self.dataset = data["dataset"]
        self.model = data["model"]
        self.metrics = data["metrics"]

    def save(self, path):
        data = {"metadata": self.metadata, "dataset": self.dataset, "model": self.model, "metrics": self.metrics}
        json.dump(data, codecs.open(os.path.join(path, self.metadata["name"] + ".json"), 'w', encoding='utf-8'),
                  indent=2)

    def load_dataset(self):
        if self.dataset["type"] == "Square":
            self.data = uniform(1000, 2)
        elif self.dataset["type"] == "Shape":
            self.data = shape(1000, "cat-silhouette.png")
        elif self.dataset["type"] == "Cube":
            self.data = uniform(1000, 3)
        elif self.dataset["type"] == "Colors":
            self.data = pixels_colors(1000, "Elijah.png")
        elif self.dataset["type"] == "Digits":
            self.data = spoken_digits("FSDD", 1000)
        elif self.dataset["type"] == "Image":
            self.data = mosaic_image("Elijah.png", [10, 10])
        else:
            print("Error : No dataset type specified !")

    def run(self):
        np.random.seed(self.metadata["seed"])
        if self.data is None:
            self.load_dataset()
        parameters = Parameters({"alpha": Variable(start=0.6, end=0.05, nb_steps=self.model["nb_epochs"]),
                                 "sigma": Variable(start=0.5, end=0.001, nb_steps=self.model["nb_epochs"]),
                                 "data": self.data,
                                 "neurons_nbr": (self.model["width"], self.model["height"]),
                                 "epochs_nbr": self.model["nb_epochs"],
                                 "topology": self.model["topology"],
                                 "bmu_search": self.model["bmu_search"]})
        if self.model["bmu_search"] == "Parallel":
            self.map = ParallelSOM(parameters)
            self.map.run_parallel()
        else:
            self.map = SOM(parameters)
            self.map.run()

    def compute_metrics(self):
        self.metrics = self.map.compute_metrics()

    def full_simulation(self, path):
        self.run()
        self.compute_metrics()
        self.save(path)
        print("Simulation", self.metadata["name"], "ended")


if __name__ == '__main__':
    exec = Execution()
    exec.metadata = {"name": "test", "seed": 1}
    exec.dataset = {"type": "Cube"}
    exec.model = {"topology": "Grid", "bmu_search": "Fast", "nb_epochs": 10, "width": 32, "height": 32}
    os.makedirs(os.path.join("Executions", "Test"), exist_ok=True)
    exec.full_simulation(os.path.join("Executions", "Test"))
    numpy.set_printoptions(precision=2, linewidth=np.inf, threshold=np.inf, suppress=True)
    print(exec.map.visited/np.max(exec.map.visited))
    print(exec.metrics)
    print(exec.map.cycle_count)
    total = 0
    for i in range(len(exec.map.cycle_count)):
        total += i*exec.map.cycle_count[i]
    print("Average cycle :", total/np.sum(exec.map.cycle_count), "Log cycle :", np.log2(np.prod(exec.map.neurons_nbr)))

    #print(exec.map.distance_to_last_bmu)
    #print(exec.map.distance_to_last_bmu/np.sum(exec.map.distance_to_last_bmu))