import itertools
import multiprocessing as mp
import os

from Execution import Execution


class SimulationRun:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "PhD_statistics")

    def create(self):
        os.makedirs(self.folder_path, exist_ok=True)
        datasets = ["Square", "Colors", "Cube", "Image"]
        # datasets = ["Square", "Shape", "Cube", "Colors", "Digits", "Image"]
        topologies = ["Hex", "Grid"]
        bmu_search = ["Parallel", "Fast", "Normal"]

        for d in datasets:
            for t in topologies:
                for b in bmu_search:
                    for s in range(0, 10):
                        exec = Execution()
                        exec.metadata = {"name": d+"_"+b+t+"_32x32n_20ep-"+str(s+1), "seed": s+1}
                        exec.dataset = {"type": d}
                        exec.model = {"topology": t, "bmu_search": b, "nb_epochs": 10, "width": 32, "height": 32}
                        self.all_runs.append(exec)

    def save(self):
        for e in self.all_runs:
            e.save(self.folder_path)

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.open(full_path)
                self.all_runs.append(exec)

    def compute(self, nb_cores=1):
        pool = mp.Pool(nb_cores)
        pool.starmap(Execution.full_simulation, zip(self.all_runs, itertools.repeat(sr.folder_path)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    sr = SimulationRun()
    if not os.path.exists(sr.folder_path):
        sr.create()
        sr.save()
    #sr.create()
    #sr.save()
    sr.open_folder(sr.folder_path)
    sr.compute(15)
