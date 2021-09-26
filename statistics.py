import numpy as np
import os

from Execution import Execution


class Statistics:
    def __init__(self):
        self.all_runs = []
        self.folder_path = os.path.join("Executions", "PhD_statistics")

    def open_folder(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                self.open_folder(full_path)
            else:
                exec = Execution()
                exec.open(full_path)
                self.all_runs.append(exec)

    def stats(self):
        datasets = ["Square", "Colors", "Cube", "Image"]
        # datasets = ["Square", "Shape", "Cube", "Colors", "Digits", "Image"]
        topologies = ["Grid", "Hex"]
        bmu_search = ["Normal", "Fast", "Parallel"]

        results = {}

        for e in self.all_runs:
            name = e.dataset["type"]+e.model["topology"]+e.model["bmu_search"]
            if name not in results:
                results[name] = []
            results[name].append([e.metrics["MSDtN"], e.metrics["MSQE_S"], e.metrics["MSQE_F"], e.metrics["Mismatch"]])

        for d in datasets:
            for t in topologies:
                for b in bmu_search:
                    values = np.asarray(results[d+t+b])
                    values = np.mean(values, 0)
                    print(d, "&", t, b, "& \\nbr{"+'{:.3e}'.format(values[0])+"} & \\nbr{"+'{:.3e}'.format(values[1])+"} & \\nbr{"+'{:.3e}'.format(values[2])+"} & "+'{:.1f}'.format(values[3]*100)+"\\%\\\\")


if __name__ == '__main__':
    sr = Statistics()
    sr.open_folder(sr.folder_path)
    sr.stats()