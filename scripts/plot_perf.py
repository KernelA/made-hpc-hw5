import argparse
import csv

from matplotlib import pyplot as plt

def plot_time(num_cpu, time):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(num_cpu, time)
    ax.grid(True)
    ax.set_ylabel("Время вычислений в секнудах")
    ax.set_xlabel("Число процессоров")
    return fig

def main(args):
    cvs_reader = csv.reader(args.file)
    
    num_processes = []
    elapsed = []

    for row in cvs_reader:
        num_cpu, time = tuple(map(float, row))
        num_processes.append(num_cpu)
        elapsed.append(time)

    fig = plot_time(num_processes, elapsed)
    fig.savefig(f"perf.jpg", dpi=150, pil_kwargs={"optimize": True, "quality": 95, "progressive": True}, bbox_inches="tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=argparse.FileType("r", encoding="utf-8"), required=True)

    args = parser.parse_args()

    main(args)