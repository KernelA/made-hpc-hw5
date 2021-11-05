import argparse
import os
from typing import Sequence

from matplotlib import pyplot as plt

def plot_cell_automate(mask: Sequence[Sequence[int]], title: str):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.imshow(mask, cmap="gray_r")
    ax.set_xticks([])
    ax.grid(False)
    ax.set_ylabel("Шаг эволюции")
    ax.set_xlabel("Номер ячейки")
    ax.set_title(title)
    ax.set_xticks(tuple(range(len(mask[0]))))
    ax.set_yticks(tuple(range(len(mask))))
    ax.tick_params(axis="x", rotation=-90)
    ax.set
    return fig

def main(args):
    mask = list(map(lambda x: list(map(int, x)), filter(lambda x: len(x) >  0, map(str.strip, args.file))))

    rule_num = os.path.splitext(args.file.name)[0].split("_")[1]
    fig = plot_cell_automate(mask, f"Эволюцию клеточного автомата\nПравило {rule_num}")
    fig.savefig(f"evolution_{rule_num}.jpg", dpi=200, pil_kwargs={"optimize": True, "quality": 95, "progressive": True}, bbox_inches="tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=argparse.FileType("r", encoding="utf-8"), required=True)

    args = parser.parse_args()

    main(args)