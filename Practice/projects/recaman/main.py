import sys
import numpy as np

import matplotlib.pyplot as plt
from collections import OrderedDict

# main subplots definition
fig, ax = plt.subplots()

def draw_semi(left, right, up: bool = True):
    # left, right = sorted((x1, x2))
    
    # make sure x1 < x2
    # if x2 < x1: x2, x1 = x1, x2 # swap if incorrect order

    # left, right = sorted((x1, x2))
    center = (left + right) / 2
    radius = (right - left) / 2

    x = np.linspace(left, right, 500)
    y = np.sqrt(radius**2 - (x - center)**2)

    if not up: y = -y

    ax.plot(x, y)

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

def generate_recaman(recaman_len: int = 10, verbose: bool = False):
    if not isinstance(recaman_len, int):
        raise ValueError(f"`iters` must be an integer not {recaman_len}")

    recaman_seq: OrderedDict = OrderedDict()

    num = 0
    jump = 1
    recaman_seq[0] = 0

    while jump <= recaman_len:
        new_num = num - jump
        if new_num <= 0 or new_num in recaman_seq.values():
            new_num = num + jump
        if verbose:
            print(num, jump, new_num, list(recaman_seq.values()), new_num in recaman_seq.values())

        recaman_seq[jump] = new_num
        num = new_num
        jump += 1

    return recaman_seq

def plot_recaman(recaman_len: int = 10, verbose: bool = False):
    if recaman_len < 2:
        ValueError('`recaman_len` must be > 2')
    recaman_seq = generate_recaman(recaman_len=recaman_len, verbose=verbose)

    # # main subplots definition
    # fig, ax = plt.subplots()
    # keys = list(recaman_seq.keys())
    up = False
    for i in range(len(recaman_seq) - 1):
        draw_semi(recaman_seq[i], recaman_seq[i + 1], up=up)
        up = not up
    plt.show()

    return recaman_seq

recaman_seq_len = 10
try:
    recaman_seq_len = sys.argv[1]
    recaman_seq_len = int(recaman_seq_len)
except:
    pass

recaman_seq = generate_recaman(recaman_seq_len)
recaman = plot_recaman(recaman_seq_len)
