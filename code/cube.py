# This file is part of the Magic Cube project.
# Copyright 2012 David W. Hogg (NYU).

"""
conventions
-----------

- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the variable `Cube.faces`.
- The layers of the cube have names that are composed of a face letter and a number, with 0 indicating the outermost face.
- The middle layers of odd-N cubes have two layer names, for instance, f1 and b1 are the same layer of a 3x3x3 cube.
- The colors have integers and one-letter names. The one-letter color names are given by the variable `Cube.colors`.

"""

import numpy as np
import matplotlib.pyplot as plt

class Cube(object):

    faces =  ['u', 'd', 'f', 'b', 'r', 'l']
    colors = ['w', 'y', 'b', 'g', 'o', 'r']
    pltcolors = [(1.00, 1.00, 1.00), (0.75, 0.75, 0.00), (0.00, 0.00, 0.75),
                 (0.00, 0.75, 0.00), (1.00, 0.50, 0.00), (0.75, 0.00, 0.00)]

    def __init__(self, N):
        self.N = N
        self.stickers = np.array([np.tile(c, (self.N, self.N)) for c in Cube.colors])
        print self.stickers
        return None

    def move(self, layers, direction):
        return None

    def randomize(self):
        return None

    def render(self, ax):
        
        return None

if __name__ == "__main__":
    c = Cube(3)
    plt.clf()
    c.render()
    plt.savefig("test.pdf")
