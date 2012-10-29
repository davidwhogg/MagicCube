# This file is part of the Magic Cube project.
# Copyright 2012 David W. Hogg (NYU).

"""
conventions
-----------

- This is a model of where the stickers are, not where the solid cubies are.  That's a bug not a feature.
- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the variable `Cube.faces`.
- The layers of the cube have names that are composed of a face letter and a number, with 0 indicating the outermost face.
- Every layer has two layer names, for instance, (F, 1) and (B, 1) are the same layer of a 3x3x3 cube; (F, 1) and (B, 3) are the same layer of a 5x5x5.
- The colors have integers and one-letter names. The one-letter color names are given by the variable `Cube.colors`.
- Convention is x before y in face arrays.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Cube(object):

    facedict = {"U":0, "D":1, "F":2, "B":3, "R":4, "L":5}
    dictface = dict([(v, k) for k, v in facedict.items()])
    colordict = {"w":0, "y":1, "b":2, "g":3, "o":4, "r":5}
    pltpos = [(0., 1.), (0., -1.), (0., 0.), (2., 0.), (1., 0.), (-1., 0.)]
    pltcolors = [(1.00, 1.00, 1.00), (0.75, 0.75, 0.00), (0.00, 0.00, 0.75),
                 (0.00, 0.75, 0.00), (1.00, 0.50, 0.00), (0.75, 0.00, 0.00)]

    def __init__(self, N):
        self.N = N
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        return None

    def move(self, f, l, d):
        i = Cube.facedict[f]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = Cube.facedict[f2]
            for d in ds:
                self.rotate([(Cube.facedict["F"], range(self.N), l2),
                             (Cube.facedict["R"], range(self.N), l2),
                             (Cube.facedict["B"], range(self.N), l2),
                             (Cube.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            f2 = "B"
            i2 = Cube.facedict[f2]
            for d in ds:
                self.rotate([(Cube.facedict["U"], range(self.N), l),
                             (Cube.facedict["L"], l2, range(self.N)),
                             (Cube.facedict["D"], range(self.N)[::-1], l2),
                             (Cube.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            f2 = "L"
            i2 = Cube.facedict[f2]
            for d in ds:
                self.rotate([(Cube.facedict["U"], l2, range(self.N)),
                             (Cube.facedict["F"], l2, range(self.N)),
                             (Cube.facedict["D"], l2, range(self.N)),
                             (Cube.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.move("R", l2, -d)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        print "moved", f, l, len(ds)
        return None

    def rotate(self, args):
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self):
        for t in range(10 * self.N * self.N): # made up
            f = Cube.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            self.move(f, l, 1)
        return None

    def render(self):
        fig = plt.figure(figsize=(4. * self.N / 5., 3. * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False,
                          xticks=[], yticks=[])
        for f, i in Cube.facedict.items():
            x0, y0 = Cube.pltpos[i]
            cs = 1. / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs), cs, cs, ec="k",
                                            fc=Cube.pltcolors[self.stickers[i, j, k]], alpha=0.75))
            ax.text(x0 + 0.5, y0 + 0.5, f, color="0.25", ha="center", va="center", rotation=20, alpha=0.5, fontsize=12)
        ax.set_xlim(-1., 3.)
        ax.set_ylim(-1., 2.)
        return fig

def edge_algo(cube):
    ls = range(cube.N)[1:-1]
    d.move("R", 0, -1)
    for l in ls:
        d.move("U", l, 1)
    d.move("R", 0, 2)
    for l in ls:
        d.move("U", l, 2)
    d.move("R", 0, -1)
    d.move("U", 0, -1)
    d.move("R", 0, 1)
    for l in ls:
        d.move("U", l, 2)
    d.move("R", 0, 2)
    for l in ls:
        d.move("U", l, -1)
    d.move("R", 0, 1)
    d.move("U", 0, 1)
    return None

if __name__ == "__main__":
    c = Cube(5)
    c.render().savefig("test.pdf")
    c.move("U", 0, 2)
    c.render().savefig("test1.pdf")
    c.move("U", 1, 1)
    c.render().savefig("test2.pdf")
    c.move("F", 0, -1)
    c.render().savefig("test3.pdf")
    c.move("U", 0, -1)
    c.render().savefig("test4.pdf")
    c.move("L", 0, 1)
    c.render().savefig("test5.pdf")
    c.move("L", 0, -1)
    c.render().savefig("test6.pdf")
    c.randomize()
    c.render().savefig("test7.pdf")
    d = Cube(3)
    edge_algo(d)
    d.render().savefig("test8.pdf")
    e = Cube(2)
    e.randomize()
    e.render().savefig("test9.pdf")
