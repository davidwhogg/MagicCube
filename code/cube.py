"""
This file is part of the Magic Cube project.

license
-------
Copyright 2012 David W. Hogg (NYU).

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301 USA.

usage
-----
- initialize a solved cube with `c = Cube(N)` where `N` is the side length.
- randomize a cube with `c.randomize(32)` where `32` is the number of random moves to make.
- make cube moves with `c.move()` and turn the whole cube with `c.turn()`.
- make figures with `c.render().savefig(fn)` where `fn` is the filename.
- change sticker colors with, eg, `c.stickercolors[Cube.colordict["w"]] = (0., 0., 0.)`.

conventions
-----------
- This is a model of where the stickers are, not where the solid cubies are.  That's a bug not a feature.
- Cubes are NxNxN in size.
- The faces have integers and one-letter names. The one-letter face names are given by the dictionary `Cube.facedict`.
- The layers of the cube have names that are composed of a face letter and a number, with 0 indicating the outermost face.
- Every layer has two layer names, for instance, (F, 1) and (B, 1) are the same layer of a 3x3x3 cube; (F, 1) and (B, 3) are the same layer of a 5x5x5.
- The colors have integers and one-letter names. The one-letter color names are given by the dictionary `Cube.colordict`.
- Convention is x before y in face arrays, plus an annoying baked-in left-handedness.  Sue me.  Or fork, fix, pull-request.

to-do
-----
- Write translations to other move languages, so you can take a string of moves from some website (eg, <http://www.speedcubing.com/chris/3-permutations.html>) and execute it.
- Figure out a physical "cubie" model to replace the "sticker" model.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

class Cube(object):
    """
    Cube
    ----
    Initialize with arguments:
    - `N`, the side length (the cube is `N`x`N`x`N`)
    - optional `whiteplastic=True` if you like white cubes
    """
    facedict = {"U":0, "D":1, "F":2, "B":3, "R":4, "L":5}
    dictface = dict([(v, k) for k, v in facedict.items()])
    normals = [np.array([0., 1., 0.]), np.array([0., -1., 0.]),
               np.array([0., 0., 1.]), np.array([0., 0., -1.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.])]
    # this xdirs has to be synchronized with the self.move() function
    xdirs = [np.array([1., 0., 0.]), np.array([1., 0., 0.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
               np.array([0., 0., -1.]), np.array([0, 0., 1.])]
    colordict = {"w":0, "y":1, "b":2, "g":3, "o":4, "r":5}
    pltpos = [(0., 1.05), (0., -1.05), (0., 0.), (2.10, 0.), (1.05, 0.), (-1.05, 0.)]
    labelcolor = (0.75, 0.25, 1.00)

    def __init__(self, N, whiteplastic=False):
        """
        (see above)
        """
        self.N = N
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.stickercolors = [(1.00, 1.00, 1.00), (0.75, 0.75, 0.00), (0.00, 0.00, 0.50),
                              (0.00, 0.50, 0.00), (1.00, 0.50, 0.00), (0.75, 0.00, 0.00)]
        if whiteplastic:
            self.plasticcolor = (0.85, 0.85, 0.85)
        else:
            self.plasticcolor = (0.15, 0.15, 0.15)
        return None

    def turn(self, f, d):
        """
        Turn whole cube (without making a layer move) around face `f`
        `d` 90-degree turns in the clockwise direction.  Use `d=3` or
        `d=-1` for counter-clockwise.
        """
        for l in range(self.N):
            self.move(f, l, d)
        return None

    def move(self, f, l, d):
        """
        Make a layer move of layer `l` parallel to face `f` through
        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `l` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        """
        i = Cube.facedict[f]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = Cube.facedict[f2]
            for d in ds:
                self._rotate([(Cube.facedict["F"], range(self.N), l2),
                              (Cube.facedict["R"], range(self.N), l2),
                              (Cube.facedict["B"], range(self.N), l2),
                              (Cube.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            f2 = "B"
            i2 = Cube.facedict[f2]
            for d in ds:
                self._rotate([(Cube.facedict["U"], range(self.N), l),
                              (Cube.facedict["L"], l2, range(self.N)),
                              (Cube.facedict["D"], range(self.N)[::-1], l2),
                              (Cube.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            f2 = "L"
            i2 = Cube.facedict[f2]
            for d in ds:
                self._rotate([(Cube.facedict["U"], l2, range(self.N)),
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

    def _rotate(self, args):
        """
        Internal function for the `move()` function.
        """
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self, number):
        """
        Make `number` randomly chosen moves to scramble the cube.
        """
        for t in range(number):
            f = Cube.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            t = 1 + np.random.randint(3)
            self.move(f, l, t)
        return None

    def _render_points(self, points, viewpoint):
        """
        Internal function for the `render()` function.  Clunky
        projection from 3-d to 2-d, but also return a zorder variable.
        """
        v2 = np.dot(viewpoint, viewpoint)
        zdir = viewpoint / np.sqrt(v2)
        xdir = np.cross(np.array([0., 1., 0.]), zdir)
        xdir /= np.sqrt(np.dot(xdir, xdir))
        ydir = np.cross(zdir, xdir)
        result = []
        for p in points:
            dpoint = p - viewpoint
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1. * viewpoint)
            result += [np.array([np.dot(xdir, dproj),
                                 np.dot(ydir, dproj),
                                 np.dot(zdir, dpoint / np.sqrt(v2))])]
        return result

    def render_views(self, ax):
        """
        Make three projected 3-dimensional views of the cube for the
        `render()` function.
        """
        csz = 2. / self.N
        for viewpoint, shift in [(np.array([-3., -3., 6.]), np.array([-1.5, 3.])),
                                 (np.array([3., 3., 6.]), np.array([0.5, 3.])),
                                 (np.array([6., 3., -3.]), np.array([2.5, 3.]))]:
            for f, i in Cube.facedict.items():
                xdir = Cube.xdirs[i]
                zdir = Cube.normals[i]
                ydir = np.cross(zdir, xdir) # insanity: left-handed!
                for j in range(self.N):
                    for k in range(self.N):
                        corners = [zdir - xdir + (j + 0) * csz * xdir - ydir + (k + 0) * csz * ydir,
                                   zdir - xdir + (j + 1) * csz * xdir - ydir + (k + 0) * csz * ydir,
                                   zdir - xdir + (j + 1) * csz * xdir - ydir + (k + 1) * csz * ydir,
                                   zdir - xdir + (j + 0) * csz * xdir - ydir + (k + 1) * csz * ydir]
                        projects = self._render_points(corners, viewpoint)
                        xys = [p[0:2] + shift for p in projects]
                        zorder = np.mean([p[2] for p in projects])
                        ax.add_artist(Polygon(xys, ec=self.plasticcolor, fc=self.stickercolors[self.stickers[i, j, k]], zorder=zorder))
                x0, y0, zorder = self._render_points([1.5 * Cube.normals[i], ], viewpoint)[0]
                ax.text(x0 + shift[0], y0 + shift[1], f, color=Cube.labelcolor,
                        ha="center", va="center", rotation=20, alpha=1.0, zorder=zorder, fontsize=12 / (-zorder))
        return None

    def render_flat(self, ax):
        """
        Make an unwrapped, flat view of the cube for the `render()`
        function.
        """
        for f, i in Cube.facedict.items():
            x0, y0 = Cube.pltpos[i]
            cs = 1. / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs), cs, cs, ec=self.plasticcolor,
                                            fc=self.stickercolors[self.stickers[i, j, k]], zorder=1.))
            ax.text(x0 + 0.5, y0 + 0.5, f, color=Cube.labelcolor,
                    ha="center", va="center", rotation=20, alpha=1.0, fontsize=14, zorder=2.)

    def render(self):
        """
        Visualize the cube in a standard layout, including a flat,
        unwrapped view and three perspective views.
        """
        fig = plt.figure(figsize=(5.8 * self.N / 5., 5.2 * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False,
                          xticks=[], yticks=[])
        self.render_views(ax)
        self.render_flat(ax)
        ax.set_xlim(-2.4, 3.4)
        ax.set_ylim(-1.2, 4.)
        return fig

def edge_algo(cube):
    """
    Do a standard edge-flipping algorithm.  Used for testing.
    """
    ls = range(cube.N)[1:-1]
    cube.move("R", 0, -1)
    for l in ls:
        cube.move("U", l, 1)
    cube.move("R", 0, 2)
    for l in ls:
        cube.move("U", l, 2)
    cube.move("R", 0, -1)
    cube.move("U", 0, -1)
    cube.move("R", 0, 1)
    for l in ls:
        cube.move("U", l, 2)
    cube.move("R", 0, 2)
    for l in ls:
        cube.move("U", l, -1)
    cube.move("R", 0, 1)
    cube.move("U", 0, 1)
    return None

if __name__ == "__main__":
    """
    Functional testing.
    """
    np.random.seed(17)
    c = Cube(4, whiteplastic=True)
    c.stickercolors[Cube.colordict["w"]] = (0., 0., 0.)
    c.turn("F", 2)
    c.turn("R", 1)
    for m in range(32):
        c.render().savefig("test%02d.pdf" % m)
        c.render().savefig("test%02d.png" % m, dpi=434 / c.N)
        c.randomize(1)
