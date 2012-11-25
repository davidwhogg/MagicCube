import numpy as np
import matplotlib.pyplot as plt
from projection import Quaternion, project_points


class PolyView3D(plt.Axes):
    def __init__(self, view=(0, 0, 10), fig=None,
                 rect=[0, 0, 1, 1], **kwargs):
        if fig is None:
            fig = plt.gcf()

        self.view = np.asarray(view)
        self.start_rot = Quaternion.from_v_theta((1, -1, 0), -np.pi / 6)

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        # Internal state variable
        self._event_xy = None
        self._current_rot = self.start_rot
        self._npts = [1]
        self._xyzs = [[0, 0, 0]]
        self._polys = []

        # initialize the axes.  We'll set some keywords by default
        kwargs.update(dict(aspect='equal',
                           xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
                           frameon=False, xticks=[], yticks=[]))
        super(PolyView3D, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)
        self.figure.canvas.mpl_connect('key_release_event',
                                       self._key_release)

    def poly3D(self, xyz, update=True, **kwargs):
        xyz = np.asarray(xyz)
        self._npts.append(xyz.shape[0] + self._npts[-1])
        self._xyzs = np.vstack([self._xyzs, xyz])

        self._polys.append(plt.Polygon(xyz[:, :2], **kwargs))
        self.add_patch(self._polys[-1])

        if update:
            self._update_projection()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def _update_projection(self):
        proj = project_points(self._xyzs, self._current_rot, self.view)
        for i in range(len(self._polys)):
            p = proj[self._npts[i]:self._npts[i + 1]]
            self._polys[i].set_xy(p[:, :2])
            self._polys[i].set_zorder(-p[:-1, 2].mean())
        self.figure.canvas.draw()

    def _key_press(self, event):
        """Handler for key press events"""
        if event.key == 'shift':
            self._ax_LR = (0, 0, 1)
            self._shift_on = True

        elif event.key == 'right':
            self.rotate(Quaternion.from_v_theta(self._ax_LR,
                                                self._step_LR))
        elif event.key == 'left':
            self.rotate(Quaternion.from_v_theta(self._ax_LR,
                                                -self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                -self._step_UD))
        self._update_projection()

    def _key_release(self, event):
        """Handler for key release event"""
        if event.key == 'shift':
            self._ax_LR = (0, -1, 0)

    def _mouse_press(self, event):
        """Handler for mouse button press"""
        if event.button == 1:
            self._event_xy = (event.x, event.y)

    def _mouse_release(self, event):
        """Handler for mouse button release"""
        if event.button == 1:
            self._event_xy = None

    def _mouse_motion(self, event):
        """Handler for mouse motion"""
        if self._event_xy is not None:
            dx = event.x - self._event_xy[0]
            dy = event.y - self._event_xy[1]
            self._event_xy = (event.x, event.y)
            rot1 = Quaternion.from_v_theta(self._ax_UD,
                                           self._step_UD * dy)
            rot2 = Quaternion.from_v_theta(self._ax_LR,
                                           self._step_LR * dx)

            self.rotate(rot1 * rot2)
            self._update_projection()


def cube_axes(N=1, **kwargs):
    stickerwidth = 0.9
    small = 0.5 * (1. - stickerwidth)
    d1 = 1 - small
    d2 = 1 - 2 * small
    d3 = 1.01
    base_sticker = np.array([[d1, d2, d3],
                             [d2, d1, d3],
                             [-d2, d1, d3],
                             [-d1, d2, d3],
                             [-d1, -d2, d3],
                             [-d2, -d1, d3],
                             [d2, -d1, d3],
                             [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)

    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta)
            for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta)
            for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    cubie_width = 2. / N
    translations = np.array([[-1 + (i + 0.5) * cubie_width,
                              -1 + (j + 0.5) * cubie_width, 0]
                             for i in range(N) for j in range(N)])

    colors = ['blue', 'green', 'white', 'yellow', 'orange', 'red']
    
    factor = np.array([1. / N, 1. / N, 1])

    ax = PolyView3D(**kwargs)
    for t in translations:
        base_face_trans = factor * base_face + t
        base_sticker_trans = factor * base_sticker + t
        for r, c in zip(rots, colors):
            face = r.rotate(base_face_trans)
            ax.poly3D(face, facecolor='k', update=False)
            sticker = r.rotate(base_sticker_trans)
            ax.poly3D(sticker, facecolor=c, update=False)

    ax._update_projection()

    ax.figure.text(0.05, 0.05, ("Drag Mouse or use arrow keys to change "
                                "perspective.\n"
                                "hold shift to rotate around z-axis"),
                   ha='left', va='bottom')
    return ax
        

if __name__ == '__main__':
    fig = plt.figure(figsize=(5, 5))
    fig.add_axes(cube_axes(N=3, fig=fig))
    plt.show()
