Magic Cube
==========

It had to happen someday.  Somebody stop me!

.. image:: http://4.bp.blogspot.com/-iruqaXDstKk/UKBejowDVkI/AAAAAAAAZkM/c2tir0qcexQ/s400/test04.png
   :alt: cube views
   :align: left

Authors
-------

- **David W. Hogg** (NYU)
- **Jacob Vanderplas** (UW)

Usage
-----

Interactive Cube
~~~~~~~~~~~~~~~~
To use the matplotlib-based interactive cube, run 

     python code/cube_interactive.py

If you want a cube with a different number of sides, use e.g.

     python code/cube_interactive.py 5

This will create a 5x5x5 cube

This code should currently be considered to be in beta --
there are several bugs and the GUI has an incomplete set of features

Controls
********
- **Click and drag** to change the viewing angle of the cube.  Holding shift
  while clicking and dragging adjusts the line-of-sight rotation.
- **Arrow keys** may also be used to change the viewing angle.  The shift
  key has the same effect
- **U/D/L/R/B/F** keys rotate the faces a quarter turn clockwise.  Hold the
  shift key to rotate counter-clockwise.  Hold a number i to turn the slab
  at a depth i (e.g. for a 3x3 cube, holding "1" and pressing "L" will turn
  the center slab).

Other
~~~~~
There are more tools available -- for now, RTFSC.


License
-------

All content copyright 2012 the authors.
**Magic Cube** is licensed under the *GPLv2 License*.
See the `LICENSE.txt` file for more information.
