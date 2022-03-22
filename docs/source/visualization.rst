.. _visualization:

Visualization
=============

There are currently two methods for visualization, and neither is anywhere near perfect.

Visualization with Paraview
===========================

Paraview provides a nice way to overlay text, view streamlines, and generally customize your
visualization. Unfortunately the renderer is only OK and getting it to work with a custom
python environment can be very finicky.


Visualization with Blender
==========================

Blender is not really designed with scientific workflows in mind, but it is very flexible,
plays more nicely with user python environments, has multiple renderers, all of which are
better than Paraview's, and is just generally way more developer friendly. It unfortunately
doesn't have many of the nice builtin-facilities of Paraview (like text overlay and
streamlines), and so is therefore very much a work in progress. There is an experimental
blender script `skelly_blend.py` that hopefully will become the primary way to visualize the
simulation in the future. For actually doing science, Paraview is still the preferred choice,
but hopefully that will change in the future.
