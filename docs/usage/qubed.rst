#####################
 Qube Expansion
#####################

The qube expansion system in ``earthkit-workflows`` provides tools
for expanding actions across multi-dimensional data structures
based on Qube definitions.

****************
 What is Qubed?
****************

The expansion system is built on `Qubed
<https://github.com/ecmwf/qubed>`_, a library for representing
multi-dimensional data structures. A Qube represents the dimensions
of your underlying data, defining the axes (dimensions) and their
coordinate values, along with optional hierarchical relationships
between different sets of dimensions.

Think of a Qube as representing how your data is organised across
multiple dimensions (time steps, parameters, vertical levels, etc.).

-  **Simple Qube**: Single set of dimensions (e.g., time steps: [6, 12, 18, 24])
-  **Hierarchical Qube**: Multiple branches with different dimension
   sets (e.g., surface variables vs. pressure level variables)

**************************
 Basic Expansion Workflow
**************************

The ``expand_as_qube`` method on Action takes a qube structure and
recursively expands the action across all dimensions defined in the qube.

Basic Example
=============

.. code:: python

   from qubed import Qube

   # Create a simple qube with time steps
   qube = Qube.from_datacube({"step": [6, 12, 18, 24, 30]})

   # View the dimensions
   print(qube.axes())
   # Output: {'step': array([ 6, 12, 18, 24, 30])}

   # Expand an action across all dimensions
   expanded_action = action.expand_as_qube(qube)

The action will be expanded across the specified dimensions, creating separate
execution paths for each coordinate value.

**************************
 Creating Qube Structures
**************************

You can manually construct Qube structures to define how your actions
should be expanded:

Simple Single-Dimension Expansion
=================================

.. code:: python

   from qubed import Qube

   # Create a simple qube with time steps
   qube = Qube.from_datacube({"step": [6, 12, 18, 24]})

   # Expand an action
   expanded_action = action.expand_as_qube(qube)

Multi-Dimensional Expansion
===========================

.. code:: python

   # Create a qube with multiple dimensions
   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q", "u", "v"],
       "level": [500, 850, 1000]
   })
   expanded_action = action.expand_as_qube(qube)

Hierarchical Expansion
======================

Create hierarchical qube structures for different variable types:

.. code:: python

   from qubed import Qube

   # Surface variables (2D fields)
   surface = Qube.from_datacube({
       "param": ["2t", "2d", "10u", "10v", "msl"]
   })
   surface.add_metadata({"name": "surface"})

   # Pressure level variables (3D fields)
   pressure = Qube.from_datacube({
       "param": ["t", "q", "u", "v"],
       "level": [500, 700, 850, 925, 1000]
   })
   pressure.add_metadata({"name": "pressure"})

   # Combine with time steps
   steps = Qube.from_datacube({"step": [6, 12, 18, 24]})
   combined = steps | (surface | pressure)

   # Expand the action
   expanded_action = action.expand_as_qube(combined)

The expanded action will have separate branches for ``/surface`` and
``/pressure``, each containing the appropriate parameters and levels.

**Understanding Name Metadata**

When a qube has multiple children (branches), the expansion creates separate
execution paths using the ``split()`` method. The path names for these branches
are determined by:

1. **Named branches**: If a child qube has ``{"name": "..."}`` metadata, that
   name is used as the path (e.g., ``/surface``, ``/pressure``)

2. **Automatic naming**: If no name metadata is provided, branches are
   automatically named using alphabetical labels (``/a``, ``/b``, ``/c``, etc.)

.. code:: python

   # Example: Automatic alphabetical naming
   child1 = Qube.from_datacube({"param": ["t", "q"]})
   child2 = Qube.from_datacube({"param": ["u", "v"]})
   parent = Qube.from_datacube({"step": [6, 12]})
   qube = parent | (child1 | child2)

   expanded = action.expand_as_qube(qube)
   # Creates branches: /a (for child1) and /b (for child2)

   # Example: Using meaningful names
   child1.add_metadata({"name": "temperature"})
   child2.add_metadata({"name": "wind"})
   qube = parent | (child1 | child2)

   expanded = action.expand_as_qube(qube)
   # Creates branches: /temperature and /wind

The name metadata is particularly useful for organising complex workflows with
multiple variable types, making the execution tree more readable and easier to
debug.

*******************
 Modifying Qubes
*******************

Dropping Axes
=============

You can remove dimensions from a qube before expanding:

.. code:: python

   # Create qube with multiple dimensions
   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q"],
       "level": [500, 850, 1000]
   })

   # Drop the time step dimension
   qube_no_steps = qube.remove_by_key("step")
   expanded = action.expand_as_qube(qube_no_steps)

   # Drop multiple dimensions
   qube_params_only = qube.remove_by_key(["step", "level"])
   expanded = action.expand_as_qube(qube_params_only)

Inspecting Axes
===============

View available dimensions in a qube:

.. code:: python

   qube = Qube.from_datacube({
       "step": [6, 12, 18],
       "param": ["t", "q"],
       "level": [500, 850, 1000]
   })

   axes = qube.axes()

   for axis_name, values in axes.items():
       print(f"{axis_name}: {sorted(values)}")

   # Check if an axis exists
   if "level" in axes:
       print(f"Pressure levels: {sorted(axes['level'])}")

*************
 API Summary
*************

**Action Methods**

-  ``action.expand_as_qube(qube)``: Expand an action according to a qube structure

**Qube Methods**

-  ``Qube.from_datacube(dims)``: Create a qube from dimension dictionary
-  ``qube.axes()``: View available dimensions
-  ``qube.remove_by_key(key)``: Remove dimension(s)
-  ``qube.add_metadata(metadata)``: Add metadata (e.g., names) to qube nodes

**See Also**

-  :doc:`/api/fluent` - Fluent API documentation
-  `Qubed Documentation <https://qubed.readthedocs.io/>`_ - Underlying
   data structure library
