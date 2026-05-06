Formula DSL Reference
=====================

``xftsim`` uses a lavaan-style domain-specific language (DSL) to define
phenogenetic architectures declaratively. Each line in a formula string
defines one component of the model.

Basic Syntax
------------

Every line follows the pattern::

    LHS ~ RHS

where ``LHS`` is the output phenotype name and ``RHS`` is either a
built-in function call or an arithmetic aggregation expression.

.. important::

   **One component per line.** Do not combine multiple components on a
   single line with ``+``. For example, this is **wrong**::

       height ~ genetic(eff) + noise(0.5)

   Instead, write::

       height.G ~ genetic(eff)
       height.E ~ noise(0.5)
       height ~ height.G + height.E

Comments and blank lines are allowed:

.. code-block:: text

   # This is a comment
   height.G ~ genetic(eff)

   # Blank lines above are fine
   height.E ~ noise(0.5)
   height ~ height.G + height.E

Available Functions
-------------------

Genetic Components
^^^^^^^^^^^^^^^^^^

``genetic(effect_name)``
   Univariate additive genetic effects. Computes the matrix-vector product
   of the haplotype operator with the named effect specification. The
   effect name must be a key in the ``effects`` dict passed to
   ``Architecture.from_formula()``.

   .. code-block:: text

      height.G ~ genetic(eff)

``mvGenetic(effect_name)``
   Multivariate genetic effects. The effect must be a ``MultivariateEffects``
   with ``k`` matching the number of outputs in the tuple LHS.

   .. code-block:: text

      (height.G, bmi.G) ~ mvGenetic(pleiotropic_eff)

``haplotypeGenetic(effect_name)`` or ``haplotypeGenetic(effect_name, haplotype='maternal')``
   Haplotype-specific genetic effects. The ``haplotype`` keyword selects
   which haplotype to use (``'maternal'`` or ``'paternal'``; default is
   ``'maternal'``).

   .. code-block:: text

      height.G_mat ~ haplotypeGenetic(eff, haplotype='maternal')
      height.G_pat ~ haplotypeGenetic(eff, haplotype='paternal')

Noise Components
^^^^^^^^^^^^^^^^

``noise(variance)``
   Independent Gaussian noise with the given variance.

   .. code-block:: text

      height.E ~ noise(0.5)

``cnoise(cov=[[...]])``
   Multivariate correlated Gaussian noise. The ``cov`` argument is a
   square covariance matrix literal. The number of outputs in the tuple
   LHS must match the dimension of the matrix.

   .. code-block:: text

      (height.E, bmi.E) ~ cnoise(cov=[[0.5, 0.1], [0.1, 0.3]])

Parental Components
^^^^^^^^^^^^^^^^^^^

These components look up phenotype values from the parent generation.

``parent(phenotype_name)``
   Average of both parents' phenotype values.

``mother(phenotype_name)``
   Mother's phenotype value only.

``father(phenotype_name)``
   Father's phenotype value only.

All parental components support a ``founder=`` keyword argument to specify
how founder (generation 0) values are generated, since founders have no
parents:

.. code-block:: text

   height.VT ~ parent(height, founder=noise(0.2))

Currently only ``noise(variance)`` is supported as the founder fallback.

.. code-block:: text

   height.mat ~ mother(height, founder=noise(0.3))
   height.pat ~ father(height, founder=noise(0.3))

Sibling Components
^^^^^^^^^^^^^^^^^^

Sibling components compute summary statistics across siblings (individuals
sharing the same family). The source phenotype must be computed before the
sibling component that reads from it.

.. important::

   When adding sibling components programmatically via ``arch.add()``,
   you must pass ``inputs=['source_name']`` explicitly. The formula parser
   handles this automatically.

``sibling_mean(source_name)``
   Mean of the source phenotype across siblings in the same family.

``sibling_sum(source_name)``
   Sum of the source phenotype across siblings.

``sibling_any(source_name)``
   1.0 if any sibling has a nonzero value, 0.0 otherwise.

``sibling_count(source_name)``
   Number of siblings (family size).

``sibling_eldest(source_name)``
   Value from the eldest sibling (lowest index in family).

``sibling_youngest(source_name)``
   Value from the youngest sibling (highest index in family).

.. code-block:: text

   height.sib_mean ~ sibling_mean(height)
   height.sib_any ~ sibling_any(height)

Aggregation Expressions
^^^^^^^^^^^^^^^^^^^^^^^

An aggregation expression combines previously defined phenotype components
using arithmetic operators (``+``, ``-``, ``*``, ``/``):

.. code-block:: text

   height ~ height.G + height.E + height.VT

Variable names in the expression must match outputs defined on earlier
lines. The parser automatically detects the dependencies.

Grouping Operator
-----------------

The ``|`` operator specifies a grouping variable for a component. When a
component has a grouping variable, it operates within groups defined by
that variable (e.g., per family, per sex).

.. code-block:: text

   height.E ~ noise(0.5) | FID
   height.E ~ noise(0.5) | sex

Only components whose class has ``accepts_grouping = True`` support the
``|`` operator. Aggregation expressions do not support grouping.

Multivariate LHS
-----------------

For components that produce multiple outputs (``mvGenetic``, ``cnoise``),
use a tuple LHS with parentheses:

.. code-block:: text

   (height.G, bmi.G) ~ mvGenetic(pleiotropic_eff)
   (height.E, bmi.E) ~ cnoise(cov=[[0.5, 0.1], [0.1, 0.3]])

The number of names in the tuple must match the dimensionality of the
component (e.g., the ``k`` of the effect or the dimension of the
covariance matrix).

Example Architectures
---------------------

Simple Additive + Noise
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   height.G ~ genetic(eff)
   height.E ~ noise(0.5)
   height ~ height.G + height.E

Two Correlated Traits
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   (height.G, bmi.G) ~ mvGenetic(pleiotropic_eff)
   (height.E, bmi.E) ~ cnoise(cov=[[0.5, 0.1], [0.1, 0.3]])
   height ~ height.G + height.E
   bmi ~ bmi.G + bmi.E

Vertical Transmission
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   height.G ~ genetic(eff)
   height.E ~ noise(0.3)
   height.VT ~ parent(height, founder=noise(0.2))
   height ~ height.G + height.E + height.VT

Haplotype-Specific Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   height.G_mat ~ haplotypeGenetic(eff, haplotype='maternal')
   height.G_pat ~ haplotypeGenetic(eff, haplotype='paternal')
   height.E ~ noise(0.5)
   height ~ height.G_mat + height.G_pat + height.E

Sibling Effects
^^^^^^^^^^^^^^^

.. code-block:: text

   height.G ~ genetic(eff)
   height.E ~ noise(0.5)
   height ~ height.G + height.E
   height.sib_mean ~ sibling_mean(height)

Programmatic Construction
-------------------------

Architectures can also be built programmatically using
``Architecture.add()``:

.. code-block:: python

   from xftsim.arch import (
       Architecture, ArchNode,
       GeneticComponent, NoiseComponent, AggregationComponent,
   )
   from xftsim.effect import AdditiveEffects

   arch = Architecture()

   arch.add(ArchNode(
       outputs=['height.G'],
       component=GeneticComponent(effects=my_effect),
       inputs=[],
   ))
   arch.add(ArchNode(
       outputs=['height.E'],
       component=NoiseComponent(variance=0.5),
       inputs=[],
   ))
   arch.add(ArchNode(
       outputs=['height'],
       component=AggregationComponent(expression='height.G + height.E'),
       inputs=['height.G', 'height.E'],
   ))

See :doc:`../api/arch` for full class documentation.
