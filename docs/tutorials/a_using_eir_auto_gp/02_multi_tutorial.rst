.. _02-multi-tutorial:

02 – Example of training on multiple phenotypes
===============================================

In this relatively short tutorial,
we will be looking at one approach of using ``EIR-auto-GP``
to train models for multiple phenotypes.
This will be relatively short and technical tutorial,
but the idea is just to show one example workflow.


We will be using the same data from the `PennCATH study`_,
as in tutorial :ref:`01-basic-tutorial`,
but now we will be predicting
some of the other phenotypes in the dataset.
In particular, we will be predicting
**age**,
**tg** (triglycerides),
**hdl** (high density lipoprotein)
and **ldl** (low density lipoprotein).
It's indeed quite silly perhaps to be predicting age,
from genotype data,
but we're just playing around a bit.
As before, you start by downloading the `processed PennCATH data`_.

.. _PennCATH study: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)61996-4/fulltext
.. _processed PennCATH data: https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p

We will be reusing the data from the previous tutorial,
so the structure should look like this:

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/input_folder.txt
    :language: console

Since we are modelling on multiple phenotypes,
we would like to avoid processing the data multiple times.
We can start by preparing only the data, without any modelling,
by using the ``--only-data`` flag:

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/AUTO_2_DATA.txt
    :language: console
    :emphasize-lines: 6

This will only generate a ``data`` folder,
with the processed data,
inside the path passed to ``global_output_folder``.

Now, we can start training models for each phenotype.
The main difference from the previous tutorial
is that we will be reusing the processed data,
but passing in separate flags for the modelling, feature selection and analysis.
For example, in the case of **tg**:

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/AUTO_2_tg.txt
    :language: console
    :emphasize-lines: 4-7

.. note::

    It's a bit counter-intuitive that we are passing in ``data_output_folder``,
    when it's an "input" folder in this scenario. This is because
    the framework checks if the data folder exists,
    and if it does, it will skip the data processing step.

For the other phenotypes, here are the commands:

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/AUTO_2_hdl.txt
    :language: console

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/AUTO_2_ldl.txt
    :language: console

.. literalinclude:: ../tutorial_files/02_multi_tutorial/commands/AUTO_2_age.txt
    :language: console

While this is a bit manually set up in this tutorial, this can
easily be extended and automated as you see fit.

In case you are interested, here are the results on the test
set for each phenotype:

.. csv-table:: TG (triglycerides)
   :file: ../tutorial_files/02_multi_tutorial/figures/tg_test_results.csv
   :header-rows: 1

.. csv-table:: HDL (high density lipoprotein)
   :file: ../tutorial_files/02_multi_tutorial/figures/hdl_test_results.csv
   :header-rows: 1

.. csv-table:: LDL (low density lipoprotein)
   :file: ../tutorial_files/02_multi_tutorial/figures/ldl_test_results.csv
   :header-rows: 1

.. csv-table:: Age
   :file: ../tutorial_files/02_multi_tutorial/figures/age_test_results.csv
   :header-rows: 1

We see that the models explain the lipid phenotypes variably,
with HDL being the best predicted phenotype. Interestingly,
the PCC for age is ~0.25, which can be due to population
bias, sex and the fact we included the lipids as inputs.

That will be it for this tutorial. Thank you for reading!