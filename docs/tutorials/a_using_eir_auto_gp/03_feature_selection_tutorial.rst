.. _02-multi-tutorial:

03 – Comparison of Feature Selection Methods
============================================

In this tutorial,
we will be quickly going over the
different feature selection methods
that are implemented in the software.

We will be using the same data from the `PennCATH study`_,
as in tutorial :ref:`01-basic-tutorial`, predicting CAD.
As before, you start by downloading the `processed PennCATH data`_.

.. _PennCATH study: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)61996-4/fulltext
.. _processed PennCATH data: https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p

As we are reusing the data from the previous tutorial,
so the structure should look like this:

.. literalinclude:: ../tutorial_files/03_feature_selection_tutorial/data_run/commands/input_folder.txt
    :language: console

Since we are doing multiple runs with different feature selection methods,
we would like to avoid processing the data multiple times.
We can start by preparing only the data, without any modelling,
by using the ``--only-data`` flag:

.. literalinclude:: ../tutorial_files/03_feature_selection_tutorial/data_run/commands/AUTO_1_DATA.txt
    :language: console

This will only generate a ``data`` folder,
with the processed data,
inside the path passed to ``global_output_folder``.


Now, we can start training models for each feature selection method.
We will specifically focus on the following methods
``gwas``, ``gwas->dl`` and ``gwas+bo``. There are other methods available,
such as ``dl``, however, since we have around 500K SNPs and only around 2K samples,
we will skip the methods that do not include a GWAS pre-filtering step. This is
to (a) save time and (b) likely the models would be grossly overfit.

.. note::
    For more information on the feature selection methods,
    please refer to the output of ``eirautogp --help`` and the
    documentation page :ref:`feature-selection-methods`.


Here are the commands we can use to train the models:

.. literalinclude:: ../tutorial_files/03_feature_selection_tutorial/gwas/commands/AUTO_FS_GWAS.txt
    :language: console

.. literalinclude:: ../tutorial_files/03_feature_selection_tutorial/gwas->dl/commands/AUTO_FS_GWAS->DL.txt
    :language: console

.. literalinclude:: ../tutorial_files/03_feature_selection_tutorial/gwas+bo/commands/AUTO_FS_GWAS+BO.txt
    :language: console

In case you are interested, here are the results on the test
set for each of the approaches:

.. csv-table:: GWAS
   :file: ../tutorial_files/03_feature_selection_tutorial/gwas/figures/CAD_test_results.csv
   :header-rows: 1

.. csv-table:: GWAS->DL
   :file: ../tutorial_files/03_feature_selection_tutorial/gwas->dl/figures/CAD_test_results.csv
   :header-rows: 1

.. csv-table:: GWAS+BO
   :file: ../tutorial_files/03_feature_selection_tutorial/gwas+bo/figures/CAD_test_results.csv
   :header-rows: 1

So here, we can see that the default GWAS+BO approach performs the roughly best.
This method has worked quite well on smaller datasets in internal tests,
while it remains to be tested thoroughly on larger datasets such as the UKBB. For
larger datasets such as the UKBB, the GWAS->DL has mostly been used, but it can
well be that the GWAS+BO approach will work even better.

Thanks for reading this tutorial!