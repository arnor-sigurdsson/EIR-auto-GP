.. _quickstart:

Quickstart
==========

This Quickstart guide aims to provide a brief overview of
how you can get started with ``EIR-auto-GP``, where we will
be training deep learning models to predict coronary artery
disease (CAD) from genomic data. See the :ref:`01-basic-tutorial` for
a more detailed tutorial.

A - Setup
---------

Download the `processed PennCATH data`_ and set up your folders.

.. _processed PennCATH data: https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p

.. code-block:: console

    $ mkdir -p eir_auto_gp_tutorials/01_basic_tutorial/data
    $ mkdir -p eir_auto_gp_tutorials/tutorial_runs/01_basic_tutorial


The downloaded data should have the following structure:

.. literalinclude:: tutorials/tutorial_files/01_basic_tutorial/commands/input_folder.txt
    :language: console


The label file ID column must be called "ID". A sample label file would look like this:

.. literalinclude:: tutorials/tutorial_files/01_basic_tutorial/commands/label_file.txt
    :language: console

B - Training
------------

To process data and train models, run the following command:

.. literalinclude:: tutorials/tutorial_files/01_basic_tutorial/commands/AUTO_1.txt
    :language: console

The command above trains a model to predict CAD risk,
using genotype and clinical data as inputs to our models.
To adjust settings such as
the number of folds,
feature selection strategy,
or the GWAS p-value threshold,
refer to the :ref:`01-basic-tutorial`

After running the command,
the output will be written to a new folder in the
``eir_auto_gp_tutorials`` directory called ``tutorial_runs/01_basic_tutorial``.
You can go through these in the order of
``data -> modelling -> feature_selection -> analysis``
for a detailed overview of the outputs.

That's it for the quickstart, thank you for trying out ``EIR-auto-GP``!
Hopefully you find it useful and informative.