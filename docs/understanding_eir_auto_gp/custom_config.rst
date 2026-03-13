Custom Configuration
====================

Advanced model and training parameters can be configured
via a YAML file passed with the ``--custom_config`` CLI argument.
When not specified, defaults are used.

Example usage:

.. code-block:: bash

    eir_auto_gp_multi_task \
        --genotype_data_path data/ \
        --label_file_path data/labels.csv \
        --global_output_folder runs/my_run \
        --output_con_columns trait_a trait_b \
        --custom_config my_config.yaml

Example YAML file:

.. code-block:: yaml

    use_fc0_skips: true
    use_lcl_to_output_skips: false
    use_lcl_fusion_skips: true
    fusion_model_type: mlp-residual-sum
    batch_size: 64
    optimize_model: true

Reference
---------

.. autoclass:: eir_auto_gp.multi_task.custom_config.CustomConfig
   :members:
   :undoc-members:
