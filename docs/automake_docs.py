from collections.abc import Iterable
from itertools import chain

from docs.a_using_eir_auto_gp import (
    a_basic_tutorial,
    b_multiple_phenotypes,
    c_feature_selection_approaches,
)
from docs.doc_modules.experiments import AutoDocExperimentInfo, make_tutorial_data
from docs.doc_modules.generate_diagrams import generate_all


def _get_a_using_eir_auto_gp_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_basic_tutorial.get_experiments()
    b_experiments = b_multiple_phenotypes.get_experiments()
    c_experiments = c_feature_selection_approaches.get_experiments()

    return chain(
        a_experiments,
        b_experiments,
        c_experiments,
    )


if __name__ == "__main__":
    a_using_eir_experiments = _get_a_using_eir_auto_gp_experiments()

    experiment_iter = chain.from_iterable([a_using_eir_experiments])
    for experiment in experiment_iter:
        make_tutorial_data(auto_doc_experiment_info=experiment)

    generate_all()
