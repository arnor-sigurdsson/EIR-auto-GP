from itertools import chain
from typing import Iterable

from docs.a_using_eir_auto_gp import a_basic_tutorial
from docs.doc_modules.experiments import make_tutorial_data, AutoDocExperimentInfo


def _get_a_using_eir_experiments() -> Iterable[AutoDocExperimentInfo]:
    a_experiments = a_basic_tutorial.get_experiments()

    return chain(
        a_experiments,
    )


if __name__ == "__main__":
    a_using_eir_experiments = _get_a_using_eir_experiments()

    experiment_iter = chain.from_iterable([a_using_eir_experiments])
    for experiment in experiment_iter:
        make_tutorial_data(auto_doc_experiment_info=experiment)
