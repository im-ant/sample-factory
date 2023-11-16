from typing import Any, Callable

from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.utils.utils import log


# [cfg, env_info, policy_versions_tensor, policy_id, param_server] -> Learner
MakeLearnerFunc = Callable[[Config, Any, Any, Any, Any], Any]  # TODO: make this specific


class LearnerFactory:
    def __init__(self):
        """
        Custom functions for creating the learner.
        """
        self.make_learner_func: MakeLearnerFunc = default_make_learner_func

    def register_learner_factory(self, make_learner_func: MakeLearnerFunc):
        """
        Override the default learner function with a custom learner
        """
        log.debug(f"register_learner_factory: {make_learner_func}")
        self.make_learner_func = make_learner_func



def default_make_learner_func(cfg: Config, env_info, policy_versions_tensor, 
                              policy_id, param_server) -> Any:  # output Learner
    """
    The default make_learner_func, to be set inside of LearnerFactory to be 
    called inside of learner_worker 
    """
    # TODO 2023-11-16: make typing more explicit in arguments
    # NOTE: import here to avoid circular imports during init
    from sample_factory.algo.learning.learner import Learner
    return Learner(cfg, env_info, policy_versions_tensor, policy_id, 
                   param_server)
    

def create_learner(cfg: Config, env_info, policy_versions_tensor, 
                   policy_id, param_server) -> Any:  # output Learner
    # TODO 2023-11-16: make typing more explicit in arguments
    from sample_factory.algo.utils.context import global_learner_factory

    make_learner_func = global_learner_factory().make_learner_func
    return make_learner_func(cfg, env_info, policy_versions_tensor, 
                             policy_id, param_server)

