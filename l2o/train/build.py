
import tensorflow as tf
from .curriculum import CurriculumLearning
from .. import problems


def deserialize_problem(p):
    if isinstance(p, problems.ProblemSpec):
        return p
    else:
        try:
            target, args, kwargs = p
            if type(target) == str:
                target = getattr(problems, target)
            return problems.ProblemSpec(target, args, kwargs)
        except Exception as e:
            raise TypeError(
                "Problem could not be deserialized: {}\n{}".format(p, e))

def build_curriculum(config):

    # Internal network
    net = config["constructor"](**config["net"])
    # Network has a ``.architecture`` bind linking to its corresponding arch.
    learner = config["constructor"].architecture(net, **config["loss"])
    # Fetch using keras API.
    optimizer = tf.keras.optimizers.get(config["optimizer"])
    # Deserialize problems
    problem_set = [
        p if isinstance(p, problems.ProblemSpec)
        else deserialize_problem(p)
        for p in config["problems"]
    ]
    curriculum = CurriculumLearning(
        learner, loss_args=config["training"],
        optimizer=optimizer, problems=problem_set, **config["curriculum"])

    return curriculum
