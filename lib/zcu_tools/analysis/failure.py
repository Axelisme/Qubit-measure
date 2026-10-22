"""Expected analysis failure shared by numerical owners and their frontends."""


class AnalysisFailed(Exception):
    """Report that analysis cannot produce a result for otherwise valid inputs.

    Numerical owners raise this for known optimizer nonconvergence or insufficient
    observations, preserving the cause when translating a backend error. Model
    callbacks, configuration, persistence and programming errors must propagate
    as their original exceptions. Frontends may turn this exception into an
    explicit failed value; they must not invent a successful fit.
    """
