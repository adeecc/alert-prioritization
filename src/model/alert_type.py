from model.distributions import Distribution, PoissonDistribution
from typing import Optional


class AlertType:
    """Representation of an alert type."""

    def __init__(
        self,
        cost: float = 1.0,
        false_alerts: Distribution = PoissonDistribution(),
        name: Optional[str] = None,
    ):
        """
        Construct an alert type object.
        :param cost: Cost of investigating an alert of this type (i.e., C_t).
        :param false_alerts: Distribution of false alerts of this type (i.e., F_t).
        :param name: Name of this alert type.
        """
        assert cost > 0
        self.cost = cost
        self.false_alerts = false_alerts
        self.name = name
