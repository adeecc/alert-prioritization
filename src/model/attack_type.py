import numpy as np
import numpy.typing as npt


class AttackType:
    """Representation of an attack type."""

    def __init__(
        self,
        loss: npt.NDArray[np.float],
        cost: float,
        pr_alert: npt.NDArray[np.float],
        name: str,
    ):
        """
        Construct an attack type object.
        :param loss: Loss inflicted by an undetected attack of this type for various ages (i.e., L_{h,a}). Single-dimensional list of floats, length equal to the time horizon.
        :param cost: Cost of mounting an attack of this type (i.e., E_a).
        :param pr_alert: Probability of triggering an alert for various alert types (i.e., P_{a,t}). Single-dimensional list of floats, length equal to the number of alert types.
        :param name: Name of this attack type.
        """
        assert cost > 0
        assert np.min(loss) >= 0
        assert np.min(pr_alert) >= 0.0
        assert np.max(pr_alert) <= 1.0

        self.loss = loss
        self.cost = cost
        self.pr_alert = pr_alert
        self.name = name
