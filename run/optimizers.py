from typing import Iterable, Union

import torch


class AdaptiveGradientNormClipper:
    """Clip gradient norm based on the average gradient norm.

    Source (On the difficulty of training Recurrent Neural Networks):
        The proposed clipping is simple to implement and computationally efficient, but it does
        however introduce an additional hyper-parameter, namely the threshold. One good heuristic
        for setting this threshold is to look at statistics on the average norm over a sufficiently
        large number of updates. In our experiments we have noticed that for a given task and model
        size, training is not very sensitive to this hyperparameter and the algorithm behaves well
        even for rather small thresholds.

    Args:
        parameters: A reference to a parameter list.
        norm_type: The "p" in p-norm. This includes `inf` for infinity norm.
    """

    # number of updates kept in history.
    # seems like enough data to get meaningful stats and also align with
    # typical mini-batch sizes.

    GRADIENT_UPDATES_HISTORY_MAXLENGTH  = 32

    def __init__(self, parameters: Union[torch.Tensor,Iterable[torch.Tensor]], norm_type: float):
        super().__init__()

        if isinstance(parameters, torch.Tensor):
            self.parameters = [parameters]
        else:
            self.parameters = parameters

        self.norm_type = float(norm_type)

        # we maintain a history of the gradient norms for the last 32 updates
        # this will allow us to compute a simple moving average of gradient norms
        self.grad_norm_history = torch.empty([0])

        # gradient norm average over history, initialized at highest value.
        # this will start updating when num_updates gradient updates have been performed.
        # the max_norm is clamped to 1 by clip_grad_norm_ during history prefill,
        # so using infinity here is not a problem.
        self.grad_norm_average = float("inf")

        # gradient norm standard deviation over history.
        # we will scale down gradients whenever their norm is 1 standard deviation greater than
        # the average (~16% of the time assuming those are normally distributed).
        self.grad_norm_std = 0

    def clip(self) -> torch.Tensor:
        """Clips gradient norm of an iterable of `self.parameters`, and update gradient norm
        history.

        Returns:
            Total norm of the unscaled parameters gradients (viewed as a single vector).
        """
        # defensive measure, in-place imputation of NaN, inf and -inf in parameter gradients
        for p in self.parameters:
            g = p.grad.detach()
            torch.nan_to_num_(g, nan=0.0, posinf=1e6, neginf=-1e6)

        # always use the norm clipping method as our standard way to collect total norm
        # and benefit from multi-device implementation
        # the returned norm is __before__ clipping
        # we let it raise if the total grad norm is inf, -inf, or nan but that should not happen
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters,
            self.grad_norm_average + self.grad_norm_std,
            self.norm_type,
            error_if_nonfinite=True,
        )

        # append total gradients norm to history
        self.grad_norm_history = torch.cat(
            (self.grad_norm_history, torch.tensor([total_grad_norm]))
        )

        # remove oldest gradient norm once num_updates updates have been performed
        if len(self.grad_norm_history) > self.GRADIENT_UPDATES_HISTORY_MAXLENGTH:
            self.grad_norm_history = self.grad_norm_history[1:]

        # we have a full history, let's compute average gradient norm
        # and its standard deviation
        if len(self.grad_norm_history) == self.GRADIENT_UPDATES_HISTORY_MAXLENGTH:
            self.grad_norm_average = torch.mean(self.grad_norm_history).item()
            self.grad_norm_std = torch.std(self.grad_norm_history).item()

        return total_grad_norm
