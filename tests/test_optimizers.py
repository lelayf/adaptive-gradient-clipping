from run.optimizers import AdaptiveGradientNormClipper
import torch
from torch import inf
from pytest import approx

def test_agc_early_history_norm_stats_compute():
    """Test that `AdaptiveGradientNormClipper` leaves the std of gradient norms
    untouched as long as the history length is insufficient."""

    # init model parameters (vector of size 1)
    # we use a very simple tensor shape to make the deterministic
    # computation of gradient norm stats more practical for this test
    z = torch.tensor([100.0])

    # initialize clipper for L2 norm
    agc = AdaptiveGradientNormClipper([z], 2)

    # prefill history, up to capacity minus one (buffer not full)
    # we simply increment the norm by 1 at each update
    for i in range(agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH - 1):
        x = torch.tensor([float(i)])
        z.grad = x
        agc.clip()

    # up to this point, our avg gradient norm is 15 and its std deviation is 8.94
    # those values should not be reflected yet in our stats tracking
    assert agc.grad_norm_average == float(
        "inf"
    ), "gradient norm average not computed because buffer is not full"

    assert (
        agc.grad_norm_std == 0
    ), "gradient norm standard deviaton not computed because buffer is not full"

    # perform a single gradient update - after this our history will reach its capacity
    # and grad norm stats should be computed for the first time
    x = torch.tensor([31.0])
    z.grad = x
    agc.clip()

    assert (
        agc.grad_norm_average == 15.5
    ), "gradient norm average computed because history is now fully initialized"

    assert agc.grad_norm_std == approx(
        9.3808315
    ), "gradient norm standard deviaton computed because history is now fully initialized"


def test_agc_early_history_prefill():
    """Test that `AdaptiveGradientNormClipper` gradients norm early history length
    is consistent with the number of performed updates."""
    
    torch.manual_seed(0)

    # init model parameters (vector of size 3)
    z = torch.normal(2, 1, size=(3,)).clamp(0.2) * 10

    # init model parameters gradients
    # this is a vector of size 3, one order of magnitude lower than params
    x = torch.normal(2, 1, size=(3,)).clamp(0.2)
    z.grad = x

    # initialize clipper for L2 norm
    agc = AdaptiveGradientNormClipper([z], 2)

    # do not reach history capacity
    for i in range(agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH - 2):
        x = torch.normal(2, 1, size=(3,)).clamp(0.2)
        z.grad = x
        agc.clip()

    assert (
        len(agc.grad_norm_history) == agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH - 2
    ), "history length consistent with number of gradient updates"


def test_agc_history_drop_oldest():
    """Test that `AdaptiveGradientNormClipper` gradients norm history drops
    its oldest value like a FIFO queue, once it has reached its full capacity
    and subsequent updates are performed
    """
    
    torch.manual_seed(0)

    # init model parameters (vector of size 3)
    z = torch.normal(2, 1, size=(3,)).clamp(0.2) * 10

    # init model parameters gradients
    # this is a vector of size 3, one order of magnitude lower than params
    x = torch.normal(2, 1, size=(3,)).clamp(0.2)
    z.grad = x

    # initialize clipper for L2 norm
    agc = AdaptiveGradientNormClipper([z], 2)

    for i in range(agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH + 1):
        x = torch.normal(2, 1, size=(3,)).clamp(0.2)
        z.grad = x
        agc.clip()

    assert (
        len(agc.grad_norm_history) == agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH
    ), "history length consistent with number of gradient updates"


def test_agc_clipping():
    """Test that `AdaptiveGradientNormClipper` gradients norm is scaled down
    when its norm is greater than 1 std above its historical average, and
    not scaled when it is below.
    """

    torch.manual_seed(0)

    # init model parameters (vector of size 3)
    z = torch.normal(2, 1, size=(3,)).clamp(0.2) * 10

    # init model parameters gradients
    # this is a vector of size 3, one order of magnitude lower than params
    x = torch.normal(2, 1, size=(3,)).clamp(0.2)
    z.grad = x

    # initialize clipper for L2 norm
    agc = AdaptiveGradientNormClipper([z], 2)

    # let gradient norms history update for 2 "capacity cycles"
    for i in range(agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH * 2):
        x = torch.normal(2, 1, size=(3,)).clamp(0.2)
        z.grad = x
        agc.clip()

    # craft a gradient bigger than our lambda threshold heuristic
    threshold = agc.grad_norm_average + agc.grad_norm_std
    x = torch.ones((3,)) * torch.tensor([threshold + 1]).div_(
        torch.sqrt(torch.tensor([3]))
    )
    z.grad = x

    pre_clipping_grad_norm = agc.clip()
    post_clipping_grad_norm = torch.norm(z.grad)

    assert (
        post_clipping_grad_norm < pre_clipping_grad_norm
    ), "Gradients were scaled down"

    # craft a gradient smaller than our lambda threshold heuristic
    threshold = agc.grad_norm_average + agc.grad_norm_std
    x = torch.ones((3,)) * torch.tensor([threshold - 0.1]).div_(
        torch.sqrt(torch.tensor([3]))
    )
    z.grad = x

    pre_clipping_grad_norm = agc.clip()
    post_clipping_grad_norm = torch.norm(z.grad)

    assert (
        post_clipping_grad_norm == pre_clipping_grad_norm
    ), "Gradients were not scaled down"


def test_agc_clipping_with_inf_values():
    """Test that `AdaptiveGradientNormClipper` gradients norm provides some exception
    handling for inf, -inf, and Nan values, on top of its default imputation strategy.
    """
    
    torch.manual_seed(0)

    # init model parameters (vector of size 3)
    z = torch.normal(2, 1, size=(3,)).clamp(0.2) * 10

    # init model parameters gradients
    # this is a vector of size 3, one order of magnitude lower than params
    x = torch.normal(2, 1, size=(3,)).clamp(0.2)
    z.grad = x

    # initialize clipper for L2 norm
    agc = AdaptiveGradientNormClipper([z], 2)

    # let gradient norms history update for 2 "capacity cycles"
    for i in range(agc.GRADIENT_UPDATES_HISTORY_MAXLENGTH * 2):
        x = torch.normal(2, 1, size=(3,)).clamp(0.2)
        z.grad = x
        agc.clip()

    # build a gradient with inf values
    x = torch.ones((3,)) + torch.tensor([inf, 1, 1])
    z.grad = x

    # this was raised before we added defensive imputation in clipper
    try:
        agc.clip()
    except RuntimeError as e:
        err = "The total norm of order 2.0 for gradients from `parameters` is non-finite, "
        err += "so it cannot be clipped. To disable this error and scale the gradients by the "
        err += "non-finite norm anyway, set `error_if_nonfinite=False`"
        assert str(e) == err

    # post-imputation, above exception does not raise anymore
    # and gradient norm is finite
    assert torch.isfinite(z.grad.norm()).item() is True
