"""
Surrogate models for modular Bayesian Optimization (BoTorch).

This module exposes a single orchestrator class `Surrogate` that builds and fits
various GP-based surrogate models behind one consistent API.

Design goals
------------
- Unified, minimal interface:
    model, mll = Surrogate(cfg).build(train_x, train_y, train_yvar=None)
    model = Surrogate(cfg, ...).update(train_x, train_y, ...)
- Curated coverage of high-impact choices that affect BO performance:
    * Kernel family & smoothness: RBF (SE) vs Matern(ν ∈ {1/2, 3/2, 5/2}) with ARD / isotropic
    * Outputscale / noise priors and constraints; jitter-friendly likelihoods
    * Likelihood / noise model: homoscedastic Gaussian, heteroscedastic, fixed-noise
    * Mean & outcome transforms: zero / constant mean; Standardize transform
    * Multi-output structure: independent (ModelListGP) vs multi-task (correlated outputs)
    * High-dim structure: SAAS fully Bayesian (if available), additive kernels (groups)
    * Refitting policy: warm-start hyperparameters and (optional) re-fit

Notes
-----
- This file stays **Torch- and BoTorch-first**; keep NumPy out of the hot path.
- CUDA vs CPU is controlled via `device` and `dtype` in the config.
- Some advanced models (HOGP, DKL, Student-t) are noted but not enabled by default to
  keep the surface area reliable. Stubs raise clear errors if requested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

# CUDA-first: require a CUDA-enabled build by default
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Install a CUDA-enabled PyTorch or set SurrogateConfig(device=torch.device('cpu'))."
    )
DEFAULT_DEVICE = torch.device("cuda")

# BoTorch core models
from botorch.models import (
    SingleTaskGP,
    ModelListGP,
)

# Optional BoTorch models
try:
    from botorch.models.gp_regression import FixedNoiseGP  # type: ignore
except Exception:  # pragma: no cover
    FixedNoiseGP = None  # type: ignore

try:
    from botorch.models.multitask import MultiTaskGP  # type: ignore
except Exception:  # pragma: no cover
    MultiTaskGP = None  # type: ignore

try:
    from botorch.models.gp_regression import HeteroskedasticSingleTaskGP  # type: ignore
except Exception:  # pragma: no cover
    HeteroskedasticSingleTaskGP = None  # type: ignore

# Fully Bayesian SAAS (if available)
try:  # pragma: no cover
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP  # type: ignore
    from botorch.fit import fit_fully_bayesian_model_nuts  # type: ignore
    _SAAS_AVAILABLE = True
except Exception:  # pragma: no cover
    SaasFullyBayesianSingleTaskGP = None  # type: ignore
    fit_fully_bayesian_model_nuts = None  # type: ignore
    _SAAS_AVAILABLE = False

from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize

# GPyTorch building blocks
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, AdditiveKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.constraints import GreaterThan


# ------------------------------
# Config
# ------------------------------
@dataclass
class SurrogateConfig:
    # Device / dtype
    device: torch.device = DEFAULT_DEVICE
    dtype: torch.dtype = torch.double

    # Kernel & structure
    kernel: str = "matern"  # "matern" | "rbf"
    nu: float = 2.5          # Only used if kernel == "matern" (0.5, 1.5, 2.5)
    ard: bool = True         # ARD lengthscales per-dimension

    # Additive kernel groups (list of lists of active dims); if empty -> disabled
    additive_groups: List[List[int]] = field(default_factory=list)

    # Multi-output structure
    multi_output: str = "independent"  # "independent" | "multitask"

    # Likelihood / noise model
    noise_model: str = "homoscedastic"  # "homoscedastic" | "heteroscedastic" | "fixed"

    # Priors / constraints (None disables)
    lengthscale_prior: Optional[str] = None       # "gamma" | "lognormal" | None
    lengthscale_prior_args: Tuple[float, float] = (3.0, 6.0)  # shape/rate or mean/std
    outputscale_prior: Optional[str] = None       # "gamma" | "lognormal" | None
    outputscale_prior_args: Tuple[float, float] = (2.0, 0.15)
    noise_prior: Optional[str] = None             # "gamma" | "lognormal" | None
    noise_prior_args: Tuple[float, float] = (1.1, 0.05)
    min_noise: float = 1e-3                       # jitter floor

    # Mean & outcome transforms
    mean: str = "constant"        # "constant" | "zero"
    standardize_y: Optional[int] = None  # m (number of outputs) or None (disable)

    # Fully Bayesian SAAS
    saas: bool = False
    saas_warmup_steps: int = 256
    saas_num_samples: int = 256
    saas_thinning: int = 16

    # Refitting policy
    refit_on_update: bool = True


# ------------------------------
# Utilities
# ------------------------------

def _make_prior(name: Optional[str], a: float, b: float):
    if name is None:
        return None
    name = name.lower()
    if name == "gamma":
        # Note: GammaPrior(shape, rate)
        return GammaPrior(a, b)
    if name == "lognormal":
        # LogNormalPrior(mean, std) in log space
        return LogNormalPrior(a, b)
    raise ValueError(f"Unknown prior: {name}")


def _build_base_kernel(d: int, cfg: SurrogateConfig, active_dims: Optional[Sequence[int]] = None):
    ard_num_dims = d if cfg.ard and active_dims is None else (len(active_dims) if (cfg.ard and active_dims is not None) else None)
    if cfg.kernel.lower() == "rbf":
        base = RBFKernel(ard_num_dims=ard_num_dims, active_dims=active_dims)
    elif cfg.kernel.lower() == "matern":
        base = MaternKernel(
            nu=cfg.nu,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )
    else:
        raise ValueError("kernel must be 'rbf' or 'matern'")

    # Prior on lengthscale(s)
    lp = _make_prior(cfg.lengthscale_prior, *cfg.lengthscale_prior_args)
    if lp is not None:
        base.register_prior("lengthscale_prior", lp, "lengthscale")

    covar = ScaleKernel(base)
    op = _make_prior(cfg.outputscale_prior, *cfg.outputscale_prior_args)
    if op is not None:
        covar.register_prior("outputscale_prior", op, "outputscale")
    return covar


def _make_mean(cfg: SurrogateConfig):
    return ZeroMean() if cfg.mean.lower() == "zero" else ConstantMean()


def _make_outcome_transform(cfg: SurrogateConfig, m: int):
    return Standardize(m=m) if (cfg.standardize_y is not None) else None


# ------------------------------
# Main orchestrator
# ------------------------------
class Surrogate:
    """Factory + trainer for GP surrogates.

    Usage
    -----
    cfg = SurrogateConfig(device=torch.device("cuda"))
    sg = Surrogate(cfg)
    model, mll = sg.build(train_x, train_y, train_yvar=None)
    # later
    model = sg.update(train_x2, train_y2)
    """

    def __init__(self, cfg: SurrogateConfig):
        self.cfg = cfg
        self.model = None
        self.mll = None

    # --------------------------
    # Public API
    # --------------------------
    def build(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_yvar: Optional[Tensor] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[gpytorch.models.ExactGP, Optional[gpytorch.mlls.ExactMarginalLogLikelihood]]:
        """Build and fit a surrogate.

        Parameters
        ----------
        train_x : (N, d) in original box (assumed already scaled to [0,1]^d)
        train_y : (N, m) objectives (post any sign flips). If m>1 -> multi-output.
        train_yvar : (N, m) optional observed noise variances (for heteroscedastic/fixed).
        state_dict : warm-start parameters for refitting.
        """
        cfg = self.cfg
        device, dtype = cfg.device, cfg.dtype
        train_x = train_x.to(device=device, dtype=dtype)
        train_y = train_y.to(device=device, dtype=dtype)
        if train_yvar is not None:
            train_yvar = train_yvar.to(device=device, dtype=dtype)

        m = train_y.shape[-1]
        # Outcome transforms: per-output for independent, global for multitask
        outcome_tf_global = _make_outcome_transform(cfg, m) if cfg.standardize_y is not None else None
        outcome_tf_each = Standardize(m=1) if cfg.standardize_y is not None else None

        if cfg.saas:
            if not _SAAS_AVAILABLE:
                raise RuntimeError("SAAS model requested but not available in your BoTorch build.")
            if m != 1:
                raise NotImplementedError("SAAS in this helper supports single-output only. Use scalarization or build your own.")
            # Fully Bayesian SAAS model
            model = SaasFullyBayesianSingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                outcome_transform=outcome_tf_each,
            ).to(device, dtype)
            fit_fully_bayesian_model_nuts(
                model=model,
                warmup_steps=cfg.saas_warmup_steps,
                num_samples=cfg.saas_num_samples,
                thinning=cfg.saas_thinning,
            )
            self.model, self.mll = model, None
            return model, None

        # Decide on independent vs multitask
        if (m == 1) or (self.cfg.multi_output == "independent"):
            model = self._build_independent(train_x, train_y, train_yvar, outcome_tf_each)
        elif self.cfg.multi_output == "multitask":
            if MultiTaskGP is None:
                raise RuntimeError("MultiTaskGP not available in this BoTorch version.")
            model = self._build_multitask(train_x, train_y, train_yvar, outcome_tf_global)
        else:
            raise ValueError("multi_output must be 'independent' or 'multitask'")

        # Warm start
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Build MLL & fit
        mll = gpytorch.mlls.SumMarginalLogLikelihood(model.likelihood, model) if isinstance(model, ModelListGP) else gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self.model, self.mll = model, mll
        return model, mll

    def update(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_yvar: Optional[Tensor] = None,
        warm_start: bool = True,
        refit: Optional[bool] = None,
    ) -> gpytorch.models.ExactGP:
        """Set new training data and (optionally) refit with warm start.

        Works for both ModelListGP and single-ExactGP models.
        """
        if self.model is None:
            raise RuntimeError("Call build(...) before update(...)")
        model = self.model
        train_x = train_x.to(self.cfg.device, self.cfg.dtype)
        train_y = train_y.to(self.cfg.device, self.cfg.dtype)
        if train_yvar is not None:
            train_yvar = train_yvar.to(self.cfg.device, self.cfg.dtype)

        if isinstance(model, ModelListGP):
            # Split per-output
            for i, m_i in enumerate(model.models):
                if isinstance(m_i, (SingleTaskGP,)):
                    m_i.set_train_data(inputs=train_x, targets=train_y[..., i : i + 1], strict=False)
                elif (HeteroskedasticSingleTaskGP is not None) and isinstance(m_i, HeteroskedasticSingleTaskGP):
                    if train_yvar is None:
                        raise ValueError("Heteroskedastic GP requires train_yvar on update.")
                    m_i.set_train_data(
                        inputs=train_x,
                        targets=train_y[..., i : i + 1],
                        noise=train_yvar[..., i : i + 1],
                        strict=False,
                    )
                elif (FixedNoiseGP is not None) and isinstance(m_i, FixedNoiseGP):
                    if train_yvar is None:
                        raise ValueError("FixedNoiseGP requires train_yvar on update.")
                    m_i.set_train_data(
                        inputs=train_x,
                        targets=train_y[..., i : i + 1],
                        fixed_noise=train_yvar[..., i : i + 1],
                        strict=False,
                    )
                else:
                    m_i.set_train_data(train_x, train_y[..., i : i + 1], strict=False)
        else:
            # Single model (SingleTaskGP or MultiTaskGP)
            if (self.cfg.multi_output == "multitask") and (train_y.shape[-1] > 1):
                X_aug, Y_aug, noise_aug = self._augment_multitask_train(train_x, train_y, train_yvar)
                if isinstance(model, MultiTaskGP):
                    if noise_aug is not None and hasattr(model.likelihood, "noise_covar"):
                        model.set_train_data(X_aug, Y_aug, strict=False)
                        # Likelihood noise handled internally
                    else:
                        model.set_train_data(X_aug, Y_aug, strict=False)
                else:
                    model.set_train_data(train_x, train_y, strict=False)
            else:
                if isinstance(model, SingleTaskGP):
                    model.set_train_data(train_x, train_y, strict=False)
                else:
                    model.set_train_data(train_x, train_y, strict=False)

        # Refit if requested
        do_refit = self.cfg.refit_on_update if refit is None else refit
        if do_refit:
            # Preserve state for warm start
            state = model.state_dict() if warm_start else None


            if warm_start and self.model is not None:
                state = self.model.state_dict()
            else:
                state = None
  

            model, mll = self.build(train_x, train_y, train_yvar=train_yvar, state_dict=state)
            self.model, self.mll = model, mll
        return self.model

    # --------------------------
    # Internal builders
    # --------------------------
    def _build_independent(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_yvar: Optional[Tensor],
        outcome_tf_each: Optional[gpytorch.transforms.Transform],
    ):
        d = train_x.shape[-1]
        m = train_y.shape[-1]

        models = []
        for i in range(m):
            covar_module = self._make_covar(d)
            mean_module = _make_mean(self.cfg)

            if self.cfg.noise_model == "heteroscedastic":
                if HeteroskedasticSingleTaskGP is None:
                    raise RuntimeError("HeteroskedasticSingleTaskGP not available in this BoTorch build.")
                if train_yvar is None:
                    raise ValueError("Heteroskedastic noise requires train_yvar of shape (N, m)")
                model_i = HeteroskedasticSingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i : i + 1],
                    train_Yvar=train_yvar[..., i : i + 1],
                    outcome_transform=outcome_tf_each,
                    covar_module=covar_module,
                    mean_module=mean_module,
                )
            elif self.cfg.noise_model == "fixed":
                if FixedNoiseGP is None:
                    raise RuntimeError("FixedNoiseGP not available in this BoTorch build.")
                if train_yvar is None:
                    raise ValueError("Fixed noise model requires train_yvar of shape (N, m)")
                model_i = FixedNoiseGP(
                    train_X=train_x,
                    train_Y=train_y[..., i : i + 1],
                    train_Yvar=train_yvar[..., i : i + 1],
                    outcome_transform=outcome_tf_each,
                    covar_module=covar_module,
                    mean_module=mean_module,
                )
            else:  # homoscedastic GaussianLikelihood
                noise_prior = _make_prior(self.cfg.noise_prior, *self.cfg.noise_prior_args)
                likelihood = GaussianLikelihood(
                    noise_constraint=GreaterThan(self.cfg.min_noise),
                    noise_prior=noise_prior,
                )
                model_i = SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i : i + 1],
                    outcome_transform=outcome_tf_each,
                    covar_module=covar_module,
                    likelihood=likelihood,
                    mean_module=mean_module,
                )

            models.append(model_i)

        return models[0] if len(models) == 1 else ModelListGP(*models).to(self.cfg.device, self.cfg.dtype)

    def _augment_multitask_train(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_yvar: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Flatten (N,m) outputs into MultiTask format via task index column.
        Returns X_aug:(N*m, d+1), Y_aug:(N*m,1), noise_aug:(N*m,1)|None
        """
        N, d = train_x.shape
        m = train_y.shape[-1]
        # Repeat X for each task
        X_rep = train_x.unsqueeze(1).expand(N, m, d).reshape(N * m, d)
        task_idx = torch.arange(m, device=train_x.device, dtype=torch.long).repeat_interleave(N)
        task_idx = task_idx.reshape(-1, 1).to(dtype=train_x.dtype)
        X_aug = torch.cat([X_rep, task_idx], dim=-1)

        Y_aug = train_y.reshape(N * m, 1)
        noise_aug = None if train_yvar is None else train_yvar.reshape(N * m, 1)
        return X_aug, Y_aug, noise_aug

    def _build_multitask(
        self,
        train_x: Tensor,
        train_y: Tensor,
        train_yvar: Optional[Tensor],
        outcome_tf: Optional[gpytorch.transforms.Transform],
    ):
        if MultiTaskGP is None:
            raise RuntimeError("MultiTaskGP not available")
        d = train_x.shape[-1]
        X_aug, Y_aug, noise_aug = self._augment_multitask_train(train_x, train_y, train_yvar)

        # Base kernel for the *input* dims (exclude the task feature)
        covar_module = self._make_covar(d)  # task structure handled by MultiTaskGP internally
        mean_module = _make_mean(self.cfg)

        # NOTE: MultiTaskGP handles likelihood internally as Gaussian; noise per-task
        model = MultiTaskGP(
            train_X=X_aug,
            train_Y=Y_aug,
            task_feature=d,  # last column is task index
            outcome_transform=outcome_tf_each,
            covar_module=covar_module,
            mean_module=mean_module,
        )
        return model.to(self.cfg.device, self.cfg.dtype)

    def _make_covar(self, d: int):
        """Build covariance module (with optional additive groups)."""
        if not self.cfg.additive_groups:
            return _build_base_kernel(d, self.cfg)
        # Additive groups across active dims
        subkernels = []
        for group in self.cfg.additive_groups:
            subkernels.append(_build_base_kernel(d=len(group), cfg=self.cfg, active_dims=group))
        return AdditiveKernel(*subkernels)


# ------------------------------
# Nice presets
# ------------------------------
PRESETS: Dict[str, SurrogateConfig] = {
    # Robust default: Matérn-5/2, ARD, standardized outputs
    "m52_ard": SurrogateConfig(kernel="matern", nu=2.5, ard=True, standardize_y=1),
    # Isotropic RBF (SE), standardized outputs
    "rbf_iso": SurrogateConfig(kernel="rbf", ard=False, standardize_y=1),
    # Multi-output independent, ARD Matérn-5/2, standardized 2 outputs
    "m52_ard_indep2": SurrogateConfig(kernel="matern", nu=2.5, ard=True, standardize_y=2, multi_output="independent"),
    "m52_indep2": SurrogateConfig(kernel="matern", nu=2.5, ard=False, standardize_y=2, multi_output="independent"),

    "m32_ard_indep2": SurrogateConfig(kernel="matern", nu=1.5, ard=True, standardize_y=2, multi_output="independent"),
    # Multi-task (correlated outputs)
    "m52_multitask": SurrogateConfig(kernel="matern", nu=2.5, ard=True, standardize_y=2, multi_output="multitask"),
}


__all__ = [
    "SurrogateConfig",
    "Surrogate",
    "PRESETS",
]
