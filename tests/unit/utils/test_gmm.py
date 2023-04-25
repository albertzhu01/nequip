import torch
import numpy as np
from nequip.utils import gmm
from sklearn import mixture
from e3nn.util.test import assert_auto_jitable

# TODO: use pytest fixtures
seed = 678912345
fit_data_rng = torch.Generator().manual_seed(seed)

# normally distributed data
fit_data = 2 * (torch.randn(100, 8, generator=fit_data_rng) - 0.5)
gmm_sklearn = mixture.GaussianMixture(
    n_components=2, covariance_type="full", random_state=seed
)

gmm_torch = gmm.GaussianMixture(n_components=2, n_features=8)


class TestGMM:
    def test_compile(self):
        assert_auto_jitable(gmm_torch)

    def test_fit_forward_simple(self):
        gmm_sklearn.fit(fit_data.numpy())
        gmm_torch.fit(fit_data, rng=seed)

        print(f"sklearn means: {torch.from_numpy(gmm_sklearn.means_)}")
        print(f"torch means: {gmm_torch.means}")
        assert torch.allclose(torch.from_numpy(gmm_sklearn.means_), gmm_torch.means)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.covariances_), gmm_torch.covariances
        )

        assert torch.allclose(torch.from_numpy(gmm_sklearn.weights_), gmm_torch.weights)

        assert torch.allclose(
            torch.from_numpy(gmm_sklearn.precisions_cholesky_),
            gmm_torch.precisions_cholesky,
        )

        test_data_rng = np.random.RandomState(123456789)
        test_data = 2 * (test_data_rng.rand(100, 8) - 0.5)

        sklearn_nll = gmm_sklearn.score_samples(test_data)
        torch_nll = gmm_torch(torch.from_numpy(test_data))

        print(f"sklearn_nll shape: {sklearn_nll.shape}")
        print(f"torch_nll shape: {torch_nll.size()}")
        print(f"sklearn nll: {torch.from_numpy(sklearn_nll)}")
        print(f"torch nll: {torch_nll}")

        assert torch.allclose(
            torch.from_numpy(sklearn_nll),
            torch_nll,
        )