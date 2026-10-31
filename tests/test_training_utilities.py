import importlib.util
import os
import tempfile
import unittest
from unittest import mock


HAS_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "numpy", "scipy", "h5py", "matplotlib", "torchinfo", "tqdm")
)

if HAS_DEPENDENCIES:
    import torch

    import helm
    from models import MgNO_helm, MgNO_helm2
    from navier import test as evaluate_navier
    from utilities3 import GaussianNormalizer, HSloss_d, HsLoss, LpLoss, getSavePath


@unittest.skipUnless(HAS_DEPENDENCIES, "training dependencies are not installed")
class TrainingUtilityTests(unittest.TestCase):
    def test_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                path = getSavePath("helm", "MgNO", "para")
                self.assertTrue(os.path.isdir(os.path.dirname(path)))
                self.assertTrue(path.endswith(".pt"))
            finally:
                os.chdir(previous)

    def test_global_normalizer_buffers_and_round_trip(self):
        x = torch.arange(8, dtype=torch.float32)
        normalizer = GaussianNormalizer(x)
        self.assertEqual(set(dict(normalizer.named_buffers())), {"mean", "std"})
        torch.testing.assert_close(normalizer.decode(normalizer.encode(x)), x)

    def test_helmholtz_loss_is_summed_per_sample(self):
        x = torch.zeros(2, 4, 4)
        y = torch.stack((torch.ones(4, 4), 2 * torch.ones(4, 4)))
        loss = HSloss_d(reduction="sum")
        torch.testing.assert_close(loss(x, y)[0], loss(x[:1], y[:1])[0] + loss(x[1:], y[1:])[0])

    def test_sobolev_higher_order_and_balanced_modes(self):
        x = torch.rand(2, 8, 8) + 1
        y = torch.rand(2, 8, 8) + 1
        for balanced in (False, True):
            loss, l2, _, _ = HsLoss(k=2, group=balanced, res=8)(x, y)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(l2))

    def test_helmholtz_normalizer_preserves_single_sample_batch(self):
        normalizer = GaussianNormalizer(torch.randn(2, 101, 101))
        for model_type, iterations in (
            (MgNO_helm, [1, 1, 1, 1, 1]),
            (MgNO_helm2, [[1, 0]] * 5),
        ):
            model = model_type(
                num_layer=1, num_channel_u=1, num_channel_f=1,
                num_classes=1, num_iteration=iterations, normalizer=normalizer,
            )
            with torch.no_grad():
                output = model(torch.randn(1, 1, 101, 101))
            self.assertEqual(tuple(output.shape), (1, 101, 101))

    def test_navier_evaluation_runs_on_cpu_and_sets_eval_mode(self):
        model = torch.nn.Conv2d(1, 1, 1)
        x = torch.ones(2, 1, 4, 4)
        y = torch.ones(2, 2, 4, 4)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=1
        )
        result = evaluate_navier(model, "MgNO", LpLoss(size_average=False),
                                 loader, 0, 0, 0, {"T": 2, "step": 1})
        self.assertEqual(len(result), 3)
        self.assertFalse(model.training)

    def test_helm_training_uses_validation_split_size(self):
        class Progress:
            def __init__(self, **kwargs):
                self.descriptions = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def set_description(self, description):
                self.descriptions.append(description)

            def update(self):
                pass

        class Scheduler:
            def step(self):
                pass

        progress = Progress()
        train_x, train_y = torch.ones(2, 4, 4), torch.ones(2, 4, 4)
        test_x, test_y = torch.ones(3, 4, 4), torch.ones(3, 4, 4)
        val_x, val_y = torch.ones(1, 4, 4), torch.ones(1, 4, 4)
        model = torch.nn.Conv2d(1, 1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        data = {"data": "helm", "batch_size": 2, "loss_type": "l2",
                "dataSize": {"train": range(2), "test": range(3), "val": range(1)}}
        dataset = (train_x, train_y, test_x, test_y, val_x, val_y, None, None)
        with mock.patch.object(helm, "getHelmDataset", return_value=dataset), \
             mock.patch.object(helm, "MgNO_helm", return_value=model), \
             mock.patch.object(helm, "getOptimizerScheduler", return_value=(optimizer, Scheduler())), \
             mock.patch.object(helm, "getSavePath", return_value="unused"), \
             mock.patch.object(helm, "tqdm", return_value=progress):
            result = helm.objective(data, {"normalizer": False}, {"epochs": 1})
        self.assertAlmostEqual(result, 1.0)
        self.assertIn("val l2 loss: 1.000e+00", progress.descriptions[0])
