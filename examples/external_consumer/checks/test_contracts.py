"""Run with unittest; imports only the relocated consumer and public framework APIs."""

import ast
import copy
import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from mexina_external_example import demo
from mexina_external_example.definitions import definitions
from mexina_external_example.method import TINY_CONV, TINY_RESIDUAL, Reconstruction, TinyNetwork
from mexina_external_example.metric import SCALED_MAX_ERROR

from virtual_staining.checkpoint_contract import CheckpointCompatibilityError
from virtual_staining.config.run import RunConfig
from virtual_staining.definitions import DefinitionNotAvailableError
from virtual_staining.evaluation.evaluator import (
    EvaluationCoverageError,
    EvaluationSample,
    evaluate_samples,
)
from virtual_staining.inference.runner import load_inference_generator
from virtual_staining.inference.single import (
    InferenceRuntime,
    PredictionContract,
    run_image_path_inference,
)
from virtual_staining.methods.builtin import builtin_definitions
from virtual_staining.metrics import resolve_metrics
from virtual_staining.training.trainer import Trainer


class ConsumerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mexina-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        self.definitions = definitions()

    def test_definitions_config_and_strict_options(self):
        raw = demo.mapping(self.root)
        config = RunConfig.from_mapping(raw, self.definitions)
        resolved = config.to_dict()
        path = self.root / "run.yaml"
        path.write_text(yaml.safe_dump(resolved))
        self.assertEqual(RunConfig.from_yaml(path, self.definitions).to_dict(), resolved)
        self.assertEqual(RunConfig.from_mapping(resolved, self.definitions).to_dict(), resolved)
        self.assertIs(config.method.options.network.definition, TINY_CONV)
        self.assertNotIn("external_reconstruction", builtin_definitions().methods)
        with self.assertRaises(TypeError):
            self.definitions.methods["bad"] = Reconstruction()
        no_method = builtin_definitions().extend(metrics=[SCALED_MAX_ERROR])
        no_component = no_method.extend(methods=[Reconstruction()])
        for supplied in (no_method, no_component):
            with self.subTest(supplied=supplied), self.assertRaises(DefinitionNotAvailableError):
                RunConfig.from_mapping(raw, supplied)
        for keyword, values in (
            ("methods", [Reconstruction()]),
            ("components", [TINY_CONV]),
            ("metrics", [SCALED_MAX_ERROR]),
        ):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(ValueError, "Duplicate"):
                self.definitions.extend(**{keyword: values})
        for options in (
            {"width": 0},
            {"width": True},
            {"width": 2.5},
            {"learning_rate": float("nan")},
            {"learning_rate": True},
            {"learning_rate": -1},
            {"class_path": "os.system"},
            {"module": "os"},
            {"generator": {}},
            {"architecture": "resnet"},
        ):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                RunConfig.from_mapping(demo.mapping(self.root, **options), self.definitions)
        for name in ("unknown", "os.system", "mexina_external_example.method.TinyNetwork"):
            with self.subTest(name=name), self.assertRaises(DefinitionNotAvailableError):
                RunConfig.from_mapping(demo.mapping(self.root, architecture=name), self.definitions)
        for section, key in (
            ("model", "generator"),
            ("model", "discriminator"),
            ("training", "lr_g"),
            ("training", "lr_d"),
            ("training", "losses"),
            ("training", "scheduler"),
            ("method", "class_path"),
            ("method", "replay_buffer_size"),
        ):
            invalid = copy.deepcopy(raw)
            invalid[section][key] = {}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unknown key"):
                RunConfig.from_mapping(invalid, self.definitions)
        invalid = copy.deepcopy(raw)
        invalid["model"]["outputs"] = ["HE", "PAS"]
        with self.assertRaisesRegex(ValueError, "example method requires exactly one"):
            RunConfig.from_mapping(invalid, self.definitions)
        for key in ("generator", "discriminator", "lr_g", "lr_d", "replay_buffer", "scheduler"):
            self.assertNotIn(key, json.dumps(resolved))

    def test_both_architectures_train_reconstruct_and_resume(self):
        for architecture in ("tiny_conv", "tiny_residual"):
            with self.subTest(architecture=architecture):
                root = self.root / architecture
                config, layout, runtime = demo.standalone_train(root, architecture)
                modules = [v for v in vars(runtime).values() if isinstance(v, torch.nn.Module)]
                self.assertEqual([type(v) for v in modules], [TinyNetwork, torch.nn.L1Loss])
                self.assertEqual(
                    sum(isinstance(v, torch.optim.Optimizer) for v in vars(runtime).values()), 1
                )
                self.assertEqual(runtime.network.body is None, architecture == "tiny_conv")
                payload = torch.load(layout.checkpoints_dir / "ep000.pth", weights_only=True)
                self.assertEqual(payload["format_version"], 4)
                self.assertEqual(payload["method"]["name"], "external_reconstruction")
                self.assertEqual(
                    payload["method"]["implementation"],
                    {"source": "mexina_external_example", "version": "1"},
                )
                self.assertEqual(
                    payload["method"]["components"]["network"],
                    {
                        "name": architecture,
                        "source": "mexina_external_example",
                        "version": "1",
                        "options": {"width": 4},
                    },
                )
                self.assertEqual(payload["method"]["inputs"], ["LF", "AF"])
                self.assertEqual(payload["method"]["outputs"], ["HE"])
                self.assertEqual(payload["image_size"], [8, 8])
                self.assertEqual(set(payload["state"]), {"network", "optimizer"})
                with layout.epochs_csv.open() as handle:
                    history = list(csv.DictReader(handle))
                self.assertTrue(float(history[0]["loss_reconstruction_val"]) > 0)
                best = json.loads(layout.checkpoint_selection.read_text())
                selected = best["metrics"]["loss_reconstruction_val"]
                self.assertEqual(selected["mode"], "min")
                self.assertEqual(selected["best"]["objective_metadata"], {"objective": "paired_l1"})
                model, _ = load_inference_generator(
                    config, layout, demo.CPU, layout.checkpoints_dir / "ep000.pth"
                )
                batch = next(iter(demo.caller_loader()))
                torch.testing.assert_close(
                    model(batch["inputs"])["HE"], runtime.network(batch["inputs"])["HE"]
                )
                # Resume a new runtime and continue an actual second epoch.
                raw = demo.mapping(root, architecture=architecture)
                raw["training"]["epochs"] = 2
                resumed_config = RunConfig.from_mapping(raw, self.definitions)
                resumed = resumed_config.method.definition.build_training_runtime(
                    resumed_config, demo.CPU, seed=999
                )
                loader = demo.caller_loader()
                trainer = Trainer(
                    resumed_config.training, layout, resumed, loader, loader, demo.CPU
                )
                self.assertEqual(trainer.resume("latest"), 1)
                for key, value in runtime.network.state_dict().items():
                    torch.testing.assert_close(value, resumed.network.state_dict()[key])
                for key, state in runtime.optimizer.state_dict()["state"].items():
                    for field, value in state.items():
                        torch.testing.assert_close(
                            value, resumed.optimizer.state_dict()["state"][key][field]
                        )
                trainer.train(seed=17, start_epoch=1)
                self.assertTrue((layout.checkpoints_dir / "ep001.pth").is_file())
                self.assertFalse(layout.metadata_dir.exists())
                self.assertFalse(layout.config_dir.exists())

    def test_checkpoint_rejects_missing_or_changed_identity_before_loading_state(self):
        config, layout, _ = demo.standalone_train(self.root / "trained", "tiny_residual")
        checkpoint = layout.checkpoints_dir / "ep000.pth"
        raw = demo.mapping(self.root)
        raw.pop("method")
        raw.pop("training")
        raw.pop("evaluation")
        builtin_config = RunConfig.from_mapping(raw)
        with self.assertRaisesRegex(DefinitionNotAvailableError, "method.*not available"):
            load_inference_generator(builtin_config, layout, demo.CPU, checkpoint)
        supplied = builtin_definitions().extend(
            methods=[Reconstruction()], components=[TINY_CONV], metrics=[SCALED_MAX_ERROR]
        )
        missing = RunConfig.from_mapping(demo.mapping(self.root), supplied)
        with self.assertRaisesRegex(DefinitionNotAvailableError, "tiny_residual.*not available"):
            load_inference_generator(missing, layout, demo.CPU, checkpoint)
        for options in (
            {"architecture": "tiny_conv"},
            {"architecture": "tiny_residual", "width": 5},
        ):
            changed = RunConfig.from_mapping(demo.mapping(self.root, **options), self.definitions)
            runtime = changed.method.definition.build_training_runtime(changed, demo.CPU, seed=17)
            before = {k: v.clone() for k, v in runtime.network.state_dict().items()}
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(CheckpointCompatibilityError, "method.components.network"),
            ):
                load_inference_generator(changed, layout, demo.CPU, checkpoint)
            loader = demo.caller_loader()
            with self.assertRaises(CheckpointCompatibilityError):
                Trainer(changed.training, layout, runtime, loader, loader, demo.CPU).resume(
                    checkpoint
                )
            for key, value in before.items():
                torch.testing.assert_close(value, runtime.network.state_dict()[key])
        changed_version = builtin_definitions().extend(
            methods=[Reconstruction()],
            components=[replace(TINY_RESIDUAL, version="2")],
            metrics=[SCALED_MAX_ERROR],
        )
        changed = RunConfig.from_mapping(config.to_dict(), changed_version)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "network.version"):
            load_inference_generator(changed, layout, demo.CPU, checkpoint)
        # A real importable provider with a side effect is metadata only, never executed.
        provider = Path.cwd() / "provider_trap.py"
        provider.write_text("raise AssertionError('checkpoint imported provider code')\n")
        self.addCleanup(provider.unlink)
        payload = torch.load(checkpoint, weights_only=True)
        payload["method"]["implementation"]["source"] = "provider_trap"
        poisoned = self.root / "metadata-only.pth"
        torch.save(payload, poisoned)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "implementation.source"):
            load_inference_generator(config, layout, demo.CPU, poisoned)
        self.assertNotIn("provider_trap", sys.modules)

    def test_standalone_preparation_and_composed_workflow(self):
        evidence = demo.composed(self.root / "composed")
        self.assertTrue(all(Path(p).is_file() for p in evidence["inference"]))
        self.assertTrue(Path(evidence["manifest"]).is_file())
        self.assertTrue(Path(evidence["checkpoint"]).is_file())
        report = json.loads(Path(evidence["evaluation"]).read_text())
        self.assertEqual(report["metrics"][0]["name"], "scaled_max_error")
        self.assertEqual(report["metrics"][0]["options"], {"scale": 2.0})

    def test_predictor_only_and_invalid_outputs(self):
        paths = demo.predictor_only(self.root / "predictor")
        contract = PredictionContract(("LF", "AF"), ("HE",), (8, 8))
        bad_outputs = [
            ({"wrong": torch.zeros(1, 3, 8, 8)}, "outputs"),
            ({"HE": torch.zeros(1, 3, 7, 8)}, "shape"),
            ({"HE": torch.full((1, 3, 8, 8), float("nan"))}, "NaN or Inf"),
            ({"HE": torch.full((1, 3, 8, 8), float("inf"))}, "NaN or Inf"),
            ({"HE": torch.full((1, 3, 8, 8), 1.1)}, "must lie"),
            ({"HE": torch.full((1, 3, 8, 8), -1.1)}, "must lie"),
        ]
        for index, (outputs, message) in enumerate(bad_outputs):
            runtime = InferenceRuntime(lambda inputs, outputs=outputs: outputs, contract, demo.CPU)
            destination = self.root / f"invalid-{index}.png"
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                run_image_path_inference(runtime, paths, destination)
            self.assertFalse(destination.exists())

    def test_explicit_metric_reports_and_failures(self):
        root = self.root / "evaluation"
        result = demo.explicit_evaluation(root)
        metadata = json.loads(result.result_json.read_text())
        identity = metadata["metrics"][0]
        for key, value in {
            "name": "scaled_max_error",
            "version": "1",
            "source": "mexina_external_example",
            "options": {"scale": 2.0},
        }.items():
            self.assertEqual(identity[key], value)
        self.assertEqual([metric.name for metric in result.metrics], ["scaled_max_error", "mae"])
        self.assertTrue(np.isclose(result.rows[0]["scaled_max_error"], 4 / 255))
        for options in (
            {"scale": 0},
            {"scale": True},
            {"scale": "2"},
            {"scale": float("inf")},
            {"unknown": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_metrics(
                    [{"name": "scaled_max_error", "options": options}], self.definitions.metrics
                )
        with self.assertRaises(ValueError):
            resolve_metrics(demo.METRICS, builtin_definitions().metrics)
        sample = EvaluationSample("missing", "HE", "S1", root / "target.png", root / "absent.png")
        for mode in ("strict", "permissive"):
            with self.subTest(mode=mode), self.assertRaises(EvaluationCoverageError):
                evaluate_samples([sample], root / mode, metrics=result.metrics, input_failures=mode)
            self.assertTrue((root / mode / "coverage.csv").is_file())
            self.assertFalse((root / mode / "evaluation_result.json").exists())

    def test_consumer_import_audit(self):
        package = Path(demo.__file__).parent
        imports = {}
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module == "tests" or node.module.startswith("tests."))
                    if node.module.startswith("virtual_staining."):
                        self.assertTrue(
                            all(not part.startswith("_") for part in node.module.split("."))
                        )
                        self.assertTrue(all(not alias.name.startswith("_") for alias in node.names))
                        imports.setdefault(node.module, set()).update(a.name for a in node.names)
                if isinstance(node, ast.Import):
                    self.assertTrue(
                        all(
                            a.name != "tests" and not a.name.startswith("tests.")
                            for a in node.names
                        )
                    )
        readme = (package.parent / "README.md").read_text()
        for module, symbols in imports.items():
            self.assertIn(module, readme)
            for symbol in symbols:
                self.assertIn(f"`{symbol}`", readme)


if __name__ == "__main__":
    unittest.main()
