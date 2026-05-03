from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_root: Path
    source_name: str
    target_name: str
    image_size: tuple[int, int] = (256, 256)
    grid_movement: tuple[int, int] = (256, 256)
    margin: int = 200
    seed: int | None = None
    save_masks: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.05
    test_ratio: float = 0.15
    min_foreground_ratio: float = 0.25
    max_white_ratio: float = 0.7
    white_threshold: int = 250
    max_largest_white_component_ratio: float = 0.20

    @classmethod
    def from_args(cls, args) -> PreprocessingConfig:
        return cls(
            dataset_root=Path(args.path),
            source_name=args.source_name,
            target_name=args.target_name,
            image_size=tuple(args.image_size),
            grid_movement=tuple(args.grid_movement),
            margin=args.margin,
            seed=args.seed,
            save_masks=args.save_masks,
            train_ratio=getattr(args, "train_ratio", 0.8),
            val_ratio=getattr(args, "val_ratio", 0.05),
            test_ratio=getattr(args, "test_ratio", 0.15),
            min_foreground_ratio=args.min_foreground_ratio,
            max_white_ratio=args.max_white_ratio,
            white_threshold=args.white_threshold,
            max_largest_white_component_ratio=args.max_largest_white_component_ratio,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingConfig:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            dataset_root=Path(data["dataset_root"]),
            source_name=data["source_name"],
            target_name=data["target_name"],
            image_size=tuple(data.get("image_size", [256, 256])),
            grid_movement=tuple(data.get("grid_movement", [256, 256])),
            margin=int(data.get("margin", 200)),
            seed=data.get("seed"),
            save_masks=bool(data.get("save_masks", False)),
            train_ratio=float(data.get("train_ratio", 0.8)),
            val_ratio=float(data.get("val_ratio", 0.05)),
            test_ratio=float(data.get("test_ratio", 0.15)),
            min_foreground_ratio=float(data.get("min_foreground_ratio", 0.25)),
            max_white_ratio=float(data.get("max_white_ratio", 0.7)),
            white_threshold=int(data.get("white_threshold", 250)),
            max_largest_white_component_ratio=float(
                data.get("max_largest_white_component_ratio", 0.20)
            ),
        )
