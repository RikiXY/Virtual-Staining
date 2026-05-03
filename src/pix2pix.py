import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from virtual_staining.data.dataset import PairedHistologyDataset
from virtual_staining.models.discriminator import PatchGANDiscriminator
from virtual_staining.models.generator import UNetGenerator
from virtual_staining.training.config import TrainingConfig
from virtual_staining.utils.cli import print_info, print_section, style


# --------------------- Working paths ---------------------
# This function builds all the main project paths starting from a single
# root. It avoids scattering hardcoded strings throughout the rest of
# the file.
def build_workspace_paths(run_root: str | Path) -> dict:
    root = Path(run_root)

    return {
        "run_root": root,
        "logs_dir": root / "logs",
        "checkpoints_dir": root / "checkpoints",
        "output_val_dir": root / "output_val",
        "output_test_dir": root / "output_test",
        "output_train_dir": root / "output_train",
    }


# --------------------- Command-line arguments ---------------------
# We separate parser and logic.
# The script supports two distinct modes: training and testing.
def build_parser():
    parser = argparse.ArgumentParser(
    prog="python src/pix2pix.py",
    description="Train or test the Pix2Pix model on a paired histology dataset.",
    epilog=(
        "Examples:\n"
        "  python src/pix2pix.py train "
        "--dataset-root local_workspace/datasets/inverted_256 "
        "--run-name inv_P-256_L1-25 "
        "--epochs 100 "
        "--image-size 256 256\n"
        "\n"
        "  python src/pix2pix.py test "
        "--dataset-root local_workspace/datasets/inverted_256 "
        "--run-path local_workspace/results/inv_P-256_L1-25 "
        "--checkpoint local_workspace/results/inv_P-256_L1-25/checkpoints/ep099.pth\n"
        "\n"
        "Use 'python src/pix2pix.py <command> --help' "
        "to see the options for a specific command."
    ),
    formatter_class=argparse.RawTextHelpFormatter
)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Pix2Pix model",
        formatter_class=argparse.RawTextHelpFormatter
    )
    train_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_train/ and dataset_val/"
    )
    train_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the output run directory to create"
    )
    train_parser.add_argument(
        "--results-path",
        type=str,
        default="local_workspace/results",
        help="Base directory where the new run folder will be created (default: local_workspace/results)"
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, a random seed is generated."
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for the DataLoader (default: 8)"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of DataLoader workers (default: min(4, cpu_count))"
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help=(
            "Resize images before training as HEIGHT WIDTH "
            "(default: 256 256). Use 512 512 for 512x512 patch experiments."
        )
    )
    train_parser.add_argument(
        "--log-rate",
        type=int,
        default=15,
        help="Log every N batches (default: 15)"
    )
    train_parser.add_argument(
        "--checkpoint-rate",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--validate-rate",
        type=int,
        default=10,
        help="Run validation every N epochs (default: 10)"
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from"
    )
    train_parser.add_argument(
        "--l1-weight",
        type=float,
        default=25.0,
        help="Weight of the L1 reconstruction loss (default: 25.0)"
    )
    train_parser.add_argument(
        "--lr-g",
        type=float,
        default=2e-4,
        help="Learning rate for the generator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--lr-d",
        type=float,
        default=2e-4,
        help="Learning rate for the discriminator (default: 2e-4)"
    )
    train_parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1 (default: 0.5)"
    )
    train_parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999)"
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Run inference on the test set",
        formatter_class=argparse.RawTextHelpFormatter
    )
    test_parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root containing dataset_test/"
    )
    test_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to use for inference"
    )
    test_parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Path to an existing training run containing checkpoints/ and output folders"
    )
    test_parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(256, 256),
        help=(
            "Resize images before inference as HEIGHT WIDTH "
            "(default: 256 256). Must match the image size used during training."
        )
    )

    return parser


def is_amp_enabled(device):
    # Mixed precision with `autocast` and `GradScaler` is mainly useful
    # on CUDA GPUs. On CPU we leave everything disabled to avoid unnecessary
    # complexity and keep behaviour as straightforward as possible.
    return isinstance(device, torch.device) and device.type == "cuda"


# --------------------- Determinism ---------------------
def set_seed(seed):
    # Fixing the seed greatly reduces variability and helps when comparing experiments.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- Logging ---------------------
def log_message(message, log_file, show_time=True, use_stdout=True):
    # We log to file as well as to the screen, so that after hours or
    # days one can understand what happened.
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if show_time:
        message = f"[{now_str}] {message}"
    with open(log_file, "a+") as f:
        f.write(message + "\n")
    if use_stdout:
        print(message)

def get_first_pair_size(dataset):
    """
    Returns the actual on-disk size of the first pair in the dataset.
    Useful to distinguish between native patches and the resize applied during training.
    """
    if len(dataset) == 0:
        return None

    source_path, target_path = dataset.pairs[0]

    with Image.open(source_path) as src_img:
        source_size = src_img.size  # (width, height)

    with Image.open(target_path) as tgt_img:
        target_size = tgt_img.size  # (width, height)

    return {
        "source": source_size,
        "target": target_size,
        "source_path": source_path,
        "target_path": target_path,
    }


def save_run_config(run_config, run_root):
    config_path = Path(run_root) / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
    return config_path


def log_run_header(log_file, run_config):
    """
    Writes an ordered initial summary of the run to the log.
    """
    log_message("=" * 80, log_file, show_time=False, use_stdout=False)
    log_message("RUN CONFIGURATION", log_file, show_time=False, use_stdout=False)
    log_message("=" * 80, log_file, show_time=False, use_stdout=False)

    for key, value in run_config.items():
        log_message(f"{key}: {value}", log_file, show_time=False, use_stdout=False)

    log_message("=" * 80, log_file, show_time=False, use_stdout=False)

class ProgressTracker:
    def __init__(
        self,
        total_epochs,
        total_batches,
        start_epoch=0,
        max_history=300,
        warmup_batches=10,
        min_eta_batches=5,
    ):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_epoch = start_epoch
        self.max_history = max_history
        self.warmup_batches = warmup_batches
        self.min_eta_batches = min_eta_batches

        self.total_steps = total_epochs * total_batches
        self.start_step = start_epoch * total_batches

        self.start_time = None
        self.last_step_time = None
        self.step_durations = []

    def start(self):
        now = time.time()
        self.start_time = now
        self.last_step_time = now
        self.step_durations = []

    def calculate_progress(self, epoch, batch):
        now = time.time()

        current_step = epoch * self.total_batches + batch + 1
        completed_since_start = current_step - self.start_step
        remaining_steps = max(self.total_steps - current_step, 0)

        step_duration = now - self.last_step_time
        self.last_step_time = now

        if completed_since_start > self.warmup_batches:
            self.step_durations.append(step_duration)

            if len(self.step_durations) > self.max_history:
                self.step_durations.pop(0)

        total_elapsed_time = now - self.start_time
        progress = current_step / self.total_steps if self.total_steps > 0 else 1.0

        if len(self.step_durations) < self.min_eta_batches:
            eta = None
            end_time = None
        else:
            avg_step_time = sum(self.step_durations) / len(self.step_durations)
            eta = avg_step_time * remaining_steps
            end_time = now + eta

        return progress, total_elapsed_time, eta, end_time


def format_duration(seconds):
    """Formats a duration in a human-readable way."""
    if seconds is None:
        return "--"

    seconds = int(max(seconds, 0))

    if seconds < 60:
        return f"{seconds}s"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def color_progress(progress: float) -> str:
    text = f"{progress:.2%}"

    if progress < 0.33:
        return style(text, "yellow")
    if progress < 0.66:
        return style(text, "cyan")
    if progress < 0.90:
        return style(text, "blue")

    return style(text, "green")


def render_progress_bar(progress: float, width: int = 40) -> str:
    progress = min(max(progress, 0.0), 1.0)

    filled = int(width * progress)

    if progress > 0 and filled == 0:
        filled = 1

    if progress >= 1:
        filled = width

    empty = width - filled

    bar = "█" * filled + "-" * empty
    return f"[{style(bar, 'green')}]"


def update_console_progress(message: str) -> None:
    """Updates a single progress line in the console."""
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 140

    clean_message = message[: terminal_width - 1]
    print("\r" + clean_message.ljust(terminal_width - 1), end="", flush=True)


def finish_console_progress() -> None:
    """Closes the current progress line."""
    print()


# --------------------- Checkpoints ---------------------
def save_checkpoint(
    checkpoint_path,
    epoch,
    G,
    D,
    opt_G,
    opt_D,
    scaler_G,
    scaler_D,
    config: TrainingConfig,
):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "optimizerG_state_dict": opt_G.state_dict(),
        "optimizerD_state_dict": opt_D.state_dict(),
        "scalerG_state_dict": scaler_G.state_dict(),
        "scalerD_state_dict": scaler_D.state_dict(),
        "l1_weight": config.l1_weight,
        "lr_g": config.lr_g,
        "lr_d": config.lr_d,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "dataset_root": str(config.dataset_root),
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, G, D, opt_G, opt_D, scaler_G, scaler_D, device=None, image_size=(256, 256)):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint to load.
        G (nn.Module): Generator model.
        D (nn.Module): Discriminator model.
        opt_G (torch.optim.Optimizer): Generator optimiser.
        opt_D (torch.optim.Optimizer): Discriminator optimiser.
        scaler_G (torch.cuda.amp.GradScaler): GradScaler for the generator.
        scaler_D (torch.cuda.amp.GradScaler): GradScaler for the discriminator.
        image_size (tuple): Image dimensions used during training.
        device (torch.device): Device on which to load the models.
    """
    # Here we load not only the model weights, but also the optimiser and scaler.
    # This makes the resume truly consistent with the point at which training
    # was interrupted.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_image_size = checkpoint.get("image_size")
    if image_size is not None and checkpoint_image_size is not None:
        checkpoint_image_size = tuple(checkpoint_image_size)
        requested_image_size = tuple(image_size)

        if checkpoint_image_size != requested_image_size:
            raise ValueError(
                "Image size mismatch between checkpoint and resumed training. "
                f"Checkpoint image_size={checkpoint_image_size}, "
                f"current image_size={requested_image_size}."
            )

    G.load_state_dict(checkpoint['generator_state_dict'])
    D.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    opt_D.load_state_dict(checkpoint['optimizerD_state_dict'])
    scaler_G.load_state_dict(checkpoint['scalerG_state_dict'])
    scaler_D.load_state_dict(checkpoint['scalerD_state_dict'])
    # We resume from the next epoch to avoid repeating the one already completed.
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

# --------------------- Utility functions ---------------------
def save_images(path, source_tensor, output, target, epoch, batch_index):
    """
    Saves the input, output and target images in TIF format.

    Args:
        path (str): Path to the save directory.
        source_tensor (Tensor): Input image.
        output (Tensor): Image generated by the model.
        target (Tensor): Target image corresponding to the current input.
        epoch (int): Current epoch number.
        batch_index (int): Current batch index.
    """
    # Images are normalised to [-1, 1]; before saving them for human viewing
    # we need to bring them back to the [0, 1] range.
    save_image((source_tensor * 0.5 + 0.5), Path(path) / f"epoch{epoch}_batch{batch_index}_input.tif")
    save_image((output * 0.5 + 0.5), Path(path) / f"epoch{epoch}_batch{batch_index}_output.tif")
    save_image((target * 0.5 + 0.5), Path(path) / f"epoch{epoch}_batch{batch_index}_target.tif")

# --------------------- Training and Validation ---------------------
def validate(
    G,
    D,
    validation_loader,
    device,
    bce_loss,
    l1_loss, epoch,
    log_file,
    output_val_dir,
    l1_weight
):
    """
    Runs a validation pass and computes the average losses
    for the generator and discriminator.
    """
    G.eval()
    D.eval()

    Path(output_val_dir).mkdir(parents=True, exist_ok=True)

    total_loss_G = 0.0
    total_loss_D = 0.0
    count = 0
    amp_enabled = is_amp_enabled(device)

    with torch.no_grad():
        for i, (x, y) in enumerate(validation_loader):
            x, y = x.to(device), y.to(device)

            # `autocast` enables mixed precision when the device supports it.
            # In practice, some operations are performed in reduced precision
            # to gain speed and save memory.
            with autocast(device_type=device.type, enabled=amp_enabled):
                fake = G(x)

                D_real = D(x, y)
                D_fake = D(x, fake)

                real_label = torch.ones_like(D_real, device=device)
                fake_label = torch.zeros_like(D_fake, device=device)

                # BCE measures the adversarial component: how well the discriminator
                # distinguishes real pairs from fake ones.
                loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)

                # The generator loss combines two objectives:
                # 1) fool the discriminator;
                # 2) stay close to the correct target.

                # The L1 component is useful to prevent plausible outputs that are
                # disconnected from the specific target of the current sample.
                loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * l1_weight

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            count += 1

            # Save a few previews.
            if i < 5:
                save_images(output_val_dir, x[0], fake[0], y[0], epoch, i)

    avg_loss_D = total_loss_D / count if count > 0 else 0.0
    avg_loss_G = total_loss_G / count if count > 0 else 0.0

    log_message(
        f"[Epoch {epoch}] Validation: loss_G={avg_loss_G:.4f} loss_D={avg_loss_D:.4f}",
        log_file, 
        use_stdout=False
    )

    return avg_loss_G, avg_loss_D

def test_inference(
    checkpoint_path,
    test_folder,
    output_folder,
    image_size=(256, 256),
    device=None
):
    """
    Runs inference on the test set and saves the generated images.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    G = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_image_size = checkpoint.get("image_size")
    if checkpoint_image_size is not None:
        checkpoint_image_size = tuple(checkpoint_image_size)
        requested_image_size = tuple(image_size)

        if checkpoint_image_size != requested_image_size:
            raise ValueError(
                "Image size mismatch between checkpoint and inference. "
                f"Checkpoint was trained with image_size={checkpoint_image_size}, "
                f"but inference is using image_size={requested_image_size}. "
                "Pass the correct --image-size or use a matching checkpoint."
            )
    G.load_state_dict(checkpoint["generator_state_dict"])
    G.eval()

    # Preprocessing must stay consistent with training,
    # otherwise the model would receive inputs with a different distribution.
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = PairedHistologyDataset(test_folder, transform=transform)

    if not dataset.pairs:
        print(f"No test source files found in: {test_folder}")
        return

    amp_enabled = is_amp_enabled(device)

    with torch.no_grad():
        for idx, (source_tensor, _) in enumerate(dataset):
            source_path = dataset.pairs[idx][0]
            prefix = source_path.stem[:-len("_source")]
            out_filename = f"{prefix}_target_generated{source_path.suffix.lower()}"

            img_tensor = source_tensor.unsqueeze(0).to(device)

            with autocast(device_type=device.type, enabled=amp_enabled):
                fake_target = G(img_tensor)

            # Bring the output back to the range suitable for saving.
            fake_target = (fake_target * 0.5) + 0.5
            fake_target = fake_target.clamp(0, 1)

            out_path = Path(output_folder) / out_filename
            save_image(fake_target, out_path)

    print(f"Test completed. Images saved in {output_folder}")
    
def train_one_epoch(
    G,
    D,
    training_loader,
    device,
    opt_G,
    opt_D,
    scaler_G,
    scaler_D,
    bce_loss,
    l1_loss,
    epoch,
    log_file,
    progress_tracker,
    log_rate,
    l1_weight,
    training_status
):
    """
    Trains the generator and discriminator for a single epoch.
    """
    G.train()
    D.train()
    amp_enabled = is_amp_enabled(device)

    last_loss_G = None
    last_loss_D = None

    for i, (x, y) in enumerate(training_loader):
        x, y = x.to(device), y.to(device)

        # ---------- DISCRIMINATOR ----------
        # We update the discriminator first. In this step its task is to
        # distinguish real pairs `(x, y)` from synthetic ones `(x, fake)`.
        with autocast(device_type=device.type, enabled=amp_enabled):
            # `.detach()` is essential: we want to use the generator output
            # as a fake example for D, but without letting gradients flow back into G.
            fake = G(x).detach()

            D_real = D(x, y)
            D_fake = D(x, fake)

            real_label = torch.ones_like(D_real, device=device)
            fake_label = torch.zeros_like(D_fake, device=device)

            loss_D = bce_loss(D_real, real_label) + bce_loss(D_fake, fake_label)

        opt_D.zero_grad()
        # With mixed precision, GradScaler helps avoid numerical issues
        # during the backward pass in reduced precision.
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # ---------- GENERATOR ----------
        # Now we update the generator. Here we do NOT detach, because
        # we want the gradient to actually flow through G and correct it.
        with autocast(device_type=device.type, enabled=amp_enabled):
            fake = G(x)
            D_fake = D(x, fake)

            # The generator loss combines two complementary drives:
            # - adversarial BCE: make the output look realistic enough
            #   to "fool" the discriminator;
            # - L1: maintain fidelity towards the real target.
            loss_G = bce_loss(D_fake, real_label) + l1_loss(fake, y) * l1_weight

        opt_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        last_loss_G = loss_G.item()
        last_loss_D = loss_D.item()

        progress, total_elapsed_time, eta, end_time = progress_tracker.calculate_progress(epoch, i)

        elapsed_str = format_duration(total_elapsed_time)
        eta_str = format_duration(eta)
        progress_bar = render_progress_bar(progress)

        epoch_progress = (i + 1) / progress_tracker.total_batches

        console_message = (
            f"{progress_bar} "
            f"global {color_progress(progress)} | "
            f"ep {epoch + 1}/{progress_tracker.total_epochs} "
            f"({epoch_progress:.0%}) | "
            f"b {i + 1}/{progress_tracker.total_batches} | "
            f"loss_G {loss_G.item():.4f} | "
            f"loss_D {loss_D.item():.4f} | "
            f"elapsed {elapsed_str} | "
            f"ETA {eta_str} | "
            f"ckpt {training_status['last_checkpoint']}"
        )

        should_update_progress = (
            i % log_rate == 0
            or i == len(training_loader) - 1
        )

        if should_update_progress:
            update_console_progress(console_message)
        
        if i % log_rate == 0:
            if end_time is None:
                end_time_str = "warming up"
            else:
                end_time_str = datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")

            log_message(
                f"[ep {epoch} | b {i}] loss_G: {loss_G.item():.4f} "
                f"loss_D: {loss_D.item():.4f} - "
                f"{progress:.2%} | elapsed {elapsed_str} | ETA {eta_str} | end {end_time_str}",
                log_file,
                use_stdout=False,
            )
    return last_loss_G, last_loss_D

# --------------------- Main ---------------------
def main(config: TrainingConfig):
    start_time = time.time()

    paths = build_workspace_paths(config.run_root)
    logs_dir = paths["logs_dir"]
    checkpoints_dir = paths["checkpoints_dir"]
    output_val_dir = paths["output_val_dir"]
    output_train_dir = paths["output_train_dir"]

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(output_train_dir).mkdir(parents=True, exist_ok=True)
    Path(output_val_dir).mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = Path(logs_dir) / f"Log-{timestamp_str}.txt"

    if log_file.exists():
        os.remove(log_file)

    seed = config.seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    set_seed(seed)
    log_message(f"Seed set to {seed}", log_file, use_stdout=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    log_message(f"Device: {device} ({device_name})", log_file)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_dir = Path(config.dataset_root) / "dataset_train"
    val_dir = Path(config.dataset_root) / "dataset_val"

    training_dataset = PairedHistologyDataset(train_dir, transform=transform)
    validation_dataset = PairedHistologyDataset(val_dir, transform=transform)

    first_train_pair_info = get_first_pair_size(training_dataset)
    first_val_pair_info = get_first_pair_size(validation_dataset)

    run_config = {
        "timestamp": timestamp_str,
        "dataset_root": str(config.dataset_root),
        "run_root": str(config.run_root),
        "logs_dir": str(logs_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "output_train_dir": str(output_train_dir),
        "output_val_dir": str(output_val_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "seed": seed,
        "device": str(device),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "image_size_resize": list(config.image_size),
        "log_rate": config.log_rate,
        "checkpoint_rate": config.checkpoint_rate,
        "validate_rate": config.validate_rate,
        "resume_checkpoint": str(config.resume) if config.resume else None,
        "l1_weight": config.l1_weight,
        "lr_g": config.lr_g,
        "lr_d": config.lr_d,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "train_samples": len(training_dataset),
        "val_samples": len(validation_dataset),
        "first_train_pair_info": first_train_pair_info,
        "first_val_pair_info": first_val_pair_info,
    }

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_config["train_batches"] = len(training_loader)
    run_config["val_batches"] = len(validation_loader)

    config_path = save_run_config(run_config, config.run_root)
    log_run_header(log_file, run_config)
    log_message(f"Run config saved to {config_path}", log_file, use_stdout=False)

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    opt_D = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))

    amp_enabled = is_amp_enabled(device)
    scaler_G = GradScaler(enabled=amp_enabled)
    scaler_D = GradScaler(enabled=amp_enabled)

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Clean the training output directory to avoid mixing material from different runs.
    for file in os.listdir(output_train_dir):
        file_path = Path(output_train_dir) / file
        if file_path.is_file():
            os.remove(file_path)

    start_epoch = 0
    if config.resume is not None:
        if Path(config.resume).exists():
            start_epoch = load_checkpoint(
                config.resume,
                generator,
                discriminator,
                opt_G,
                opt_D,
                scaler_G,
                scaler_D,
                device,
                image_size=config.image_size,
            )
            log_message(f"Checkpoint loaded from {config.resume}, epoch {start_epoch}", log_file, use_stdout=False)
        else:
            log_message(f"WARNING - Checkpoint not found: {config.resume}", log_file, use_stdout=False)
    else:
        log_message("Training started from scratch", log_file, use_stdout=False)

    print_section("Pix2Pix training")
    print_info("Run root", str(config.run_root))
    print_info("Dataset root", str(config.dataset_root))
    print_info("Device", str(device))
    print_info("Epochs", str(config.epochs))
    print_info("Start epoch", str(start_epoch))
    print_info("Train samples", str(len(training_dataset)))
    print_info("Validation samples", str(len(validation_dataset)))
    print_info("Train batches/epoch", str(len(training_loader)))
    print_info("Validation batches", str(len(validation_loader)))
    print_info("Detailed log", str(log_file))
    print()
    print(style("Training progress:", "bold", "cyan"))

    progress_tracker = ProgressTracker(
        total_epochs=config.epochs,
        total_batches=len(training_loader),
        start_epoch=start_epoch,
        warmup_batches=max(10, config.log_rate),
    )
    progress_tracker.start()

    log_message("Training started", log_file, use_stdout=False)
    log_message(
        f"Hyperparameters | l1_weight={config.l1_weight} | lr_g={config.lr_g} | lr_d={config.lr_d} | "
        f"beta1={config.beta1} | beta2={config.beta2}",
        log_file,
        use_stdout=False,
    )

    training_status = {
        "last_checkpoint": Path(config.resume).name if config.resume else "none "
    }

    for epoch in range(start_epoch, config.epochs):
        log_message(f"Starting epoch {epoch}", log_file, use_stdout=False)

        last_loss_G, last_loss_D = train_one_epoch(
            generator,
            discriminator,
            training_loader,
            device,
            opt_G,
            opt_D,
            scaler_G,
            scaler_D,
            bce_loss,
            l1_loss,
            epoch,
            log_file,
            progress_tracker,
            config.log_rate,
            config.l1_weight,
            training_status,
        )

        log_message(f"Finished epoch {epoch}", log_file, use_stdout=False)

        # Save periodic checkpoints: if training is interrupted, we do not have to start from scratch.
        if (epoch + 1) % config.checkpoint_rate == 0:
            checkpoint_path = Path(checkpoints_dir) / f"ep{epoch:03d}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch,
                generator,
                discriminator,
                opt_G,
                opt_D,
                scaler_G,
                scaler_D,
                config,
            )
            training_status["last_checkpoint"] = Path(checkpoint_path).name
            log_message(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}", log_file, use_stdout=False)
            if epoch == config.epochs - 1:
                update_console_progress(
                    f"{render_progress_bar(1.0)} "
                    f"global {color_progress(1.0)} | "
                    f"ep {epoch + 1}/{config.epochs} (100%) | "
                    f"b {len(training_loader)}/{len(training_loader)} | "
                    f"loss_G {last_loss_G:.4f} | "
                    f"loss_D {last_loss_D:.4f} | "
                    f"elapsed {format_duration(time.time() - start_time)} | "
                    f"ETA 0s | "
                    f"ckpt {training_status['last_checkpoint']}"
                )

        # Periodic validation measures whether the model is improving
        # on data outside the training set as well.
        if (epoch + 1) % config.validate_rate == 0:
            validate(
                generator,
                discriminator,
                validation_loader,
                device,
                bce_loss,
                l1_loss,
                epoch,
                log_file,
                output_val_dir,
                config.l1_weight,
            )

    finish_console_progress()
    total_seconds = time.time() - start_time
    log_message(f"Execution completed. Total time = {total_seconds:.2f} seconds", log_file, use_stdout=False)
    
              

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        main(TrainingConfig.from_args(args))

    elif args.mode == "test":
        test_dir = Path(args.dataset_root) / "dataset_test"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        run_root = Path(args.run_path)
        paths = build_workspace_paths(run_root)

        print(f"Starting test for run: {run_root}")
        print(f"Using checkpoint: {args.checkpoint}")
        test_inference(
            checkpoint_path=args.checkpoint,
            test_folder=str(test_dir),
            output_folder=str(paths["output_test_dir"]),
            image_size=tuple(args.image_size),
            device=device
        )
