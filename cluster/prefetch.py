#!/usr/bin/env python3
"""
Pre-download everything a compute node without internet access will need:
the Fashion-MNIST archives (torchvision) and the pretrained timm weights
used by the sweeps. Run this ONCE on the login node before submitting jobs.

    python cluster/prefetch.py                       # dataset + default backbones
    python cluster/prefetch.py --models resnet50 vit_small
    python cluster/prefetch.py --no-weights          # dataset only

Caches go to ./data (dataset) and $HF_HOME / $TORCH_HOME (weights); export
those two variables to a shared filesystem path in the sbatch script so the
compute nodes see the same cache.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BACKBONES = ["resnet18", "efficientnet_b0", "convnext_tiny", "vit_tiny"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--models", nargs="*", default=DEFAULT_BACKBONES)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--csv", action="store_true",
                   help="also write data/processed/*.csv via src/cli/prepare_data.py")
    args = p.parse_args()

    from torchvision import datasets
    for train in (True, False):
        ds = datasets.FashionMNIST(root=args.data_dir, train=train, download=True)
        print(f"dataset {'train' if train else 'test'}: {len(ds)} samples -> {args.data_dir}")

    if args.csv:
        from src.data.preparation import prepare_data
        prepare_data(data_dir=args.data_dir, output_dir=os.path.join(args.data_dir, "processed"))

    if not args.no_weights:
        from src.models.registry import build_model, resolve_name
        for name in args.models:
            m = build_model(name, pretrained=True)
            n = sum(p.numel() for p in m.parameters())
            print(f"weights {resolve_name(name)}: {n / 1e6:.1f}M params cached "
                  f"(HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')})")
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
