"""
Download LogHub datasets for log parsing experiments.

Supported datasets: HDFS, BGL, Thunderbird
Source: https://github.com/logpai/loghub-2.0
"""

import os
import argparse
import subprocess
import sys


# LogHub-2.0 dataset URLs (2k samples with ground truth for benchmarking)
LOGHUB_2K_URLS = {
    "HDFS": "https://raw.githubusercontent.com/logpai/loghub-2.0/main/2k_dataset/HDFS/HDFS_2k.log_structured.csv",
    "BGL": "https://raw.githubusercontent.com/logpai/loghub-2.0/main/2k_dataset/BGL/BGL_2k.log_structured.csv",
    "Thunderbird": "https://raw.githubusercontent.com/logpai/loghub-2.0/main/2k_dataset/Thunderbird/Thunderbird_2k.log_structured.csv",
}

# Full dataset info (need manual download due to size)
FULL_DATASET_INFO = {
    "HDFS": {
        "description": "Hadoop Distributed File System logs",
        "size": "~1.5 GB",
        "messages": "11,172,157",
        "url": "https://zenodo.org/records/8196385",
        "note": "Download HDFS_v1/ folder, main file: HDFS.log",
    },
    "BGL": {
        "description": "BlueGene/L Supercomputer logs",
        "size": "~700 MB",
        "messages": "4,747,963",
        "url": "https://zenodo.org/records/8196385",
        "note": "Download BGL/ folder, main file: BGL.log",
    },
    "Thunderbird": {
        "description": "Thunderbird supercomputer logs",
        "size": "~30 GB",
        "messages": "211,212,192",
        "url": "https://zenodo.org/records/8196385",
        "note": "Very large. Download Thunderbird/ folder.",
    },
}


def download_2k_dataset(dataset_name: str, output_dir: str):
    """
    Download the 2k benchmark dataset from LogHub-2.0.
    These are small (2000 log messages) pre-parsed datasets with ground truth.
    Good for initial development and testing.
    """
    if dataset_name not in LOGHUB_2K_URLS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(LOGHUB_2K_URLS.keys())}")
        return

    url = LOGHUB_2K_URLS[dataset_name]
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{dataset_name}_2k.log_structured.csv"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    print(f"Downloading {dataset_name} 2k dataset...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", output_path, url],
            check=True,
        )
        print(f"Done! Saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"wget failed, trying curl...")
        try:
            subprocess.run(
                ["curl", "-L", "-o", output_path, url],
                check=True,
            )
            print(f"Done! Saved to {output_path}")
            return output_path
        except subprocess.CalledProcessError:
            print(f"Download failed. Please download manually from: {url}")
            return None


def print_full_dataset_info(dataset_name: str = None):
    """Print download information for full-size datasets."""
    datasets = [dataset_name] if dataset_name else FULL_DATASET_INFO.keys()

    for name in datasets:
        if name not in FULL_DATASET_INFO:
            print(f"Unknown dataset: {name}")
            continue
        info = FULL_DATASET_INFO[name]
        print(f"\n{'='*60}")
        print(f"  {name}: {info['description']}")
        print(f"  Size: {info['size']} | Messages: {info['messages']}")
        print(f"  URL: {info['url']}")
        print(f"  Note: {info['note']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LogHub datasets")
    parser.add_argument(
        "--dataset", type=str, default="HDFS",
        choices=["HDFS", "BGL", "Thunderbird", "all"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output", type=str, default="./data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--size", type=str, default="2k", choices=["2k", "full"],
        help="Dataset size: '2k' for benchmark (quick), 'full' for complete dataset"
    )
    args = parser.parse_args()

    if args.size == "full":
        print("Full datasets must be downloaded manually:")
        print_full_dataset_info(None if args.dataset == "all" else args.dataset)
    else:
        if args.dataset == "all":
            for name in LOGHUB_2K_URLS:
                download_2k_dataset(name, args.output)
        else:
            download_2k_dataset(args.dataset, args.output)
