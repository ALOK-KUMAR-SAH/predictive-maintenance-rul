import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .config import REPORT_DIR, ensure_project_dirs
from .data import load_cmapss


def create_plots(path: str, output_dir: str | Path = REPORT_DIR) -> list[Path]:
    frame = load_cmapss(path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plots = []
    for sensor in ["sensor_2", "sensor_3", "sensor_4"]:
        figure, axis = plt.subplots(figsize=(9, 4))
        sample = frame[frame["unit_id"] <= 5]
        sns.lineplot(data=sample, x="cycle", y=sensor, hue="unit_id", ax=axis, legend=False)
        axis.set_title(f"{sensor} over cycle for first five engines")
        figure.tight_layout()
        target = destination / f"{sensor}_over_cycle.png"
        figure.savefig(target, dpi=120)
        plt.close(figure)
        plots.append(target)
    return plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create basic C-MAPSS exploratory plots.")
    parser.add_argument("--data", default="data/raw/train_FD001.txt")
    args = parser.parse_args()
    ensure_project_dirs()
    print("\n".join(str(path) for path in create_plots(args.data)))


if __name__ == "__main__":
    main()
