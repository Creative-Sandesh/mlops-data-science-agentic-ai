"""Run the whole Track A workflow end to end:
data -> training -> tracking -> comparison -> registry -> monitoring (-> retraining recommendation).

Run:  uv run python src/run_pipeline.py
"""
import compare
import get_data
import monitor
import register
import train
from common import DATA_RAW


def main() -> None:
    if not DATA_RAW.exists():
        get_data.main()
    train.main()
    compare.main()
    register.main()
    monitor.main()


if __name__ == "__main__":
    main()
