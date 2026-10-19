"""Download the IBM Telco Customer Churn CSV (same data as Kaggle's blastchar/telco-customer-churn)."""
import requests

from common import DATA_RAW

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    DATA_RAW.write_bytes(resp.content)
    print(f"saved {DATA_RAW} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    main()
