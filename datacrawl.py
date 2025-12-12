import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

BASE_URL = "https://iswa.ccmc.gsfc.nasa.gov/iswa_data_tree/model/heliosphere/sep_scoreboard/mag4_2019/WF-HMI-NRT-JSON/2023/"
RAW_DIR = "data/msg4_LOSr_2023/raw"
OUT_DIR = "data/msg4_LOSr_2023/processed"

def list_files(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href.endswith(".json"):
            files.append(url + href)
        elif href.endswith("/"):  # subdir (month)
            files.extend(list_files(url + href))
    return files

def download_file(file_url, outdir):
    os.makedirs(outdir, exist_ok=True)
    local_path = os.path.join(outdir, file_url.split("/")[-1])
    if not os.path.exists(local_path):
        r = requests.get(file_url)
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path

def parse_json(local_file):
    with open(local_file, "r") as f:
        data = json.load(f)

    sub = data.get("sep_forecast_submission", {})
    model_short = sub.get("model", {}).get("short_name", "mag4_l")
    issue_time = sub.get("issue_time")
    mode = sub.get("mode", "LOSr")

    records = []
    for fc in sub.get("forecasts", []):
        record = {
            "model_short": model_short,
            "mode": mode,
            "issue_time": issue_time,
            "species": fc.get("species"),
            "location": fc.get("location"),
            "energy_min": fc.get("energy_channel", {}).get("min"),
            "energy_max": fc.get("energy_channel", {}).get("max"),
            "energy_units": fc.get("energy_channel", {}).get("units"),
            "pred_start": fc.get("prediction_window", {}).get("start_time"),
            "pred_end": fc.get("prediction_window", {}).get("end_time"),
            "prob_value": float(fc.get("probabilities", [{}])[0].get("probability_value", 0)),
            "prob_uncertainty": float(fc.get("probabilities", [{}])[0].get("uncertainty", 0)),
            "threshold": fc.get("probabilities", [{}])[0].get("threshold"),
            "all_clear": fc.get("all_clear", {}).get("all_clear_boolean"),
            "all_clear_prob_thresh": fc.get("all_clear", {}).get("probability_threshold"),
        }
        records.append(record)
    return records

def main():
    all_files = list_files(BASE_URL)
    print(f"Found {len(all_files)} files.")

    all_records = []
    for f in all_files:
        local_file = download_file(f, RAW_DIR)
        try:
            recs = parse_json(local_file)
            all_records.extend(recs)
        except Exception as e:
            print("Error parsing", local_file, e)

    df = pd.DataFrame(all_records)
    if df.empty:
        print(" No data parsed, check JSON structure!")
        return

    # Convert dates
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["pred_start"] = pd.to_datetime(df["pred_start"])
    df["pred_end"] = pd.to_datetime(df["pred_end"])

    # Partition columns
    df["year"] = df["issue_time"].dt.year
    df["month"] = df["issue_time"].dt.month

    # Save partitioned parquet
    df.to_parquet(OUT_DIR, engine="pyarrow", partition_cols=["year","month"], index=False)
    print(f" Saved partitioned parquet under {OUT_DIR}")

if __name__ == "__main__":
    main()

# TO CHECK NUMBER OF FILES DOWNLADED :  ls data/magpy_2023/raw/ | wc -l 
