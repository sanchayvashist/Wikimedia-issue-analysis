import pandas as pd
import re
import os

import time
from src.utils import get_gemini_response
from datetime import datetime
from tqdm import tqdm


def parse_incident_report(plain_txt_dir):
    data_df = pd.DataFrame(
        columns=["summary", "detection", "conclusion", "actionable", "time"]
    )

    for filename in tqdm(os.listdir(plain_txt_dir)):
        with open(os.path.join(plain_txt_dir, filename), "r") as file:
            content = file.read()
        if "_" in filename:
            date = filename.split("_")[0]
        else:
            date = filename.split(" ")[0]

        # Updated patterns to be more flexible with section markers
        summary_pattern = (
            r"==\s*Summary\s*==\n(.*?)(?=\n==|$)|Summary:\s*(.*?)(?=\n==|$)"
        )
        detection_pattern = r"==\s*Detection\s*==(.*?)=="
        conclusion_pattern = r"==\s*Conclusions\s*==(.*?)=="
        actionable_pattern = r"==\s*Actionables\s*==(.*)"

        summary_match = re.search(summary_pattern, content, re.DOTALL)
        detection_match = re.search(detection_pattern, content, re.DOTALL)
        conclusion_match = re.search(conclusion_pattern, content, re.DOTALL)
        actionable_match = re.search(actionable_pattern, content, re.DOTALL)

        # Handle both types of summary matches
        summary = (
            (summary_match.group(1) or summary_match.group(2)).strip()
            if summary_match
            else ""
        )
        detection = detection_match.group(1).strip() if detection_match else ""
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        actionable = actionable_match.group(1).strip() if actionable_match else ""

        incident_data = {
            "Summary": re.sub(r"^[^a-zA-Z]*|\n", "", summary),
            "Detection": re.sub(r"^[^a-zA-Z]*|\n{2,}", "", detection),
            "Conclusion": re.sub(r"^[^a-zA-Z]*|\n{2,}", "", conclusion),
            "Actionable": re.sub(r"^[^a-zA-Z]*|\n{2,}", "", actionable),
            "Time": datetime.strptime(date, "%Y-%m-%d"),
        }
        data_df = pd.concat([data_df, pd.DataFrame([incident_data])], ignore_index=True)

    summary_count = (
        data_df["Summary"]
        .apply(lambda x: x is not None and str(x).strip() != "" and len(str(x)) >= 10)
        .sum()
    )
    print(
        f"Count of summaries not null, not empty, and at least 10 characters long: {summary_count}"
    )

    return data_df


import json


def update_parsed_df_with_json(parsed_df, json_data):
    # Load the JSON data
    incident_data = json.loads(json_data.strip("```").strip("json"))

    # Iterate over each incident in the JSON data
    for incident in incident_data:
        # Get the SNo to find the correct index in the DataFrame
        sno = incident["SNo"] - 1  # Adjust for zero-based index

        # Update the DataFrame with the JSON values
        parsed_df.at[sno, "cause_of_incident"] = incident["cause_of_incident"]
        parsed_df.at[sno, "severity_level"] = incident["severity_level"]
        parsed_df.at[sno, "major_impact"] = incident["major_impact"]


def categoies_summaries(prompt, input_text):
    message = []
    print("Generating gemnini response")
    prompt = prompt.replace(f'{{{"input_text"}}}', input_text)
    message.append({"role": "user", "parts": [prompt]})

    return get_gemini_response(message)


if __name__ == "__main__":
    parsed_df = parse_incident_report("dataplain_text")
    parsed_df.to_csv("extracted_dataframe.csv", index=False)

    total_token_usage = 0
    batch_size = 20
    batch_start = 0
    summaries = parsed_df["Summary"].values.tolist()
    with open("src/data_analysis_prompt.txt", "r") as file:
        prompt = file.read()

    for batch_start in tqdm(
        range(0, len(summaries), batch_size), desc="Processing Batches"
    ):
        batch_descriptions = summaries[batch_start : batch_start + batch_size]
        input_text = "SNo | summary\n" + "\n".join(
            [
                f"{batch_start + i+1} | {desc}"
                for i, desc in enumerate(batch_descriptions)
            ]
        )

        json_data, usage_dict = categoies_summaries(
            prompt=prompt, input_text=input_text
        )
        update_parsed_df_with_json(parsed_df, json_data)
        total_token_usage += usage_dict.total_token_count
        time.sleep(30)

    print(f"Total Token Consumption {total_token_usage}")
    parsed_df.to_csv("categorised.csv", index=False)
