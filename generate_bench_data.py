import json
import os

from loguru import logger
from requests import post


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def completion(
    messages: list[dict[str, str]],
    endpoint="http://127.0.0.1:8000/v1/chat/completions",
    api_key="huajunlaw.ai",
):
    resp = post(
        endpoint,
        json={
            "messages": messages,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1000,
    )
    logger.info(resp.text)
    return resp.json()


def main():
    """生成LawBench."""
    data_path = "./data"
    prediction_path = "./predictions"
    model_name = "lawchat"
    data_dirs = os.listdir(data_path)
    for data_dir in data_dirs:
        if data_dir.startswith("."):
            continue
        data_dir_path = os.path.join(data_path, data_dir)
        if not os.path.isdir(data_dir_path):
            continue
        data_files = os.listdir(data_dir_path)
        logger.info(f"*** Evaluating System: {data_dir} ***")
        out_path = os.path.join(prediction_path, data_dir, model_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for data_file in data_files:
            input_file = os.path.join(data_dir_path, data_file)
            data_list = read_json(input_file)
            predictions = {}
            for cnt, item in enumerate(data_list):
                promopt = f"{item['instruction']}\n{item['question']}"
                messages = [{"role": "user", "content": promopt}]
                logger.info(messages)
                resp = completion(messages)
                logger.info(resp)
                prediction = resp["content"]
                predictions[f"{cnt}"] = {
                    "origin_prompt": promopt,
                    "prediction": prediction,
                    "refr": item["answer"],
                }
            with open(os.path.join(out_path, data_file), "w") as f:
                f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main()
