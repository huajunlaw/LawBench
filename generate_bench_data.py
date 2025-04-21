import json
import os
import argparse, sys

from loguru import logger
from requests import post


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def completion(
    messages: list[dict[str, str]],
    endpoint="http://127.0.0.1:11434/v1/chat/completions",
    api_key="xxx",
):
    resp = post(
        endpoint,
        json={
            "messages": messages,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1000,
    )
    return resp.json()


def main(argv):
    """生成LawBench."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", dest="endpoint",
                  help="endpoint: it should be a url ")
    parser.add_argument("-m", "--model", dest="model",
                  help="model: it should be a str ")
    parser.add_argument("-k", "--key", dest="api_key",
                  help="key: it should be a str")
    args = parser.parse_args(argv)
    logger.info(args)
    endpoint = args.endpoint
    api_key = args.api_key
    model_name = args.model or "lawchat"
    logger.info(model_name)
    data_path = "./data"
    prediction_path = "./predictions"
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
            if not os.path.exists(input_file):
                continue
            output_file = os.path.join(out_path, data_file)
            if os.path.exists(output_file):
                continue
            data_list = read_json(input_file)
            predictions = {}
            for cnt, item in enumerate(data_list):
                logger.info(item)
                promopt = f"{item['instruction']}\n{item['question']}"
                messages = [{"role": "user", "content": promopt}]
                logger.info(messages)
                resp = completion(messages, endpoint=endpoint, api_key=api_key)
                logger.info(resp)
                prediction = resp['choices'][0]['message']["content"]
                predictions[f"{cnt}"] = {
                    "origin_prompt": promopt,
                    "prediction": prediction,
                    "refr": item["answer"],
                }
            with open(output_file, "w") as f:
                f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
