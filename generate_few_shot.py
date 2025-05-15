import json
import os, random

from loguru import logger


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list

def main():

    data_path = "./data/one_shot"
    logger.info(data_path)
    out_path = "./data/few_shot"
    data_files = os.listdir(data_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data_file in data_files:
        input_file = os.path.join(data_path, data_file)
        if not os.path.exists(input_file):
            logger.info(input_file)
            continue
        output_file = os.path.join(out_path, data_file)
        if os.path.exists(output_file):
            continue
        data_list = read_json(input_file)
        predictions = []
        for item in data_list:
            promopt = item['instruction']
            predictions.append({
                "instruction": promopt,
                "question": item["question"], 
                "answer": item["answer"],
            })
            logger.info(predictions)
        with open(output_file, "w") as f:
            f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main()
