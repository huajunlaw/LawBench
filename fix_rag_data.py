import json
import os

from loguru import logger


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list

def main():

    data_path = "./data/one_shot"
    rag_path = "./data/one_shot_rag"
    logger.info(data_path)
    out_path = "./data/rag_one_shot"
    data_files = os.listdir(data_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data_file in data_files:
        input_file = os.path.join(data_path, data_file)
        rag_file = os.path.join(rag_path, data_file)
        output_file = os.path.join(out_path, data_file)
        logger.info(input_file)
        data_list = read_json(input_file)
        rag_list = read_json(rag_file)
        predictions = []
        for cnt, item in enumerate(data_list):
            rag = rag_list[cnt]
            logger.info(rag['instruction'])
            predictions.append({
                "instruction": item['instruction'],
                "rag_instruction": rag['instruction'],
                "question": item["question"],
                "answer": item["answer"],
            })
        with open(output_file, "w") as f:
            f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main()
