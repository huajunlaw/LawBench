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
        output_file = os.path.join(out_path, data_file)
        logger.info(input_file)
        data_list = read_json(input_file)
        predictions = []
        data_len = len(data_list)
        for cnt, item in enumerate(data_list):
            promopt = item['instruction']
            logger.info(item)
            _1 = data_list[-(data_len - 1 -cnt)]
            logger.info(_1)
            logger.info(data_list[1])
            _2 = data_list[-(data_len - 2 -cnt)]
            logger.info(_2)
            if '<eoa>' in promopt:
                sep = '<eoa>'
            else:
                sep = ''
            promopt = promopt.replace("下面是一个例子:\n", f"下面是三个例子:\n{_1['question']}\n{_1['answer']}{sep}\n{_2['question']}\n{_2['answer']}{sep}\n")
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
