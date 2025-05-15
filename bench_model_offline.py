from vllm import LLM
from vllm.sampling_params import SamplingParams
import argparse, sys, os, random, json
from loguru import logger

def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def main(argv):
    """生成LawBench."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", dest="endpoint",
                  help="endpoint: model的绝对路径")
    parser.add_argument("-m", "--model", dest="model",
                  help="model:  model的名字it should be a str ")
    parser.add_argument("-s", "--shot", dest="shot",
                  help="shot: it should be a str")
    parser.add_argument("-p", "--parameters", dest="parameters",
                  help="shot: parameters")


    args = parser.parse_args(argv)
    logger.info(args)
    endpoint = args.endpoint
    shot = args.shot or "one_shot"
    model_name = args.model or "lawchat"
    params = args.parameters or None
    sampling_params = {"temperature": 0.7}
    if params:
        params = params.replace("'", '"')
        params = json.loads(params)

    sampling_params = SamplingParams(**sampling_params)
    llm = LLM(model=endpoint)
    data_path = f"./data/{shot}"
    logger.info(data_path)
    prediction_path = "./predictions"
    data_files = os.listdir(data_path)
    out_path = os.path.join(prediction_path, shot, model_name)
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
        predictions = {}
        for cnt, item in enumerate(random.sample(data_list, 50)):
            promopt = f"{item['instruction']}\n{item['question']}"
            # messages = [{"role": "system", "content": "你是一个法官，旨在针对各种案件类型、审判程序和事实生成相应的法院裁决。你的回答不能含糊、有争议或者离题"},{"role": "user", "content": promopt}]
            messages = [{"role": "user", "content": promopt}]
            if len(json.dumps(messages)) > 28192:
                logger.info(messages)
            try:
                outputs = llm.chat(messages, sampling_params=sampling_params)
                prediction = outputs[0].outputs[0].text.strip()
            except Exception as E:
                logger.info(E)
                continue

            predictions[f"{cnt}"] = {
                "origin_prompt": promopt,
                "prediction": prediction.replace("<think>\n\n</think>\n\n", ""),
                "refr": item["answer"],
            }
            logger.info(prediction)
        with open(output_file, "w") as f:
            f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
