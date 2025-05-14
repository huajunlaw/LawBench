import json
import os, random
import argparse, sys

from loguru import logger
from requests import post, get


def read_json(input_file):
    with open(input_file, encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def get_models(endpoint="http://127.0.0.1:11434", api_key="xxx"):
    resp = get(
        f"{endpoint}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=1000,
    )
    logger.info(resp.text)
    return resp.json()
    

def completion(
    messages: list[dict[str, str]],
    endpoint="http://127.0.0.1:11434",
    api_key="xxx",
    model_name="",
    params=None,
):
    req_json = {"messages": messages, "repetition_penalty": 1.5, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    if model_name:
        req_json['model'] = model_name 
    if params and isinstance(params, str):
        req_json.update(json.loads(params))
    logger.info(req_json)
    resp = post(
        f"{endpoint}/v1/chat/completions",
        json=req_json,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    logger.info(resp.text)
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
    parser.add_argument("-s", "--shot", dest="shot",
                  help="shot: it should be a str")
    parser.add_argument("-p", "--parameters", dest="parameters",
                  help="shot: parameters")


    args = parser.parse_args(argv)
    logger.info(args)
    endpoint = args.endpoint
    api_key = args.api_key
    shot = args.shot or "one_shot"
    model_name = args.model or "lawchat"
    params = args.parameters or None
    if params:
        params = params.replace("'", '"')
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
            # promopt = f"\n{item['question']}"
            # messages = [{"role": "system", "content": "你是一个法官，旨在针对各种案件类型、审判程序和事实生成相应的法院裁决。你的回答不能含糊、有争议或者离题"},{"role": "user", "content": promopt}]
            messages = [{"role": "system", "content": {item['instruction']}}, {"role": "user", "content": item['question']}]
            if len(json.dumps(messages)) > 28192:
                logger.info(messages)
            try:
                resp = completion(messages, endpoint=endpoint, api_key=api_key, model_name=model_name, params=params)
                prediction = resp['choices'][0]['message']["content"] or ""
            except Exception as E:
                logger.info(E)
                continue

            predictions[f"{cnt}"] = {
                "origin_prompt": item["question"],
                "prediction": prediction.replace("<think>\n\n</think>\n\n", ""),
                "refr": item["answer"],
            }
            logger.info(prediction)
        with open(output_file, "w") as f:
            f.write(json.dumps(predictions, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
