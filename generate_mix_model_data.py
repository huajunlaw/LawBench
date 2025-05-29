import asyncio
import os
import argparse, sys

from loguru import logger
from utils import read_json, query_complitions


def main(argv):
    """生成mixoutput."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--baseurl", dest="base_url",
                  help="endpoint: it should be a url ")
    parser.add_argument("-k", "--key", dest="api_key",
                  help="key: it should be a str")
    parser.add_argument("-s", "--shot", dest="shot",
                  help="shot: it should be a str")
    parser.add_argument("-p", "--parameters", dest="parameters",
                  help="params: parameters")

    args = parser.parse_args(argv)
    shot = args.shot or "one_shot"
    api_key = args.api_key
    params = args.parameters or None
    if params:
        params = params.replace("'", '"')
    logger.info(args)
    base_url = args.base_url or "http://127.0.0.1"
    model_dict = {'1-1.json': 'Qwen3-8b-rag-one', '1-2.json': 'Qwen3-8b-rag-one', '2-1.json': 'Qwen3-8B-rag-zero', '2-2.json': 'Qwen3-8B', '2-3.json': 'Qwen3-8B', '2-4.json': 'Qwen3-8B', '2-5.json': 'Qwen3-8b-rag-one', '2-6.json': 'Qwen3-8B', '2-7.json': 'Qwen3-8b-rag-one', '2-8.json': 'Qwen3-8B-zero', '2-9.json': 'Qwen3-8B', '2-10.json': 'Qwen3-8B', '3-1.json': 'Qwen3-8B', '3-2.json': 'Qwen3-8b-rag-one', '3-3.json': 'Qwen3-8B', '3-4.json': 'Qwen3-8b-rag-one', '3-5.json': 'Qwen3-8B-LoRA-2', '3-6.json': 'Qwen3-8B-LoRA-2', '3-7.json': 'Qwen3-8B', '3-8.json': 'Qwen3-8B'}
    data_path = f"./data/{shot}"
    logger.info(data_path)
    data_files = os.listdir(data_path)
    out_path = "mix_output"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data_file in data_files:
        if 'rag' in model_dict[data_file]:
            input_file = os.path.join(f"{data_path}_rag", data_file)
        else:
            input_file = os.path.join(data_path, data_file)
        if not os.path.exists(input_file):
            continue
        logger.info(input_file)
        output_file = os.path.join(out_path, data_file)
        if os.path.exists(output_file):
            continue
        data_list = read_json(input_file)
        if 'LoRA' in model_dict[data_file]:
            endpoint = f"{base_url}:8001"
            model_name = 'Qwen3-8B-LoRA-checkpoint1890'
        else:
            model_name = 'Qwen3-8B'
            endpoint = f"{base_url}:8000"
        logger.info(endpoint)
        logger.info(input_file)
        asyncio.run(query_complitions(endpoint, api_key, model_name, params, output_file, data_list))

if __name__ == "__main__":
    main(sys.argv[1:])
    logger.info('')
