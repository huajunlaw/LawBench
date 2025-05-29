import asyncio
import os
import argparse, sys

from loguru import logger
from utils import read_json, query_complitions


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
        logger.info(input_file)
        asyncio.run(query_complitions(endpoint, api_key, model_name, params, output_file, data_list))


if __name__ == "__main__":
    main(sys.argv[1:])
