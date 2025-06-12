import sys, argparse

from loguru import logger
from langchain_community.tools import DuckDuckGoSearchRun


def main(argv):
    """."""
    # 0. constant
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--message", help="搜索的内容", required=True)

    args = parser.parse_args(argv)
    logger.info(args)
    search = DuckDuckGoSearchRun()

    logger.info(search.invoke(args.message))

if __name__ == "__main__":
    main(sys.argv[1:])
