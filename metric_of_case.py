import argparse
import sys

from loguru import logger


def mertic():
    """."""


def main(argv):
    """."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--message", help="搜索的内容", required=True)

    args = parser.parse_args(argv)
    logger.info(args)


if __name__ == "__main__":
    main(sys.argv[1:])
