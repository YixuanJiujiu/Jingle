import argparse
import logging
import os

from driver import WebDriver

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_env_name(k8s_svc_name: str) -> str:
    """
    Gets the envvar name for a k8s service by replacing - with _ and makign it all uppercase
    """
    env_name = k8s_svc_name.replace('-', '_')
    env_name = env_name.upper()
    return env_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to run the web_serving load generator.')

    # ======= WrkDriver Args ===========
    parser.add_argument('--logdir', type=str, default="/Jinglelogs/",
                        help='Output log dir.')
    parser.add_argument('--url', type=str,
                        default="http://localhost:6000",
                        help='Target URL')
    parser.add_argument('--ms', type=str,
                        default="coding-assign",
                        help='Target microservice')
    args = parser.parse_args()

    # ======== Initialize Driver =========
    os.makedirs(args.logdir, exist_ok=True)
    driver = WebDriver(logdir=args.logdir,
                       url=args.url,
                       ms = args.ms)

    # ======== Run Driver =========
    driver.run_loop()
