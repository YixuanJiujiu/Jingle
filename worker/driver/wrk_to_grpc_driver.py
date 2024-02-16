"""
    This driver fetches metrics wrk logs and publishes them to the scheduler.
"""

import argparse
import logging
# local
from Jingle.worker.client import Client
from Jingle.worker.parser.wrk2_log_parser import WrkLogParser
from Jingle.worker.logfolder_data_source import LogFolderDataSource
from Jingle.worker.grpc_publisher import GRPCPublisher
from Jingle.worker.stdout_publisher import StdoutPublisher



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    """ main function. """
    parser = argparse.ArgumentParser(
        description='A client which fetches wrk logs and publishes them.')
    # Add parser args
    LogFolderDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    Client.add_args_to_parser(parser)
    WrkLogParser.add_args_to_parser(parser)
    args = parser.parse_args()
    logger.info(f"Command line args: {args}")

    # log_parser = DummyLogParser()
    log_parser = WrkLogParser()

    # Define objects:
    data_source = LogFolderDataSource(log_dir_path=args.log_folder_path,
                                      log_parser=log_parser,
                                      log_extension=args.log_extension)
    grpcpublisher = GRPCPublisher(client_id=args.grpc_client_id,
                                  ip=args.grpc_ip,
                                  port=args.grpc_port)
    stdoutpublisher = StdoutPublisher()
    client = Client(data_source,
                                publisher=[grpcpublisher, stdoutpublisher],
                                poll_frequency=args.poll_frequency)
    client.run_loop()


if __name__ == '__main__':
    main()
