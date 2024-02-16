"""

"""

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches

import argparse
import asyncio
import os
import time
from datetime import datetime
import logging
# Local
from Jingle.scheduler.Jinglescheduler import JingleScheduler
from Jingle.utility.allocation_timeouts import AllocExpirationEventSource
from Jingle.utility.grpc.utility_event_source import UtilityEventSource
from Jingle.utility.kube_manager import KubernetesManager
from Jingle.utility.info_write_load_utils import write_experiment_info_to_files
from Jingle.scheduler.core.performance_recorder import PerformanceRecorder, PerformanceRecorderBank
from Jingle.scheduler.logger.data_logger_bank import DataLoggerBank
from Jingle.scheduler.logger.data_logger import DataLogger
from Jingle.scheduler.logger.event_logger import SimpleEventLogger
from Jingle.scheduler.performance_learners.learner_bank import LearnerBank
from Jingle.scheduler.performance_learners.base_learner import BaseLearner
from Jingle.scheduler.performance_learners.ibtree import IntervalBinaryTree
from Jingle.scheduler.workload_learners.workload_learner_bank import TSForecasterBank
from Jingle.scheduler.workload_learners.base_load_learner import TSBaseLearner
from Jingle.scheduler.allocation_policies.autoscaling import BanditAutoScaler
from Jingle.scheduler.allocation_policies.as_baselines import K8sAutoScaler, PIDAutoScaler, DS2AutoScaler, SimpleJingleAutoScaler, BinaryJingleAutoScaler, SingleJingleAutoScaler
from Jingle.scheduler.allocation_policies.sensing_policy import JingleAutoScaler

# In Demo
from env_gen import generate_env, MICROSERVICES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)

DFLT_REAL_OR_DUMMY = 'real'

INT_UPPER_BOUND = 2
LIP_CONST = 10

# For the data logger -----------------------------------------------------
MAX_INMEM_TABLE_SIZE = 1000
REAL_DATA_LOG_WRITE_TO_DISK_EVERY = 30
REAL_ALLOC_EXPIRATION_TIME = 30 * 2  # Allocate every this many seconds
ALLOC_GRANULARITY = 1  # we cannot assign fractional resources
GRPC_PORT = 10000
LEARNER_DATAPOLL_FREQUENCY = 15  # Learners fetch from data loggers every this many seconds
# This is used by Scheduler if no learners etc. are specified, if they are not provided
# expternally.
NUM_PARALLEL_TRAINING_THREADS = 8
SLEEP_TIME_BETWEEN_TRAINING = 5
ASYNC_SLEEP_TIME = 1
APP_CLIENT_KEY = 'ca-client'

# For logging and saving results ----------------------------------------
SCRIPT_TIME_STR = datetime.now().strftime('%m%d%H%M%S')
# REPORT_RESULTS_EVERY = 60 * 4
REPORT_RESULTS_EVERY = 60


def main():
    """ Main function. """
    # Parse args ===================================================================================
    parser = argparse.ArgumentParser(description='Arguments for running dummy demo')
    parser.add_argument('--env-descr', '-env', type=str, default="test1",
                        help='Environment for running experiment.')
    parser.add_argument('--policy', '-pol', type=str, default="aslearn",
                        help='Which policy to run.')
    parser.add_argument('--cluster-type', '-clus', type=str, default="kind",
                        help='Which cluster_type to run, eks or kind.')
    parser.add_argument('--real-or-dummy', '-rod', type=str, default=DFLT_REAL_OR_DUMMY,
                        help='To run a real or dummy workload.')
    args = parser.parse_args()
    alloc_expiration_time = REAL_ALLOC_EXPIRATION_TIME
    data_log_write_to_disk_every = REAL_DATA_LOG_WRITE_TO_DISK_EVERY

    # Create the environment and other initial set up ==============================================
    env = generate_env(args.env_descr, args.cluster_type, args.real_or_dummy)
    alloc_leaf_order = sorted(list(env.leaf_nodes))

    logger.info('Created Env: %s.\n%s', str(env), env.write_to_file(None))
    entitlements = env.get_entitlements()

    # Create event loggers, framework managers, and event sources ==================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    framework_manager = KubernetesManager(event_queue,
                                          update_loop_sleep_time=1,
                                          dry_run=False)
    event_sources = [UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)]
    # Create the allocation expiration event source -----------------------------------------------
    alloc_expiration_event_source = AllocExpirationEventSource(event_queue, alloc_expiration_time)
    event_sources = [alloc_expiration_event_source, *event_sources]

    # Create directories where we will store experimental results =================================
    # hardcoded for dummy experiments
    # num_resources = framework_manager.get_cluster_resources()
    num_resources = framework_manager.get_cluster_resources() * 6
    experiment_workdir = 'workdirs/%s_%s_%d_%s_%s' % (args.policy, args.env_descr, num_resources,
                                                      args.cluster_type, SCRIPT_TIME_STR)
    if not os.path.exists(experiment_workdir):
        os.makedirs(experiment_workdir, exist_ok=True)
    save_results_file_path = os.path.join(experiment_workdir, 'in_run_results.p')
    # Write experimental information to file before commencing experiments ------------------------
    experiment_info = {}
    experiment_info['resource_quantity'] = num_resources
    experiment_info['alloc_granularity'] = framework_manager.get_alloc_granularity()
    write_experiment_info_to_files(experiment_workdir, env, experiment_info)

    # Create data loggers, learners, time_series forcaseters and performance recorder for each leaf
    # node ========================================================================================
    data_logger_bank = DataLoggerBank(write_to_disk_dir=experiment_workdir,
                                      write_to_disk_every=data_log_write_to_disk_every)
    learner_bank = LearnerBank(num_parallel_training_threads=NUM_PARALLEL_TRAINING_THREADS,
                               sleep_time_between_trains=SLEEP_TIME_BETWEEN_TRAINING)
    load_forecaster_bank = TSForecasterBank(
        num_parallel_training_threads=NUM_PARALLEL_TRAINING_THREADS,
        sleep_time_between_trains=SLEEP_TIME_BETWEEN_TRAINING)
    performance_recorder_bank = PerformanceRecorderBank(
        resource_quantity=num_resources,
        alloc_granularity=framework_manager.get_alloc_granularity(),
        report_results_every=REPORT_RESULTS_EVERY, save_file_name=save_results_file_path,
        report_results_descr=args.policy,
    )

    for leaf_path, leaf in env.leaf_nodes.items():
        logger.info("register all modules for node: " + leaf_path)
        data_logger = DataLogger(
            leaf_path, ['load', 'DEBUG.allocs', 'reward', 'event_start_time', 'event_end_time'],
            index_fld='event_start_time', max_inmem_table_size=MAX_INMEM_TABLE_SIZE)
        data_logger_bank.register(leaf_path, data_logger)

        if args.policy in ['jingle', 'ds2', 'binaryjingle', 'simplejingle', 'singlejingle']:
            app_client_ms_performance_recorder = PerformanceRecorder(
                descr=leaf_path, data_logger=data_logger,
                fields_to_report=['allocs','load', 'reward'])
            performance_recorder_bank.register(leaf_path, app_client_ms_performance_recorder)

        if args.policy in ['cilantro', 'binaryjingle', 'jingle']:
            # learning ------------------------------------------------------------------------------
            model = IntervalBinaryTree(leaf_path, int_lb=0, int_ub=INT_UPPER_BOUND,
                                       lip_const=LIP_CONST)  # Can customise for each leaf.
            learner = BaseLearner(
                app_id=leaf_path, data_logger=data_logger, model=model)
            learner_bank.register(leaf_path, learner)
            learner.initialise()

        if args.policy in ['jingle', 'simplejingle', 'cilantro','ds2', 'binaryjingle', 'singlejingle']:
            # load forecaster ------------------------------------------------------------------------------
            app_client_load_forecaster = TSBaseLearner(
                app_id=leaf_path, data_logger=data_logger, model='arima-default',
                field_to_forecast='load')
            load_forecaster_bank.register(leaf_path, app_client_load_forecaster)
            app_client_load_forecaster.initialise()


    # Create policy ================================================================================
    # Autoscaling policies -------------------------------------------------------------------------
    if args.policy == 'jingle':
        policy = JingleAutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank,
            learner_bank=learner_bank,)
    elif args.policy == 'simplejingle':
        policy = SimpleJingleAutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank,
            learner_bank=learner_bank,)
    elif args.policy == 'binaryjingle':
        policy = BinaryJingleAutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank,
            learner_bank=learner_bank, )
    elif args.policy == 'singlejingle':
        policy = SingleJingleAutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank,
            learner_bank=learner_bank, )
    elif args.policy == 'cilantro':
        policy = BanditAutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            learner_bank=learner_bank,)
    elif args.policy == 'k8sas':
        policy = K8sAutoScaler(
            env=env, resource_quantity=num_resources,
            performance_recorder_bank=performance_recorder_bank)
    elif args.policy == 'pidas':
        policy = PIDAutoScaler(
            env=env, resource_quantity=num_resources,
            performance_recorder_bank=performance_recorder_bank)
    elif args.policy == 'ds2':
        policy = DS2AutoScaler(
            env=env, resource_quantity=num_resources,
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank)
    else:
        raise ValueError('Unknown policy_name %s.' % (args.policy))
    policy.initialise()
    logger.info('Initialised policy %s.', args.policy)

    # Pass learner bank and time series model to the scheduler =====================================
    Jingle = JingleScheduler(event_queue=event_queue,
                           framework_manager=framework_manager,
                           event_logger=event_logger,
                           env=env,
                           policy=policy,
                           data_logger_bank=data_logger_bank,
                           learner_bank=learner_bank,
                           performance_recorder_bank=performance_recorder_bank,
                           load_forecaster_bank=load_forecaster_bank,
                           learner_datapoll_frequency=LEARNER_DATAPOLL_FREQUENCY)
    # Initiate learning/reporting etc --------------------------------------------------------------
    performance_recorder_bank.initiate_report_results_loop()
    data_logger_bank.initiate_write_to_disk_loop()
    learner_bank.initiate_training_loop()
    load_forecaster_bank.initiate_training_loop()

    # Create the workloads and deploy them =========================================================
    all_deps_ready = False
    while not all_deps_ready:
        current_deps = framework_manager.get_deployments().keys()
        logger.info(current_deps)
        all_deps_ready = all(('root--' + d in current_deps) for d in MICROSERVICES)
        if not all_deps_ready:
            not_ready_deps = [d for d in MICROSERVICES if
                              ('root--' + d not in current_deps)]
            ready_deps = [d for d in MICROSERVICES if
                          ('root--' + d in current_deps)]
            logger.info(f"Not all microservices are ready.\nNot ready: {not_ready_deps}.\n" +
                        f"Ready: {ready_deps}.\nUse kubectl create -Rf <path_to_yamls> to run them."
                        f" Currently running: {list(current_deps)}.")
            time.sleep(1)
    logger.info('Workloads ready!')
    # Create event sources =========================================================================
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(Jingle.scheduler_loop())
    finally:
        event_loop.close()


if __name__ == '__main__':
    main()
