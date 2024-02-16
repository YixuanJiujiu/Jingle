"""
    Generates the environment.
"""

from Jingle.utility.hierarchy_env import LinearLeafNode, InternalNode, TreeEnvironment
from Jingle.worker.k8s_data_source import get_abs_unit_demand
# In Demo
# from Jingle.experiments.k8s.dummy.dummy_workload_generator import DummyWorkloadGenerator, PropAllocWorkloadGenerator

MICROSERVICES = ['assign-distribute']

def generate_env(env_descr, cluster_type, real_or_dummy):
    # Autoscaling environments ---------------------------------------------------------------------
    if env_descr == 'test1':
        env = generate_env_test1(real_or_dummy)
    elif env_descr == 'test2':
        env = generate_env_test2(real_or_dummy)
    elif env_descr == 'simple':
        env = generate_env_simple(real_or_dummy)
    else:
        raise NotImplementedError('Not implemented env_descr=%s yet.'%(env_descr))
    # Create workload info -------------------------------------------------------------------------
    # generate_workload_info_for_environment(env, cluster_type)
    return env



def generate_env_test1(real_or_dummy):
    """ Generate coding environment. """
    # job_dict = {ms: {'threshold': 0.99, 'util_scaling': 'linear', 'workload_type': real_or_dummy}
    #             for ms in MICROSERVICES}
    # return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)
    c1_info = {'name': 'coding-assign', 'threshold': 1, 'util_scaling': 'linear', 'workload_type': 'coding-assign'}
    # Create environment --------------------------------------
    root = InternalNode('root')
    child1 = _get_leaf_node_from_info_dict(c1_info, real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env


def generate_env_test2(real_or_dummy):
    """ Generate web-serving environment. """
    # job_dict = {ms: {'threshold': 0.99, 'util_scaling': 'linear', 'workload_type': real_or_dummy}
    #             for ms in MICROSERVICES}
    # return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)
    c1_info = {'name': 'assign-distribute', 'threshold': 1, 'util_scaling': 'linear', 'workload_type': 'assign-distribute'}
    # Create environment --------------------------------------
    root = InternalNode('root')
    child1 = _get_leaf_node_from_info_dict(c1_info, real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env


def generate_env_simple(real_or_dummy):
    """ A simple environment. """
    # Set thresholds ------------------------------------------
    c1_info = {'name': 'coding-assign', 'threshold': 0.99, 'util_scaling': 'linear', 'workload_type': 'coding-assign'}
    c2_info = {'name': 'assign-distribute', 'threshold': 1, 'util_scaling': 'linear', 'workload_type': 'assign-distribute'}
    # Create environment --------------------------------------
    root = InternalNode('root')
    child1 = _get_leaf_node_from_info_dict(c1_info, real_or_dummy)
    child2 = _get_leaf_node_from_info_dict(c2_info, real_or_dummy)
    root.add_children([child1, child2], [1, 1])
    env = TreeEnvironment(root, 1)
    return env


def get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy):
    """ job_dict is a dicitonary of dictionaries. """
    for key, val in job_dict.items():
        if not ('name' in val):
            val['name'] = key  # Make sure you add the name to the dictionary.
    children_as_list = [_get_leaf_node_from_info_dict(val, real_or_dummy)
                        for _, val in job_dict.items()]
    weights = [1] * len(children_as_list)
    root = InternalNode('root')
    root.add_children(children_as_list, weights)
    env = TreeEnvironment(root, 1)
    return env


def _get_leaf_node_from_info_dict(info_dict, real_or_dummy):
    """ Returns a leaf node from a dictionary. """
    leaf = LinearLeafNode(info_dict['name'], threshold=info_dict['threshold'],
                          util_scaling=info_dict['util_scaling'])
    if real_or_dummy == 'dummy':
        workload_type = 'dummy' + info_dict['workload_type']
    elif real_or_dummy == 'real':
        workload_type = info_dict['workload_type']
    else:
        raise ValueError(
            'Unknown value for real_or_dummy: %s.' % (real_or_dummy))
    leaf.update_workload_info({'workload_type': workload_type})
    if 'workload_info' in info_dict:
        leaf.update_workload_info(info_dict['workload_info'])
    return leaf


# Autoscaling environments ------------------------------------------------------------------
def generate_env_asds1(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'dataserving'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asds2(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.99, 'workload_type': 'dataserving'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asws1(real_or_dummy):
    """ Autoscaling environment with a single web search job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'websearch'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asws2(real_or_dummy):
    """ Autoscaling environment with a single web search job. """
    job_dict = {'j01': {'threshold': 0.99, 'workload_type': 'websearch'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asim1(real_or_dummy):
    """ Autoscaling environment with a single mem analytics serving job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'inmemoryanalytics'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asim2(real_or_dummy):
    """ Autoscaling environment with a single mem analytics job. """
    job_dict = {'j01': {'threshold': 0.999, 'workload_type': 'inmemoryanalytics'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)


# def generate_env_demo(cluster_type):
#     """ Generate a synthetic organisational tree. """
#     # Create the environment -----------------------------------------------------------------------
#     root = InternalNode('root')
#     child1 = InternalNode('c1')
#     child2 = LinearLeafNode('c2', threshold=2.1)
#     root.add_children([child1, child2], [1, 1])
#     child11 = LinearLeafNode('c11', threshold=0.6)
#     child12 = LinearLeafNode('c12', threshold=7.2)
#     child1.add_children([child11, child12], [2, 1])
#     env = TreeEnvironment(root, 1)
#
#     # Create the workload --------------------------------------------------------------------------
#     propworkgen = PropAllocWorkloadGenerator(cluster_type=cluster_type)
#     def generate_propalloc_workload_info(leaf: LinearLeafNode, workload_type):
#         [_, weight, _] = leaf.parent.children[leaf.name]
#         path = leaf.get_path_from_root()
#         workload_server_objs = propworkgen.generate_workload_server_objects(
#             app_name=path, threshold=leaf.threshold, app_weight=weight,
#             app_unit_demand=get_abs_unit_demand(leaf.threshold))
#         workload_cilantro_client_objs = propworkgen.generate_cilantro_client_objects(app_name=path)
#         k8s_objects = [*workload_server_objs, *workload_cilantro_client_objs]
#         leaf.update_workload_info({"k8s_objects": k8s_objects,
#                                    "workload_type": workload_type})
#     # Add workload information to clients ----------------------------------------------------------
#     generate_propalloc_workload_info(child2, 'dummy1')
#     generate_propalloc_workload_info(child11, 'dummy2')
#     generate_propalloc_workload_info(child12, 'dummy1')
#     return env
#
# def generate_env_demo1(cluster_type):
#     """ Generate a synthetic organisational tree. """
#     # Create the environment -----------------------------------------------------------------------
#     root = InternalNode('root')
#     child2 = LinearLeafNode('c2', threshold=2.1)
#     root.add_children([child2], [1])
#     env = TreeEnvironment(root, 1)
#
#     # Create the workload --------------------------------------------------------------------------
#     propworkgen = PropAllocWorkloadGenerator(cluster_type=cluster_type)
#     def generate_propalloc_workload_info(leaf: LinearLeafNode, workload_type):
#         [_, weight, _] = leaf.parent.children[leaf.name]
#         path = leaf.get_path_from_root()
#         workload_server_objs = propworkgen.generate_workload_server_objects(
#             app_name=path, threshold=leaf.threshold, app_weight=weight,
#             app_unit_demand=get_abs_unit_demand(leaf.threshold))
#         workload_Jingle_client_objs = propworkgen.generate_Jingle_client_objects(app_name=path)
#         k8s_objects = [*workload_server_objs, *workload_Jingle_client_objs]
#         leaf.update_workload_info({"k8s_objects": k8s_objects,
#                                    "workload_type": workload_type})
#     # Add workload information to clients ----------------------------------------------------------
#     generate_propalloc_workload_info(child2, 'dummy1')
#
#     return env

