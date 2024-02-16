'''
    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
'''

from typing import List, Dict, TypeVar

from Jingle.scheduler.workload_generator.k8s_utils import get_template_deployment, get_template_service
from kubernetes.client import V1Deployment, V1Service, V1ObjectMeta, V1DeploymentSpec, V1PodTemplateSpec, \
    V1LabelSelector, V1PodSpec, V1Container, V1ContainerPort, V1ServiceSpec, V1ServicePort, V1SecurityContext, \
    V1Capabilities, V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector, V1Lifecycle, V1ExecAction, \
    V1VolumeMount, V1Volume, V1EmptyDirVolumeSource, V1Probe, V1ResourceFieldSelector, V1ResourceRequirements

CODING_PORTS = {
    "containerPort": 6000
}

Jingle_PORTS = {
    'grpc': 10000
}

def convert_dictargs_to_listargs(dict_args: Dict):
    # Converts dictionary args to sequential args for cmd line
    listargs = []
    for k,v in dict_args.items():
        listargs.append(k)
        listargs.append(v)
    return listargs

def get_coding_server_template_deployment(app_name: str,
                                        *args,
                                        **kwargs
                                        ) -> V1Deployment:
    """
    Defines the coding assignment server/workload objects. This is deployment that gets scaled with more resources.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    is_workload = "true"
    default_replicas = 1
    container_image = "yixuannine/coding-assignment:latest"
    container_ports = list(CODING_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = ["/bin/bash", "-c", "--"]
    server_envs = [
        V1EnvVar("POD_IP", value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP"))),
            V1EnvVar("MY_CPU_REQUEST", value_from=V1EnvVarSource(
                resource_field_ref=V1ResourceFieldSelector(resource="requests.cpu")))
            ]

    server_deployment = get_template_deployment(app_name=app_name,
                                                is_workload=is_workload,
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=container_command)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    podspec = server_deployment.spec.template.spec

    # Add dshm volume
    dshm_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(medium="Memory"), name="dshm")
    podspec.volumes = [dshm_volume]
    dshm_mount = V1VolumeMount(mount_path="/dev/shm", name="dshm")

    # Set resource requirements
    resreq = V1ResourceRequirements(requests={"cpu": "500m",
                                              "memory": "512m"})

    container = podspec.containers[0]
    container.volume_mounts = [dshm_mount]
    container.resources = resreq

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = server_envs
    else:
        current_envs.extend(server_envs)

    return server_deployment


# ==========================================================
# ================= CLIENT OBJECTS =========================
# ==========================================================

def get_coding_client_template_deployment(app_name: str,
                                        coding_client_cmd: List[str],
                                        coding_client_args: Dict[str, str],
                                        ca_client_cmd: List[str],
                                        ca_client_args: Dict[str, str],
                                        *args,
                                        Jingle_image: str = "yixuannine/Jinglescheduler:latest",
                                        **kwargs) -> V1Deployment:
    """
    Defines the deployment for the cray client.
    A pod contains two containers - coding client and ca client.
    Also not counted as a workload, thus is_workload is false.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    client_name = app_name + "-client"
    is_workload = "false"
    default_replicas = 1
    container_image = "yixuannine/coding-assignment:latest"
    container_ports = []
    container_image_pull_policy = "Always"

    envs = [V1EnvVar("POD_IP", value_from=V1EnvVarSource(
        field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    coding_listargs = convert_dictargs_to_listargs(coding_client_args)

    client_deployment = get_template_deployment(app_name=client_name,
                                                is_workload=is_workload,
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=coding_client_cmd,
                                                container_args=coding_listargs)

    # Add volume to the pod spec
    podspec = client_deployment.spec.template.spec
    log_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(), name="log-share")
    podspec.volumes = [log_volume]
    podspec.termination_grace_period_seconds = 0

    # Create volume mount for shared log mount
    log_mount = V1VolumeMount(mount_path="/Jinglelogs", name="log-share")

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    coding_client_container = podspec.containers[0]
    coding_client_container.name = "coding"
    coding_client_container.volume_mounts = [log_mount]

    current_envs = coding_client_container.env
    if not current_envs:
        coding_client_container.env = envs
    else:
        current_envs.extend(envs)

    Jingle_listargs = convert_dictargs_to_listargs(ca_client_args)

    # Define Jingle Client container and add it to list of containers
    Jingle_container = V1Container(name="ca-client",
                                      image=Jingle_image,
                                      image_pull_policy="Always",
                                      volume_mounts=[log_mount],
                                      ports=[V1ContainerPort(container_port=p) for p in CODING_PORTS.values()],
                                      command=ca_client_cmd,
                                      args=Jingle_listargs
                                      )
    podspec.containers.append(Jingle_container)
    return client_deployment
