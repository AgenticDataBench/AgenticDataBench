"""
DA_Agent_Env Module - Docker environment manager for the data analysis agent.

This module manages the execution environment:
- Creates and manages Docker containers
- Handles action execution (Bash, Python, SQL commands)
- Sets up the workspace with task-specific data files
- Tracks file changes for output evaluation

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/envs/da_agent.py
"""

import logging
import os
import getpass
import time
from typing import Callable, Any, Optional
import glob

from typing import Dict
from docker.models.containers import Container
from docker.client import DockerClient
from docker.errors import ImageNotFound
import gymnasium as gym
import pathlib, docker, time
from da_agent.controllers.python import PythonController
from da_agent.controllers.setup import SetupController
from da_agent.envs.utils import *
from da_agent.agent.action import Bash, Action, Terminate, Python, SQL

logger = logging.getLogger("da_agent.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


# constants
START_UP_DELAY = 2 # start up delay for docker container
DEFAULT_TIME_OUT = 60 # default waiting time for each action
MAX_OBS_LENGTH = 3000
EMPTY_DATA_PATH = 'da_agent/data/empty' # an empty data directory
DEFAULT_IMAGE_DIR = 'da_agent/images' # default directory to store docker images
DEFAULT_WORK_DIR = '/workspace' # default working directory in the container
DEFAULT_MNT_DIR = 'da_agent/mnt' # default directory to copy and mount data path, also the output directory
TASK_FINISHED = "task_finished" # infos key
ACTION_EXEC = "action_executed" # infos key


class DA_Agent_Env(gym.Env):
    """
    DesktopEnv with OpenAI Gym interface.
    Fixme: refactor the logic when implementing the multi-process version
    """
    def __init__(self, env_config, task_config, source_dir, cache_dir, mnt_dir):
        """
        Args:
            path_to_vm (str): path to .vmx file
            action_space (str): "computer_13" | "pyautogui"

            task_config (Dict[str, Any]): manages task configs integratedly,
              including
              * base snapshot
              * task id (uuid)
              * instruction

            tmp_dir (str): temporary directory to store trajectory stuffs like
              the extracted screenshots
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
        """
        super().__init__()
        self.task_config = task_config
        self.cache_dir_base = cache_dir
        self.container_name = env_config['init_args']['name']
        self.image_name = env_config['image_name']
        self.source_dir = source_dir
        self.mnt_dir = mnt_dir
        self.work_dir = DEFAULT_WORK_DIR
        self.kwargs = env_config['init_args']

        self._set_task_info(task_config)
        logger.info("Initializing...")
        self._construct_container()
        
        self.controller = PythonController(container=self.container, work_dir=self.work_dir)
        self.setup_controller = SetupController(container=self.container, cache_dir=self.cache_dir)
        
        logger.info("Setting up environment...")
        
        dir = os.path.join(self.source_dir, self.domain)
        assert os.path.isdir(dir), f"Task directory {dir} does not exist."
        
        # Record source file names before copying
        self.source_files = set()
        for root, dirs, files in os.walk(dir):
            # Skip dabench directory if it exists in source
            if 'dabench' in dirs:
                dirs.remove('dabench')
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), dir)
                self.source_files.add(rel_path)
        logger.info(f"Recorded {len(self.source_files)} source files")

        self.source_files_prompt = self.build_tree_prompt()
        
        self.setup_controller.setup_cp_dir(dir)
        if any([post_func for post_func in self.post_process_func if post_func.startswith("image_post_process")]):
            self.setup_controller.setup_cp_dir('da_agent/configs/scripts/image.py')
        self.init_files_hash = self._get_env_files_hash()
        time.sleep(2)
        logger.info("Environment setup complete.")

    def build_tree_prompt(self):
        tree = {}

        # build nested dict tree
        for path in sorted(self.source_files):
            parts = path.split(os.sep)
            node = tree
            for part in parts:
                node = node.setdefault(part, {})

        lines = [self.work_dir]

        def render(node, prefix=""):
            items = list(node.items())
            for i, (name, child) in enumerate(items):
                connector = "`-- " if i == len(items) - 1 else "|-- "
                lines.append(prefix + connector + name)
                if child:
                    extension = "    " if i == len(items) - 1 else "|   "
                    render(child, prefix + extension)

        render(tree)
        return "\n".join(lines)
        
    def _set_task_info(self, task_config: Dict[str, Any]):
        self.domain: str = task_config["domain"]
        self.task_id: str = task_config['id']
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        # os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["question"]
        self.post_process_func = [post_process_f for post_process_f in task_config["post_process_func"] if post_process_f] if "post_process_func" in task_config else []
        
    def close(self):
        self.container.stop()
        self.container.remove()
        logger.info(f"Container {self.container_name} stopped and removed.")
        
    def _construct_container(self):
        client = docker.from_env()
        container_name = self.container_name
        #### delete existing container
        try:
            container = client.containers.get(container_name)
            container.stop()
            container.remove()
            print(f"Container {container_name} stopped and removed.")
        except docker.errors.NotFound:
            pass
        except docker.errors.APIError as e:
            pass
        
        create_folder_if_not_exists(self.mnt_dir)
        src_dir = pathlib.Path(self.mnt_dir).absolute().__str__()
        delete_files_in_folder(self.mnt_dir)
        
        volumes = {src_dir: {'bind': self.work_dir, 'mode': 'rw'}}
        allowed_params = ['command', 'ports', 'restart_policy', 'entrypoint', 'hostname', 'domainname', 'name', 'user', 'mac_address', 'platform', 'network_mode', 'network_disabled', 'healthcheck', "environment"]
        kwargs = {k: self.kwargs[k] for k in self.kwargs if k in allowed_params}
        extra_params = {'detach': True, 'tty': True, 'stdout': True, 'stderr': True, 'stdin_open': True, **kwargs}

        try:
            client: DockerClient = docker.from_env()
            image = client.images.get(self.image_name)
            self.container: Container = client.containers.run(image=image, volumes=volumes, **extra_params)
        except ImageNotFound as e:
            dockerfile_path = os.path.join(DEFAULT_IMAGE_DIR, self.image_name)
            if os.path.exists(dockerfile_path):
                logger.info(f"Image {self.image_name} not found, try to build from dockerfile {dockerfile_path} ...")
                # Match the container's non-root user to the host user (name + uid)
                # so bind-mounted volumes keep correct ownership. Dockerfiles that
                # don't declare these ARGs ignore them harmlessly.
                buildargs = {
                    "HOST_USER": getpass.getuser(),
                    "HOST_UID": str(os.getuid()),
                }
                image = client.images.build(path=dockerfile_path, tag=self.image_name, rm=True, buildargs=buildargs)[0]
            else:
                logger.info(f"Image {self.image_name} not found, try to pull from Dockerhub ...")
                image = client.images.pull(self.image_name)[0]
            self.container: Container = client.containers.run(image=image, volumes=volumes, **extra_params)
        except Exception as e:
            logger.info(f"Failed to construct container from image {self.image_name} with error: {e}")
            raise e

        time.sleep(START_UP_DELAY)
        logger.info(f"Connected to container[name={self.container.name}, id={self.container.id}] from image {self.image_name} ...")    
        
        return self.container

    def _get_env_files_hash(self) -> Dict[str, str]:
        """
        Returns:
            Dict[str, str]: a dictionary of the hash of the files in the
              environment
        """
        files_hash = {}
        for root, dirs, files in os.walk(self.mnt_dir):
            for f in files:
                file_path = os.path.join(root, f)
                files_hash[file_path] = calculate_sha256(file_path)
        return files_hash
    
    
    def post_process(self):
        """
        Evaluate whether the task is successfully completed.
        """
        diff_files = self._find_diff_files_init(self.init_files_hash)

        def image_post_process(output_file_name: str) -> Optional[str]:
            if output_file_name in self.task_config['output_file_name']:
                print(output_file_name)
                mnt_files = os.listdir(self.mnt_dir)
                # assert len(png_files) > 0, 'Agent fails to plot image'
                if output_file_name not in mnt_files:
                    error = f'Agent fails to plot image {output_file_name}'
                    return error
                
                output_json = os.path.splitext(output_file_name)[0] + ".json"
                # output_npy = os.path.splitext(output_file_name)[0] + ".npy"
                if output_json not in mnt_files:
                    error = f'Agent fails to generate json file {output_json} for the plotted image {output_file_name}'
                    return error
            return None

        errors = []
        for post_process_f in self.post_process_func:
            error = eval(post_process_f)
            if error:
                errors.append(error)

        return {**diff_files, "error": errors}

    def _find_diff_files_init(self, init_file_dict)-> Dict:
        init_file_paths = init_file_dict.keys()
        added_files_list = []
        changed_files_list = []
        for root, dirs, files in os.walk(self.mnt_dir):
            for f in files:
                file_path = os.path.join(root, f)
                if file_path not in init_file_paths:
                    added_files_list.append(file_path)
                else:
                    if init_file_dict[file_path] != calculate_sha256(file_path):
                        changed_files_list.append(file_path)
        return {"added_files": added_files_list, "changed_files": changed_files_list}
    
    def _cleanup_source_files(self):
        """Remove source files from output directory to save memory."""
        if not hasattr(self, 'source_files'):
            return
        
        cleanup_enabled = os.environ.get('DA_CLEANUP_SOURCE_FILES', 'true').lower() == 'true'
        if not cleanup_enabled:
            logger.info("Source file cleanup is disabled")
            return
            
        removed_count = 0
        files_to_remove = self.source_files.copy()
        files_to_remove.add('image.py')
        for rel_path in files_to_remove:
            file_path = os.path.join(self.mnt_dir, rel_path)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    removed_count += 1
                    logger.debug(f"Removed source file: {rel_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {rel_path}: {e}")

        if self.domain == 'transportation':
            patterns = [
                '**/taxi_zones/taxi_zones.cpg',
                '**/taxi_zones/taxi_zones.dbf',
                '**/taxi_zones/taxi_zones.prj',
                '**/taxi_zones/taxi_zones.shp',
                '**/taxi_zones/taxi_zones.shx',
            ]

            for pattern in patterns:
                full_pattern = os.path.join(self.mnt_dir, pattern)
                for file_path in glob.glob(full_pattern, recursive=True):
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            removed_count += 1
                            logger.debug(f"Removed source file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {file_path}: {e}")
        
        # Clean up empty directories (but keep dabench directory)
        for root, dirs, files in os.walk(self.mnt_dir, topdown=False):
            # Skip dabench directory and mnt_dir itself
            if root == self.mnt_dir or root.endswith('/dabench') or '/dabench/' in root:
                continue
            if not files and not dirs:
                try:
                    os.rmdir(root)
                    logger.debug(f"Removed empty directory: {root}")
                except:
                    pass
                    
        logger.info(f"Cleaned up {removed_count} source files from output directory")
    
    def step(self, action: Action):
        try:
            with timeout(DEFAULT_TIME_OUT,"Action execution time exceeded!"):
                done = False
                if isinstance(action, Bash):
                    observation = self.execute_code_action(action)
                elif isinstance(action, SQL):
                    observation = self.execute_sql_action(action)
                elif isinstance(action, Python):
                    observation = self.execute_python_action(action)
                elif isinstance(action, Terminate):
                    observation = "Terminate"
                    done = True
                else:
                    raise ValueError(f"Unrecognized action type {action.action_type} !")
        except TimeoutError as e:
            observation = str(e)
        
        observation = self._handle_observation(observation)
        # logger.info("Observation: %s", observation)
        return observation, done
    
    def _handle_observation(self, observation):
        max_length = MAX_OBS_LENGTH  
        if len(observation) > max_length:
            truncated_observation = observation[:max_length] + "\n[Observation too long, truncated; Try other commands to get the left part.]"
            return truncated_observation
        return observation


    def execute_code_action(self, action: Bash):
        """ Execute action in bash shell """
        
        obs = self.controller.execute_command(action.code)
        if obs is None or obs == '':
            obs = "Command executed successfully. No output."
        
        return obs

    def execute_python_action(self, action: Python):
        """ Execute action in python """
        obs = self.controller.execute_python_file(action.filepath, action.code)
        if obs is None or obs == '':
            obs = f"{action.filepath} executed successfully. No output."
        
        return obs
    
    def execute_sql_action(self, action: Python):
        """ Execute action in sql"""
        obs = self.controller.execute_sql_code(action.file_path, action.code, action.output)
        if obs is None or obs == '':
            obs = f"SQL command executed successfully. No output."
        
        return obs
