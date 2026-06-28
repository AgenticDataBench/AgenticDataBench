"""
Smolagents Module - Alternative agent implementation using the smolagents library.

This module provides an alternative agent implementation that:
- Uses smolagents' CodeAgent for code execution
- Executes Python code in Docker containers via Jupyter kernel
- Supports websocket-based code execution with timeout handling
"""

from smolagents.agents import CodeAgent
from smolagents.remote_executors import PythonExecutor, RemotePythonExecutor
from smolagents.local_python_executor import CodeOutput
from contextlib import closing
from smolagents.remote_executors import _websocket_send_execute_request, _create_kernel_http
from smolagents.monitoring import LogLevel
from smolagents.utils import AgentError
import requests
import time
import json
import pickle
import base64
import select
from websocket import WebSocketTimeoutException, WebSocketConnectionClosedException, WebSocket

import requests
from typing import Any
from smolagents.models import ApiModel, ChatMessage, Tool, ChatMessageStreamDelta, TokenUsage
from smolagents import Generator, RunResult

from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.agent.base import BaseAgent

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.config import DASHSCOPE_API_KEY, DASHSCOPE_API_BASE

DEFAULT_TIME_OUT = 300

def _websocket_run_code_raise_errors(code: str, ws: WebSocket, logger) -> CodeOutput:
    """Run code over a websocket."""
    try:
        # Send execute request
        msg_id = _websocket_send_execute_request(code, ws)

        # Collect output and results
        outputs = []
        result = None
        is_final_answer = False

        start_time = time.time()

        while True:
            if time.time() - start_time > DEFAULT_TIME_OUT + 5:
                raise TimeoutError("No response from kernel (timeout)")
            
            rlist, _, _ = select.select([ws.sock], [], [], 0.1)
            if not rlist:
                continue

            try:
                msg = ws.recv()
                msg = json.loads(msg)
            except WebSocketTimeoutException:
                time.sleep(0.01)
                continue
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except WebSocketConnectionClosedException:
                raise

            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            # Skip unrelated messages
            if parent_msg_id != msg_id:
                continue
            msg_type = msg.get("msg_type", "")
            msg_content = msg.get("content", {})
            if msg_type == "stream":
                outputs.append(msg_content["text"])
            elif msg_type == "execute_result":
                result = msg_content["data"].get("text/plain", None)
            elif msg_type == "error":
                if msg_content.get("ename", "") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                    result = pickle.loads(base64.b64decode(msg_content.get("evalue", "")))
                    is_final_answer = True
                else:
                    raise AgentError("\n".join(msg_content.get("traceback", [])), logger)
            elif msg_type == "status" and msg_content["execution_state"] == "idle":
                break

        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

    except Exception as e:
        logger.log_error(f"Code execution failed: {e}")
        raise

def indent_code(code, n_spaces=4):
    """在每行前加 n_spaces 个空格"""
    indent = ' ' * n_spaces
    return '\n'.join(indent + line if line.strip() != '' else '' for line in code.splitlines())

class MyDockerExecutor(RemotePythonExecutor):
    def __init__(
        self,
        additional_imports: list[str],
        logger,
        env: DA_Agent_Env
    ):
        super().__init__(additional_imports, logger)

        host = 'localhost'
        port = env.kwargs['ports']['8888/tcp']
        self.base_url = f"http://{host}:{port}"

        # Wait for Jupyter to start
        self._wait_for_server()

        # Create new kernel via HTTP
        self.kernel_id = _create_kernel_http(f"{self.base_url}/api/kernels", logger)
        self.ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            wrapped_code = f'''
import signal

def handler(signum, frame):
    raise TimeoutError("Execution timeout")

signal.signal(signal.SIGALRM, handler)
signal.alarm({DEFAULT_TIME_OUT})

try:
{indent_code(code, 4)}
finally:
    signal.alarm(0)
'''
            ws.settimeout(0)
            return _websocket_run_code_raise_errors(wrapped_code, ws, self.logger)

    def cleanup(self):
        """Clean up the Docker container and resources."""
        pass

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self):
        retries = 0
        jupyter_ready = False
        while not jupyter_ready and retries < 10:
            try:
                if requests.get(f"{self.base_url}/api/kernelspecs", timeout=2).status_code == 200:
                    jupyter_ready = True
                else:
                    self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            except requests.RequestException:
                self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            if not jupyter_ready:
                time.sleep(1)
                retries += 1

class MyCodeAgent(CodeAgent):
    def create_python_executor(self) -> PythonExecutor:
        if self.managed_agents:
            raise Exception("Managed agents are not yet supported with remote code execution.")
        remote_executors = {
            "docker": MyDockerExecutor,
        }
        install_imports = [package for package in self.additional_authorized_imports if package != '*']
        return remote_executors[self.executor_type](
            install_imports, self.logger, **self.executor_kwargs
        )
            
class MyServerModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        raise NotImplementedError("MyServerModel does not support streaming generation.")

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            extra_body={"enable_thinking": True},
            **kwargs,
        )
        self._apply_rate_limit()
        response = self.client.chat.completions.create(**completion_kwargs)
        data = response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        return ChatMessage.from_dict(
            data,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
    
class PromptAgent(BaseAgent):
    # smolagents only uses model and max_steps; the other knobs are accepted
    # to keep the construction interface uniform with the other agents. No
    # cross-task state, so __init__ is inherited from BaseAgent.

    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.instruction = self.env.task_config['question']
        wrapped_model = MyServerModel(
            model_id=self.model,
            api_base=DASHSCOPE_API_BASE,
            api_key=DASHSCOPE_API_KEY
        )
        self.code_agent = MyCodeAgent(
            tools=[],
            model=wrapped_model,
            additional_authorized_imports=['*'],
            max_steps=self.max_steps,
            planning_interval=3,
            return_full_result=True,
            max_print_outputs_length=1000,
            executor_type="docker",
            executor_kwargs={"env": self.env}
        )
        self.trajectory = []

    def run(self):
        assert self.env is not None, "Environment is not set."
        task = self.instruction + f"""

You are working in the directory: {self.work_dir}.
All required data files are available in this directory."""
        def image_post_process(output_file_name):
            if output_file_name in self.env.task_config['output_file_name']:
                return output_file_name
            return None
        image_file_names = []
        for post_process_f in self.env.post_process_func:
            output_file_name = eval(post_process_f)
            if output_file_name:
                image_file_names.append(output_file_name)
        if image_file_names:
            task += f"""

### Plotting (REQUIRED)

If you create a matplotlib plot, you MUST call:

    Plotprocess.plot_process(fig, "<image_file_name>")

Use ONLY these file names:
{", ".join(image_file_names)}

Rules:
- Call AFTER plotting is complete
- Call BEFORE saving the figure
- Use: fig = plt.gcf()
- Replace <image_file_name> with one from the list above

Example:
```python
from image import Plotprocess
import matplotlib.pyplot as plt

# plotting code ...

fig = plt.gcf()
Plotprocess.plot_process(fig, "{image_file_names[0]}")
```"""
        result: RunResult = self.code_agent.run(task)
        self.trajectory = result.steps
        if result.state != 'success':
            return False, f"ERROR: {result.state}"
        return True, str(result.output)

    def get_trajectory(self):
        trajectory_log = {
            "task": self.instruction,
            "trajectory": self.trajectory
        }
        return trajectory_log
