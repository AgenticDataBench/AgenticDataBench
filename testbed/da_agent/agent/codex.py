"""
Codex Agent Module - Agent implementation driving the Codex CLI.

This module provides an agent that:
- Generates a ~/.codex/config.toml for the Codex CLI targeting an
  OpenAI-compatible provider (e.g. DashScope) and injects the API key
- Runs `codex exec` inside the agent's Docker container with a timeout
- Parses Codex's JSONL event stream into a normalized trajectory
- Retries on recoverable errors (429 rate limits, 400 invalid parameter)
"""

import json
import logging
import os
import getpass
import subprocess
import time
from pathlib import Path
import sys

# Make the project root importable so `utils.config` can be resolved.
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.config import OPENAI_API_KEY, OPENAI_API_BASE

from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.agent.base import BaseAgent

logger = logging.getLogger("da_agent")

DEFAULT_TIME_OUT = 3600
MAX_OBS_LENGTH = 3000


class PromptAgent(BaseAgent):
    # Codex drives the `codex` CLI; only model is used from the shared config
    # (the timeout uses DEFAULT_TIME_OUT instead of max_steps). No cross-task
    # state, so __init__ is inherited from BaseAgent.

    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.instruction = self.env.task_config['question']
        self.trajectory = []
        self.raw_output = ""
        self.event_timestamps = []

    def _build_task_prompt(self):
        task = self.instruction
        task += f"\n\nYou are working in the directory: {self.work_dir}."
        task += " All required data files are available in this directory."
        task += " Complete the task and ensure all output files are saved in this directory."

        image_file_names = self._get_image_file_names()
        if image_file_names:
            task += self._build_plotting_instructions(image_file_names)

        return task

    def _get_image_file_names(self):
        image_file_names = []
        for post_process_f in self.env.post_process_func:
            def image_post_process(output_file_name):
                if output_file_name in self.env.task_config.get('output_file_name', []):
                    return output_file_name
                return None
            output_file_name = eval(post_process_f)
            if output_file_name:
                image_file_names.append(output_file_name)
        return image_file_names

    def _build_plotting_instructions(self, image_file_names):
        return f"""
### Plotting (REQUIRED)

If you create a matplotlib plot, you MUST call:

    from image import Plotprocess
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

    def _write_wrapper_script(self):
        # Read API config from utils.config (LLM settings live there).
        # Build TOML config for ~/.codex/config.toml
        # Codex CLI requires model_providers section with wire_api="chat"
        # for non-OpenAI providers (e.g., DashScope)
        config_toml = f'''model_provider = "DashScope"

[model_providers.DashScope]
name = "DashScope"
base_url = "{OPENAI_API_BASE}"
env_key = "OPENAI_API_KEY"
wire_api = "chat"
'''

        wrapper_code = f'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import threading
import time

# Set API key as env var (Codex reads it via env_key in config.toml)
os.environ["OPENAI_API_KEY"] = "{OPENAI_API_KEY}"

# Write config.toml before launching codex
config_dir = os.path.expanduser("~/.codex")
os.makedirs(config_dir, exist_ok=True)
with open(os.path.join(config_dir, "config.toml"), "w") as f:
    f.write("""{config_toml}""")

with open("{self.work_dir}/.task_prompt.txt") as f:
    prompt = f.read()

MAX_RETRIES = 3
RETRY_BACKOFF_429 = 30  # seconds to wait on rate limit
RETRY_BACKOFF_400 = 10   # seconds to wait on invalid parameter error

current_proc = None

def timeout_handler():
    global current_proc
    if current_proc:
        current_proc.terminate()
        kill_timer = threading.Timer(300, current_proc.kill)
        kill_timer.daemon = True
        kill_timer.start()

timer = threading.Timer({DEFAULT_TIME_OUT}, timeout_handler)
timer.daemon = True
timer.start()

def run_codex(cmd_args, is_resume=False):
    global current_proc
    cmd = ["stdbuf", "-oL"] + cmd_args
    current_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    last_line = ""
    for line in iter(current_proc.stdout.readline, b""):
        decoded = line.decode("utf-8", errors="ignore")
        sys.stdout.write(decoded)
        sys.stdout.flush()
        last_line = decoded.strip()

    # Read remaining stderr after stdout is exhausted
    stderr = current_proc.stderr.read()
    if stderr:
        stderr_text = stderr.decode("utf-8", errors="ignore").strip()
        if stderr_text:
            last_line = stderr_text.split("\\n")[-1]
            sys.stderr.write(stderr_text + "\\n")
            sys.stderr.flush()

    current_proc.wait()
    return current_proc.returncode, last_line

def is_retryable_error(last_line):
    try:
        entry = json.loads(last_line)
        if entry.get("type") == "turn.failed":
            error_msg = json.dumps(entry.get("error", {{}}))
            if "429" in error_msg:
                return "429"
            if "400" in error_msg or "InvalidParameter" in error_msg:
                return "400"
    except (json.JSONDecodeError, TypeError):
        pass
    return None

# Initial run
cmd_args = ["codex", "exec", prompt,
            "--model", "{self.model}",
            "--sandbox", "danger-full-access",
            "--skip-git-repo-check",
            "--json"]

exit_code, last_line = run_codex(cmd_args)

# Retry on recoverable errors
for attempt in range(MAX_RETRIES):
    error_type = is_retryable_error(last_line)
    if not error_type:
        break

    backoff = RETRY_BACKOFF_429 if error_type == "429" else RETRY_BACKOFF_400
    print(f"[RETRY] turn.failed ({{error_type}}), waiting {{backoff}}s before retry {{attempt+1}}/{{MAX_RETRIES}}...", flush=True)
    time.sleep(backoff)

    exit_code, last_line = run_codex(cmd_args, is_resume=True)

sys.exit(exit_code)
'''
        wrapper_path = os.path.join(self.env.mnt_dir, ".run_codex.py")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_code)

    def run(self):
        assert self.env is not None, "Environment is not set."

        task_prompt = self._build_task_prompt()
        container_name = self.env.container.name

        # Write task prompt and wrapper script to mounted directory
        task_path = os.path.join(self.env.mnt_dir, ".task_prompt.txt")
        with open(task_path, "w") as f:
            f.write(task_prompt)
        self._write_wrapper_script()

        # Execute wrapper script inside the container as the non-root user named
        # after the host user.
        process = subprocess.Popen(
            ["docker", "exec", "--user", getpass.getuser(), str(container_name),
             "python3", f"{self.work_dir}/.run_codex.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        output_lines = []
        self.event_timestamps = []
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    decoded = line.decode("utf-8", errors="ignore")
                    output_lines.append(decoded)
                    self.event_timestamps.append(time.time())
                    logger.debug("Codex: %s", decoded.strip())
        except Exception as e:
            process.kill()
            logger.error("Error running Codex: %s", e)
            self.raw_output = "".join(output_lines)
            self._parse_trajectory()
            return False, f"Error: {e}"

        self.raw_output = "".join(output_lines)
        exit_code = process.returncode

        self._parse_trajectory()

        # Codex exec --json exits 0 even when the turn is cut off (e.g. step
        # limit). Detect by trajectory tail: a normal turn ends with
        # turn.completed (normalized to "result"), an error event ("error"),
        # a turn.failed event, or unparsed stdout fragments ("raw"). Anything
        # else means the turn was interrupted mid-step.
        NORMAL_END_TYPES = {"result", "error", "raw", "turn.failed"}
        if exit_code == 0:
            last_type = self.trajectory[-1].get("type") if self.trajectory else None
            if last_type in NORMAL_END_TYPES:
                return True, "Task completed"
            else:
                return False, f"Agent stopped without turn completion (last_type={last_type})"
        else:
            return False, f"Agent exited with code {exit_code}"

    def _parse_trajectory(self):
        self.trajectory = []

        # With --json, Codex outputs JSONL events line-by-line
        lines = self.raw_output.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                normalized = self._normalize_jsonl_entry(entry)
                if i < len(self.event_timestamps):
                    ts = self.event_timestamps[i]
                    prev_ts = self.event_timestamps[i - 1] if i > 0 else ts
                    normalized["timing"] = {
                        "start_time": prev_ts,
                        "end_time": ts,
                        "duration": ts - prev_ts,
                    }
                if normalized.get("type") not in ("thread_started", "turn_started"):
                    self.trajectory.append(normalized)
            except json.JSONDecodeError:
                step = {"type": "raw", "content": line}
                if i < len(self.event_timestamps):
                    ts = self.event_timestamps[i]
                    prev_ts = self.event_timestamps[i - 1] if i > 0 else ts
                    step["timing"] = {
                        "start_time": prev_ts,
                        "end_time": ts,
                        "duration": ts - prev_ts,
                    }
                self.trajectory.append(step)

    def _normalize_jsonl_entry(self, entry):
        if not isinstance(entry, dict):
            return {"type": "raw", "content": str(entry)}

        event_type = entry.get("type", "unknown")

        if event_type == "thread.started":
            return {"type": "thread_started"}

        elif event_type == "turn.started":
            return {"type": "turn_started"}

        elif event_type == "turn.completed":
            result = {"type": "result"}
            usage = entry.get("usage", {})
            if usage:
                result["usage"] = usage
            return result

        elif event_type == "item.completed":
            item = entry.get("item", {})
            item_type = item.get("type", "unknown")

            if item_type == "agent_message":
                text = item.get("text", "")
                return {"type": "assistant", "content": text}

            elif item_type == "command_execution":
                cmd = item.get("command", "")
                output = item.get("aggregated_output", "")
                exit_code = item.get("exit_code")
                if len(output) > MAX_OBS_LENGTH:
                    output = output[:MAX_OBS_LENGTH] + f"\n... (truncated, original {len(output)} chars)"
                result = {
                    "type": "assistant",
                    "code_action": cmd,
                    "observations": f"Execution logs:\n{output}",
                }
                if exit_code is not None:
                    result["exit_code"] = exit_code
                return result

            elif item_type == "todo_list":
                items = item.get("items", [])
                return {"type": "assistant", "content": f"Todo: {json.dumps(items)}"}

            return {"type": item_type, "content": json.dumps(item)}

        elif event_type == "item.started":
            item = entry.get("item", {})
            return {"type": "item_started", "content": json.dumps(item)}

        elif event_type == "error":
            return {"type": "error", "content": entry.get("message", "")}

        return {"type": event_type, "content": json.dumps(entry)}

    def get_trajectory(self):
        return {
            "task": self.instruction,
            "trajectory": self.trajectory
        }