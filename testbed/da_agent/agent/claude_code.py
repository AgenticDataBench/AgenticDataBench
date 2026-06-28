"""
Claude Code Agent Module - Agent implementation driving the Claude Code CLI.

This module provides an agent that:
- Writes a task prompt and a wrapper script to the agent's Docker container
- Runs the `claude` CLI inside the container with a timeout, reading Anthropic
  API credentials from the container's environment (injected by run.py)
- Parses Claude Code's stream-json (or json) output into a normalized trajectory,
  capturing per-step timing, tool uses, and token usage
"""

import json
import logging
import os
import getpass
import subprocess
import time
from da_agent.envs.da_agent import DA_Agent_Env
from da_agent.agent.base import BaseAgent

logger = logging.getLogger("da_agent")

DEFAULT_TIME_OUT = 3600  # 60 minutes


class PromptAgent(BaseAgent):
    # Claude Code drives the `claude` CLI; only model is used from the shared
    # config (the timeout uses DEFAULT_TIME_OUT instead of max_steps). No
    # cross-task state, so __init__ is inherited from BaseAgent.

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
        wrapper_code = f'''#!/usr/bin/env python3
import subprocess
import sys
import threading

with open("{self.work_dir}/.task_prompt.txt") as f:
    prompt = f.read()

proc = subprocess.Popen(
    ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
     "--model", "{self.model}",
     "--dangerously-skip-permissions"],
    stdout=sys.stdout, stderr=sys.stderr
)

def timeout_handler():
    proc.terminate()
    kill_timer = threading.Timer(300, proc.kill)
    kill_timer.daemon = True
    kill_timer.start()

timer = threading.Timer({DEFAULT_TIME_OUT}, timeout_handler)
timer.daemon = True
timer.start()

sys.exit(proc.wait())
'''
        wrapper_path = os.path.join(self.env.mnt_dir, ".run_claude.py")
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
             "python3", f"{self.work_dir}/.run_claude.py"],
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
                    logger.debug("Claude Code: %s", decoded.strip())
        except Exception as e:
            process.kill()
            logger.error("Error running Claude Code: %s", e)
            self.raw_output = "".join(output_lines)
            self._parse_trajectory()
            return False, f"Error: {e}"

        self.raw_output = "".join(output_lines)
        exit_code = process.returncode

        self._parse_trajectory()

        if exit_code == 0:
            return True, "Task completed"
        else:
            return False, f"Agent exited with code {exit_code}"

    def _parse_trajectory(self):
        self.trajectory = []

        # Try parsing as a single JSON array (--output-format json)
        try:
            entries = json.loads(self.raw_output.strip())
            if isinstance(entries, list):
                for entry in entries:
                    normalized = self._normalize_entry(entry)
                    self.trajectory.append(normalized)
                return
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL parsing (--output-format stream-json)
        lines = self.raw_output.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                normalized = self._normalize_entry(entry)
                # Add timing info from recorded timestamps (stream-json only)
                if i < len(self.event_timestamps):
                    ts = self.event_timestamps[i]
                    prev_ts = self.event_timestamps[i - 1] if i > 0 else ts
                    normalized["timing"] = {
                        "start_time": prev_ts,
                        "end_time": ts,
                        "duration": ts - prev_ts,
                    }
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

    def _normalize_entry(self, entry):
        entry_type = entry.get("type", "unknown")

        if entry_type == "system":
            subtype = entry.get("subtype", "")
            if subtype == "init":
                return {
                    "type": "system_init",
                    "session_id": entry.get("session_id", ""),
                    "model": entry.get("model", ""),
                    "cwd": entry.get("cwd", ""),
                }
            return {"type": "system", "subtype": subtype}

        if entry_type == "assistant":
            message = entry.get("message", {})
            msg_id = message.get("id", "")
            content = message.get("content", [])
            text_parts = []
            code_action = None
            tool_uses = []

            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    tool_uses.append({"name": tool_name, "input": tool_input})
                    if tool_name == "Bash" and "command" in tool_input:
                        code_action = tool_input["command"]

            result = {"type": "assistant", "content": "\n".join(text_parts)}
            if msg_id:
                result["msg_id"] = msg_id
            if tool_uses:
                result["tool_uses"] = tool_uses
            if code_action:
                result["code_action"] = code_action
            # Extract per-message usage if available
            usage = message.get("usage", {})
            if usage:
                result["usage"] = usage
            return result

        elif entry_type == "tool_result":
            content = entry.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)
            result = {"type": "tool_result", "observations": f"Execution logs:\n{content}"}
            return result

        elif entry_type == "result":
            result_entry = {
                "type": "result",
                "subtype": entry.get("subtype", ""),
                "is_error": entry.get("is_error", False),
                "result": entry.get("result", ""),
                "stop_reason": entry.get("stop_reason", ""),
                "duration_ms": entry.get("duration_ms", 0),
                "num_turns": entry.get("num_turns", 0),
            }
            # Extract aggregate usage from the result event
            usage = entry.get("usage", {})
            if usage:
                result_entry["usage"] = usage
            return result_entry

        return entry

    def get_trajectory(self):
        return {
            "task": self.instruction,
            "trajectory": self.trajectory
        }