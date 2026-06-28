"""
PromptAgent Module - Core agent class implementing the thought-action-observation loop.

This module implements the main agent logic that:
- Manages conversation history with the LLM
- Parses actions from LLM responses
- Executes actions in the Docker environment
- Tracks the agent's trajectory (thoughts, actions, observations)

Reference: https://github.com/yiyihum/da-code/tree/main/da_agent/agent/agents.py
"""

import logging
import re
import time
from typing import Dict, List
from da_agent.agent.prompts import SYS_PROMPT_IN_OUR_CODE
from da_agent.agent.action import Bash, Action, Terminate, Python, SQL
from da_agent.envs.da_agent import DA_Agent_Env
from typing import Dict, List

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from utils.llm_client import QwenClient
from da_agent.agent.base import BaseAgent



logger = logging.getLogger("da_agent")


class PromptAgent(BaseAgent):
    def __init__(
        self,
        model,
        max_tokens,
        top_p,
        temperature,
        max_memory_length,
        max_steps,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_memory_length=max_memory_length,
            max_steps=max_steps,
        )
        # Cross-task state (reused across tasks); per-task state is set in
        # set_env_and_task.
        self._AVAILABLE_ACTION_CLASSES = [Bash, Python, SQL, Terminate]
        self.client = QwenClient()

    def set_env_and_task(self, env: DA_Agent_Env):
        self.env = env
        self.instruction = self.env.task_config['question']
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.usages = []
        self.timings = []
        self.codes = []
        self.history_messages = []
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        self.system_message = SYS_PROMPT_IN_OUR_CODE.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message
                },
            ]
        })
        
    def predict(self, obs: Dict=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """    
        
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        start_time = time.time()
        status = False
        while not status:
            messages = self.history_messages.copy()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })
            try:
                _, response, usage = self.client.generate(
                    messages=messages, 
                    model=self.model,
                    # max_tokens=self.max_tokens,
                    # temperature=self.temperature,
                    # top_p=self.top_p,
                    enable_thinking=True
                )
                status = True
            except Exception as e:
                logging.getLogger("api-llms").error("Failed to call LLM: " + str(e))
                error_info = e.response.json()  
                code_value = error_info['error']['code']
                response = code_value
                status = False
            response = response.strip()
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
            

        try:
            action = self.parse_action(response)
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            if thought:
                thought = thought.group(1).strip()
            else:
                thought = response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
        
        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)

        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)
        self.usages.append(dict(usage))
        end_time = time.time()
        self.timings.append({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time})
        if action is not None:
            self.codes.append(action.code)
        else:
            self.codes.append(None)

        return response, action
    
    
    
    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output: str) -> Action:
        """ Parse action from text """
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
        
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    
    
    def run(self):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        while not done and step_idx < self.max_steps:

            _, action = self.predict(
                obs
            )
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                logger.info("Step %d: %s", step_idx + 1, action)
                if last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, please provide a different action."
                        repeat_action = True
                else:
                    obs, done = self.env.step(action)
                    last_action = action
                    repeat_action = False

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result

    def get_trajectory(self):
        trajectory = []
        for i in range(len(self.observations)):
            trajectory.append({
                "observation": self.observations[i],
                "thought": self.thoughts[i],
                "action": str(self.actions[i]),
                "code": self.codes[i],
                "response": self.responses[i],
                "usage": self.usages[i],
                "timing": self.timings[i]
            })
        trajectory_log = {
            "task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }
        return trajectory_log

