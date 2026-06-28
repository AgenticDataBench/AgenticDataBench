"""
Base Agent Module - Defines the unified PromptAgent interface shared by all
agent implementations under testbed/da_agent/agent/.

All agents are constructed by run.py ONCE with the same keyword arguments
(model, max_tokens, top_p, temperature, max_memory_length, max_steps), then
set_env_and_task(env) is called per task before run(). This base class owns
that constructor and the cross-task state, so subclasses never re-declare the
parameter list. Per-task state belongs solely in set_env_and_task (no
duplication with __init__). Subclasses must implement set_env_and_task, run,
and get_trajectory.
"""

from da_agent.envs.da_agent import DA_Agent_Env


class BaseAgent:
    """Common constructor config and interface contract for all agents.

    LLM/generation knobs that a given backend does not need are still accepted
    (and stored) so every PromptAgent can be built with the same call site.
    __init__ holds only cross-task state; per-task state is (re)initialized in
    set_env_and_task, which runs before every run().
    """

    def __init__(
        self,
        model,
        max_tokens,
        top_p,
        temperature,
        max_memory_length,
        max_steps,
    ):
        # LLM / generation config (some agents use only a subset).
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps

        # Cross-task runtime state (set_env_and_task overwrites per task).
        self.env: DA_Agent_Env | None = None
        self.instruction = ""
        self.trajectory = []
        self.work_dir = "/workspace"

    def set_env_and_task(self, env: DA_Agent_Env):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def get_trajectory(self):
        raise NotImplementedError