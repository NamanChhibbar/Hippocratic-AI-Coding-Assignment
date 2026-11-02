'''Model and pipeline configurations for the story pipeline.'''

from dataclasses import dataclass
from typing import Literal

MODEL = 'claude-haiku-5-5'

# Haiku 5.5 rejects non-default temperature, so each role is tuned through
# `effort` (how much the model reasons before answering) instead.
Effort = Literal['low', 'medium', 'high']


@dataclass(frozen=True)
class AgentConfig:
    '''Generation settings for one agent.'''

    max_tokens: int  # upper bound on output, including the model's reasoning
    effort: Effort


ANALYZER = AgentConfig(max_tokens=2000, effort='low')  # extraction and classification
PLANNER = AgentConfig(max_tokens=3000, effort='medium')  # six-beat outline
STORYTELLER = AgentConfig(max_tokens=8000, effort='medium')  # 500-700 word story plus reasoning
JUDGE = AgentConfig(max_tokens=8000, effort='high')  # careful, rubric-based review

# Judge loop
MAX_ROUNDS = 3  # judge rounds per generation; bounds cost and latency
PASS_SCORE = 8  # every criterion must reach this for the judge to accept a draft
POLISH_SCORE = 9  # criteria scoring below this get a concrete fix from the judge, even when they pass
