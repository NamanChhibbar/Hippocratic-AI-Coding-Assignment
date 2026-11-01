'''Thin wrapper around the Claude API shared by every agent.'''

from functools import cache
from typing import Literal, TypeVar

import anthropic
from pydantic import BaseModel

MODEL = 'claude-haiku-5-5'

# Haiku 5.5 rejects non-default temperature, so each role is tuned through
# `effort` (how much the model reasons before answering) instead.
Effort = Literal['low', 'medium', 'high']

T = TypeVar('T', bound=BaseModel)


class ModelError(RuntimeError):
    '''The model declined the request or ran out of tokens.'''


@cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def _check(resp) -> None:
    if resp.stop_reason == 'refusal':
        raise ModelError('The model declined this request.')
    if resp.stop_reason == 'max_tokens':
        raise ModelError('The response was cut off (max_tokens reached).')


def call_model(prompt: str, system: str, max_tokens: int = 3000, effort: Effort = 'medium') -> str:
    resp = _client().messages.create(
        model=MODEL,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        output_config={'effort': effort},
    )
    _check(resp)
    # the response may also contain thinking blocks; return only the text
    return ''.join(block.text for block in resp.content if block.type == 'text').strip()


def parse_model(prompt: str, system: str, schema: type[T], max_tokens: int = 3000, effort: Effort = 'medium') -> T:
    '''Like call_model, but the reply is constrained to `schema` and returned validated.'''
    resp = _client().messages.parse(
        model=MODEL,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        output_config={'effort': effort},
        output_format=schema,
    )
    _check(resp)
    if resp.parsed_output is None:
        raise ModelError('The model did not return a valid response.')
    return resp.parsed_output
