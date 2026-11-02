'''
Wrapper around the Claude API shared by every agent.

All model access goes through `call_model` (free text) and `parse_model` (structured output).
'''

import os
from functools import cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from .configs import MODEL, AgentConfig

T = TypeVar('T', bound=BaseModel)


class ModelError(RuntimeError):
    '''Raised if the model declined the request, ran out of tokens, or returned unusable output.'''


@cache
def _client() -> anthropic.Anthropic:
    '''Create the API client once.'''
    # User-level keys (sk-ant-usr...) can act on several workspaces, so each request must name one.
    # Workspace-scoped keys (sk-ant-api...) don't need it; if sent, it must match the key's workspace.
    workspace_id = os.environ.get('ANTHROPIC_WORKSPACE_ID')
    headers = {'anthropic-workspace-id': workspace_id} if workspace_id else None
    return anthropic.Anthropic(default_headers=headers)  # reads ANTHROPIC_API_KEY


def _check(resp) -> None:
    '''
    Raise ModelError if the response did not finish normally.

    A refusal comes back as HTTP 200 with `stop_reason == 'refusal'`, and a truncated reply as
    `stop_reason == 'max_tokens'`; neither raises in the SDK, so both are checked explicitly.
    '''
    if resp.stop_reason == 'refusal':
        raise ModelError('The model declined this request.')
    if resp.stop_reason == 'max_tokens':
        raise ModelError('The response was cut off (max_tokens reached).')


def call_model(prompt: str, system: str, config: AgentConfig) -> str:
    '''
    Send a single-turn request and return the model's text reply.

    Args:
        prompt: The user message.
        system: The system prompt that sets the agent's role.
        config: The agent's max_tokens and effort, from `configs`.

    Raises:
        ModelError: If the model refused or the reply was truncated.
    '''
    resp = _client().messages.create(
        model=MODEL,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=config.max_tokens,
        output_config={'effort': config.effort},
    )
    _check(resp)
    # the response may also contain thinking blocks; return only the text
    return ''.join(block.text for block in resp.content if block.type == 'text').strip()


def parse_model(prompt: str, system: str, schema: type[T], config: AgentConfig) -> T:
    '''
    Like call_model, but the reply is constrained to `schema` and returned validated.

    Uses structured outputs, so the API guarantees JSON matching the schema's shape; the SDK then
    validates it into a `schema` instance. Field descriptions in the schema act as instructions.

    Raises:
        ModelError: If the model refused, the reply was truncated, or nothing could be parsed.
    '''
    resp = _client().messages.parse(
        model=MODEL,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=config.max_tokens,
        output_config={'effort': config.effort},
        output_format=schema,
    )
    # check the stop reason first: a refusal or truncation also leaves parsed_output empty
    _check(resp)
    if resp.parsed_output is None:
        raise ModelError('The model did not return a valid response.')
    return resp.parsed_output
