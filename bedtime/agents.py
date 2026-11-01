'''The four agents. Each is one model call with its own system prompt and effort level.'''

from . import prompts
from .llm import call_model, parse_model
from .schemas import StoryBrief, StoryPlan, Verdict


def _story_context(request: str, brief: StoryBrief, feedback: list[str]) -> str:
    feedback_block = prompts.FEEDBACK.format(items='\n'.join(f'- {f}' for f in feedback)) if feedback else ''
    return prompts.STORY_CONTEXT.format(request=request, brief=brief.model_dump_json(indent=2), feedback=feedback_block)


def analyze(request: str) -> StoryBrief:
    return parse_model(
        prompts.ANALYZER_USER.format(request=request),
        system=prompts.ANALYZER_SYSTEM,
        schema=StoryBrief,
        max_tokens=2000,
        effort='low',
    )


def plan(request: str, brief: StoryBrief) -> StoryPlan:
    return parse_model(
        prompts.PLANNER_USER.format(
            context=_story_context(request, brief, []),
            strategy=prompts.CATEGORY_STRATEGIES[brief.category],
        ),
        system=prompts.PLANNER_SYSTEM,
        schema=StoryPlan,
        max_tokens=3000,
        effort='medium',
    )


def write(
    request: str,
    brief: StoryBrief,
    outline: StoryPlan,
    feedback: list[str],
    draft: str | None = None,
    notes: str | None = None,
) -> str:
    '''Write a first draft, or revise `draft` according to `notes` when both are given.'''
    task = prompts.STORYTELLER_REVISE.format(notes=notes, draft=draft) if draft and notes else prompts.STORYTELLER_WRITE
    prompt = prompts.STORYTELLER_USER.format(
        context=_story_context(request, brief, feedback),
        strategy=prompts.CATEGORY_STRATEGIES[brief.category],
        title=outline.title,
        heart=outline.heart,
        beats='\n'.join(f'{i}. {beat}' for i, beat in enumerate(outline.beats, 1)),
        task=task,
    )
    return call_model(prompt, system=prompts.STORYTELLER_SYSTEM, max_tokens=8000, effort='medium')


def judge(request: str, brief: StoryBrief, feedback: list[str], story: str) -> Verdict:
    # The judge sees what was asked for and the story, never the storyteller's prompt or outline,
    # so it grades the result rather than the intent.
    return parse_model(
        prompts.JUDGE_USER.format(context=_story_context(request, brief, feedback), story=story),
        system=prompts.JUDGE_SYSTEM,
        schema=Verdict,
        max_tokens=8000,
        effort='high',
    )
