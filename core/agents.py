'''
The four agents. Each is one model call with its own system prompt and generation settings
(max_tokens and effort, defined per agent in `configs.py`).

    analyze  request -> StoryBrief
    plan     brief   -> StoryPlan
    write    outline -> story text
    judge    story   -> Verdict

Agents are stateless functions; the graph in `graph.py` decides when each runs.
'''

from . import configs, prompts
from .llm import call_model, parse_model
from .schemas import StoryBrief, StoryPlan, Verdict


def analyze(request: str) -> StoryBrief:
    '''
    Request Analyzer: turns a free-form request into a structured, child-safe brief.

    Picks the story category, keeps the details the user asked for, and softens anything
    unsuitable for ages 5-10 instead of refusing.
    '''
    return parse_model(prompts.ANALYZER_USER.format(request=request),
                       system=prompts.ANALYZER_SYSTEM,
                       schema=StoryBrief,
                       config=configs.ANALYZER)

def _story_context(request: str, brief: StoryBrief, feedback: list[str]) -> str:
    '''Renders the shared context for planner, storyteller and judge: request, brief and any user feedback.'''
    feedback_block = prompts.FEEDBACK.format(items='\n'.join(f'- {f}' for f in feedback)) if feedback else ''
    return prompts.STORY_CONTEXT.format(request=request,
                                        brief=brief.model_dump_json(indent=2),
                                        feedback=feedback_block)


def plan(request: str, brief: StoryBrief) -> StoryPlan:
    '''Planner: outlines the story as six beats, using the strategy for the brief's category.'''
    prompt = prompts.PLANNER_USER.format(context=_story_context(request, brief, []),
                                              strategy=prompts.CATEGORY_STRATEGIES[brief.category])
    return parse_model(prompt,
                       system=prompts.PLANNER_SYSTEM,
                       schema=StoryPlan,
                       config=configs.PLANNER)


def write(request: str,
          brief: StoryBrief,
          outline: StoryPlan,
          feedback: list[str],
          draft: str | None = None,
          notes: str | None = None) -> str:
    '''Storyteller: writes a new draft, or revises `draft` according to `notes` when both are given.

    Args:
        request: The user's original request.
        brief: The analyzer's brief.
        outline: The planner's beats, followed beat by beat.
        feedback: User change requests so far; these take priority over the brief.
        draft: The story to revise, if this is a revision.
        notes: What the revision must fix (judge notes or a user request).

    Returns:
        The story as plain text: a title line, a blank line, then the story.
    '''
    beats = '\n'.join(f'{i}. {beat}' for i, beat in enumerate(outline.beats, 1))
    task = prompts.STORYTELLER_REVISE.format(notes=notes, draft=draft) if draft and notes else prompts.STORYTELLER_WRITE
    prompt = prompts.STORYTELLER_USER.format(context=_story_context(request, brief, feedback),
                                             strategy=prompts.CATEGORY_STRATEGIES[brief.category],
                                             title=outline.title,
                                             heart=outline.heart,
                                             beats=beats,
                                             task=task)
    return call_model(prompt, system=prompts.STORYTELLER_SYSTEM, config=configs.STORYTELLER)


def judge(request: str, brief: StoryBrief, feedback: list[str], story: str) -> Verdict:
    '''Judge: score a story against the rubric, with evidence and a concrete fix per criterion.

    Safety is a separate pass/fail gate; see `Verdict.passed` for the acceptance rule.
    '''
    # The judge sees what was asked for and the story, never the storyteller's prompt or outline,
    # so it grades the result rather than the intent.
    prompt = prompts.JUDGE_USER.format(context=_story_context(request, brief, feedback),
                                            story=story)
    return parse_model(prompt,
                       system=prompts.JUDGE_SYSTEM,
                       schema=Verdict,
                       config=configs.JUDGE)
