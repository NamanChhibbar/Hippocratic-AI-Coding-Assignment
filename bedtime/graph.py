'''LangGraph flow: analyzer -> planner -> storyteller <-> judge -> listener feedback.'''

import logging
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import agents
from .schemas import Draft, StoryBrief, StoryPlan, best_draft

log = logging.getLogger(__name__)

MAX_ROUNDS = 3  # judge rounds per generation; bounds cost and latency


class StoryState(TypedDict, total=False):
    request: str
    brief: StoryBrief
    outline: StoryPlan
    feedback: list[str]  # listener change requests, oldest first
    drafts: list[Draft]  # judged drafts since the last listener feedback
    story: str  # newest draft, not yet judged
    revise_from: str  # draft for the storyteller to revise; absent for a first draft
    notes: str  # what the revision must fix


def analyze(state: StoryState) -> StoryState:
    log.info('Reading your request...')
    brief = agents.analyze(state['request'])
    log.info('  category: %s', brief.category)
    if brief.safety_notes:
        log.info('  made gentler: %s', brief.safety_notes)
    return {'brief': brief}


def plan(state: StoryState) -> StoryState:
    log.info('Planning the story...')
    outline = agents.plan(state['request'], state['brief'])
    log.debug('  outline:\n%s', '\n'.join(f'  {i}. {b}' for i, b in enumerate(outline.beats, 1)))
    return {'outline': outline}


def write(state: StoryState) -> StoryState:
    log.info('  revising...' if state.get('revise_from') else 'Writing...')
    story = agents.write(
        state['request'],
        state['brief'],
        state['outline'],
        state['feedback'],
        draft=state.get('revise_from'),
        notes=state.get('notes'),
    )
    return {'story': story}


def judge(state: StoryState) -> StoryState:
    verdict = agents.judge(state['request'], state['brief'], state['feedback'], state['story'])
    drafts = state.get('drafts', []) + [Draft(story=state['story'], verdict=verdict)]

    scores = ', '.join(f'{name} {c.score}' for name, c in verdict.scores().items())
    safety = 'safe' if verdict.safety_pass else 'UNSAFE'
    log.info('  judge round %d: %s | %s | avg %.1f', len(drafts), safety, scores, verdict.average)
    if verdict.revision_notes():
        log.debug('  revision notes:\n%s', verdict.revision_notes())

    if verdict.passed:
        log.info('  accepted.')
        return {'drafts': drafts}
    if len(drafts) >= MAX_ROUNDS:
        log.info('  round limit reached; keeping the best draft.')
        return {'drafts': drafts}
    # revise from the best draft so far, so a revision that made things worse is discarded
    best = best_draft(drafts)
    return {'drafts': drafts, 'revise_from': best.story, 'notes': best.verdict.revision_notes()}


def route_after_judge(state: StoryState) -> Literal['write', 'present']:
    drafts = state['drafts']
    if drafts[-1].verdict.passed or len(drafts) >= MAX_ROUNDS:
        return 'present'
    return 'write'


def present(state: StoryState) -> Command[Literal['write', '__end__']]:
    '''Show the best draft and wait for the listener's feedback; an empty reply ends the session.'''
    best = best_draft(state['drafts'])
    # Pauses the graph and hands the story to the caller; resumes with Command(resume=<feedback>).
    # The node re-runs from the top on resume, so nothing before this line may have side effects.
    feedback = (interrupt(best.story) or '').strip()
    if not feedback:
        return Command(goto=END)
    log.info('Rewriting with your changes...')
    return Command(
        goto='write',
        update={
            'feedback': state['feedback'] + [feedback],
            'drafts': [],
            'revise_from': best.story,
            'notes': f'- [listener request, must apply] {feedback}',
        },
    )


def build_graph():
    builder = StateGraph(StoryState)
    builder.add_node('analyze', analyze)
    builder.add_node('plan', plan)
    builder.add_node('write', write)
    builder.add_node('judge', judge)
    builder.add_node('present', present)

    builder.add_edge(START, 'analyze')
    builder.add_edge('analyze', 'plan')
    builder.add_edge('plan', 'write')
    builder.add_edge('write', 'judge')
    builder.add_conditional_edges('judge', route_after_judge)
    # present -> write | END is declared by its Command return type

    # the checkpointer keeps state across the interrupt in `present`; our state types must be allow-listed
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(m.__module__, m.__name__) for m in (StoryBrief, StoryPlan, Draft)])
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
