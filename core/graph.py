'''
LangGraph flow: analyzer -> planner -> storyteller <-> judge -> user feedback.

    START -> analyze -> plan -> write -> judge --(passed or round limit)--> present --(no feedback)--> END
                                  |        |                                  |
                                  +----<---+ (revise with judge notes)        |
                                  |                                           |
                                  +-----------------------<-------------------+ (revise with user feedback)

Each node is a thin adapter: it reads what it needs from `StoryState`, calls one agent, logs
progress, and returns only the state keys it changed. `present` pauses the graph with `interrupt`
so the caller can show the story and collect feedback;
'''

import logging
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import agents
from .configs import MAX_ROUNDS
from .schemas import Draft, StoryBrief, StoryPlan, best_draft

log = logging.getLogger(__name__)


class StoryState(TypedDict, total=False):
    '''Everything the graph carries between nodes. Keys fill in as the run progresses.'''

    request: str  # The user's original request
    brief: StoryBrief  # Set by Analyzer
    outline: StoryPlan  # Set by Planner
    feedback: list[str]  # User change requests, oldest first
    drafts: list[Draft]  # Judged drafts since the last user feedback
    story: str  # Newest draft, not yet judged
    revise_from: str  # Draft for the storyteller to revise; absent for a first draft
    notes: str  # What the revision must fix


def analyze(state: StoryState) -> StoryState:
    '''Builds a story brief for Planner from the user's request.'''
    log.info('Reading your request...')
    brief = agents.analyze(state['request'])
    log.info('  category: %s', brief.category)
    if brief.safety_notes:
        log.info('  made gentler: %s', brief.safety_notes)
    return {'brief': brief}


def plan(state: StoryState) -> StoryState:
    '''Plans an outline of the story from the brief.'''
    log.info('Planning the story...')
    outline = agents.plan(state['request'], state['brief'])
    log.debug('  outline:\n%s', '\n'.join(f'  {i}. {b}' for i, b in enumerate(outline.beats, 1)))
    return {'outline': outline}


def write(state: StoryState) -> StoryState:
    '''Writes the first draft, or a revision when `revise_from` and `notes` are set.'''
    log.info('  revising...' if state.get('revise_from') else 'Writing...')
    story = agents.write(state['request'],
                         state['brief'],
                         state['outline'],
                         state['feedback'],
                         draft=state.get('revise_from'),
                         notes=state.get('notes'))
    return {'story': story}


def judge(state: StoryState) -> StoryState:
    '''
    Scores the newest story draft and prepares the revision, if more rounds remain.

    Every judged draft is kept in `drafts`. When the draft fails and rounds remain, the next
    revision starts from the best draft so far with that draft's notes.
    '''
    verdict = agents.judge(state['request'], state['brief'], state['feedback'], state['story'])
    drafts = state.get('drafts', []) + [Draft(story=state['story'], verdict=verdict)]
    # Logging
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
    # revise from the best draft so far
    best = best_draft(drafts)
    return {'drafts': drafts, 'revise_from': best.story, 'notes': best.verdict.revision_notes()}


def route_after_judge(state: StoryState) -> Literal['write', 'present']:
    '''Go to `present` once the latest draft passes or the round limit is hit; otherwise revise.'''
    drafts = state['drafts']
    if drafts[-1].verdict.passed or len(drafts) >= MAX_ROUNDS:
        return 'present'
    return 'write'


def present(state: StoryState) -> Command[Literal['write', '__end__']]:
    '''
    Show the best draft and wait for the user's feedback; an empty reply ends the session.

    Feedback starts a new generation: it is added to `feedback` (so the storyteller and judge both
    see it from now on), the judged drafts are cleared, and the shown story is revised to apply it.
    '''
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
            'notes': f'- [user request, must apply] {feedback}',
        },
    )


def build_graph():
    '''Assemble and compile the story graph.

    Invoke it with `{'request': ..., 'feedback': []}` and a `thread_id` in the config. It returns
    with `__interrupt__` set whenever a story is ready; resume with `Command(resume=<feedback>)`.
    '''
    # Nodes
    builder = StateGraph(StoryState)
    builder.add_node('analyze', analyze)
    builder.add_node('plan', plan)
    builder.add_node('write', write)
    builder.add_node('judge', judge)
    builder.add_node('present', present)
    # Edges
    builder.add_edge(START, 'analyze')
    builder.add_edge('analyze', 'plan')
    builder.add_edge('plan', 'write')
    builder.add_edge('write', 'judge')
    builder.add_conditional_edges('judge', route_after_judge)
    # present -> write | END is declared by its Command return type

    # Saves custom models in memory during interrupts used in `present` function
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[(m.__module__, m.__name__)
                                 for m in (StoryBrief, StoryPlan, Draft)]
    )
    return builder.compile(checkpointer=InMemorySaver(serde=serde))
