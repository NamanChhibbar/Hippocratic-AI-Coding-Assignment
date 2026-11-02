'''
Structured outputs exchanged between the agents, and the judge's acceptance rule.

The agent models (StoryBrief, StoryPlan, Verdict) are passed to the API as structured-output
schemas, so their class docstrings and Field descriptions are sent to the model as instructions.
'''

from typing import Literal

from pydantic import BaseModel, Field

from .configs import PASS_SCORE, POLISH_SCORE

# The story categories the analyzer chooses from; each maps to a strategy in `prompts.CATEGORY_STRATEGIES`.
Category = Literal['calm_bedtime', 'adventure', 'funny', 'friendship_feelings', 'curious_learning']


class StoryBrief(BaseModel):
    '''Output schema for Request Analyzer: agent extracts these fields from the user's request, softening anything unsuitable for ages 5-10 with safety notes.'''

    category: Category
    characters: list[str] = Field(description='Named characters with one short descriptor each, e.g. "Bob, a curious cat"')
    setting: str
    tone: str = Field(description='Two or three words, e.g. "gentle and silly"')
    must_include: list[str] = Field(description='Specific things the user asked for that the story must contain')
    safety_notes: str = Field(description='What was softened to make the request suitable for ages 5-10, or "" if nothing')
    rewritten_request: str = Field(description='One or two sentences restating the request as a child-safe story idea')


class StoryPlan(BaseModel):
    '''Output schema for Story Planner: a six-beat outline with a title and the story's heart.'''

    title: str
    heart: str = Field(description='The gentle idea the story leaves the child with, in one sentence')
    beats: list[str] = Field(description='Six beats in order, one or two sentences each')


class CriterionScore(BaseModel):
    '''The judge's score for one rubric criterion.'''

    score: int = Field(description='1-10, using the rubric anchors')
    evidence: str = Field(description='A short quote or specific reference from the story that justifies the score')
    fix: str = Field(description=f'One concrete instruction to improve this criterion, or "" if the score is {POLISH_SCORE} or higher')


class Verdict(BaseModel):
    '''Output schema for Story Judge: whether the draft is safe and meets the rubric criteria, with notes.'''

    safety_pass: bool = Field(description='False if anything is frightening, violent, unkind, or models unsafe behaviour')
    safety_issue: str = Field(description='What must change for safety, or "" if safety_pass is true')
    request_fit: CriterionScore
    story_arc: CriterionScore
    engagement: CriterionScore
    language_level: CriterionScore
    bedtime_tone: CriterionScore

    def scores(self) -> dict[str, CriterionScore]:
        '''The five rubric criteria by name, in rubric order.'''
        return {
            'request_fit': self.request_fit,
            'story_arc': self.story_arc,
            'engagement': self.engagement,
            'language_level': self.language_level,
            'bedtime_tone': self.bedtime_tone,
        }

    @property
    def average(self) -> float:
        '''Mean score across the rubric criteria; used to rank drafts that did not pass.'''
        scores = self.scores()
        return sum(c.score for c in scores.values()) / len(scores)

    @property
    def passed(self) -> bool:
        '''Accepted when the draft is safe and every criterion reaches PASS_SCORE.'''
        return self.safety_pass and all(c.score >= PASS_SCORE for c in self.scores().values())

    def revision_notes(self) -> str:
        '''
        Format the judge's fixes as a bullet list for the storyteller.

        Safety comes first, then every criterion scoring below POLISH_SCORE, weakest first, so the storyteller
        also polishes criteria that already pass.
        '''
        notes = []
        if not self.safety_pass:
            notes.append(f'- [SAFETY, must fix] {self.safety_issue}')
        weak = sorted((c.score, name, c) for name, c in self.scores().items() if c.score < POLISH_SCORE and c.fix)
        notes += [f'- [{name}, {score}/10] {c.fix}' for score, name, c in weak]
        return '\n'.join(notes)


class Draft(BaseModel):
    '''A story draft together with the judge's verdict on it.'''

    story: str
    verdict: Verdict


def best_draft(drafts: list[Draft]) -> Draft:
    '''The latest passing draft, otherwise the highest scoring one (safe drafts first).'''
    passing = [d for d in drafts if d.verdict.passed]
    if passing:
        return passing[-1]
    return max(drafts, key=lambda d: (d.verdict.safety_pass, d.verdict.average))
