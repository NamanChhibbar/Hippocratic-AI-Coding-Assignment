'''Structured outputs exchanged between the agents.'''

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal['calm_bedtime', 'adventure', 'funny', 'friendship_feelings', 'curious_learning']

PASS_SCORE = 8  # every criterion must reach this for the judge to accept a draft


class StoryBrief(BaseModel):
    '''What the Request Analyzer extracts from the user's request.'''

    category: Category
    characters: list[str] = Field(description='Named characters with one short descriptor each, e.g. "Bob, a curious cat"')
    setting: str
    tone: str = Field(description='Two or three words, e.g. "gentle and silly"')
    must_include: list[str] = Field(description='Specific things the user asked for that the story must contain')
    safety_notes: str = Field(description='What was softened to make the request suitable for ages 5-10, or "" if nothing')
    rewritten_request: str = Field(description='One or two sentences restating the request as a child-safe story idea')


class StoryPlan(BaseModel):
    '''The Planner's beat outline.'''

    title: str
    heart: str = Field(description='The gentle idea the story leaves the child with, in one sentence')
    beats: list[str] = Field(description='Six beats in order, one or two sentences each')


class CriterionScore(BaseModel):
    score: int = Field(description='1-10, using the rubric anchors')
    evidence: str = Field(description='A short quote or specific reference from the story that justifies the score')
    fix: str = Field(description='One concrete instruction to improve this criterion, or "" if the score is 9 or 10')


class Verdict(BaseModel):
    '''The Judge's assessment of one draft.'''

    safety_pass: bool = Field(description='False if anything is frightening, violent, unkind, or models unsafe behaviour')
    safety_issue: str = Field(description='What must change for safety, or "" if safety_pass is true')
    request_fit: CriterionScore
    story_arc: CriterionScore
    engagement: CriterionScore
    language_level: CriterionScore
    bedtime_tone: CriterionScore

    def scores(self) -> dict[str, CriterionScore]:
        return {
            'request_fit': self.request_fit,
            'story_arc': self.story_arc,
            'engagement': self.engagement,
            'language_level': self.language_level,
            'bedtime_tone': self.bedtime_tone,
        }

    @property
    def average(self) -> float:
        scores = self.scores()
        return sum(c.score for c in scores.values()) / len(scores)

    @property
    def passed(self) -> bool:
        return self.safety_pass and all(c.score >= PASS_SCORE for c in self.scores().values())

    def revision_notes(self) -> str:
        '''Fixes for every criterion short of excellent, weakest first.'''
        notes = []
        if not self.safety_pass:
            notes.append(f'- [SAFETY, must fix] {self.safety_issue}')
        weak = sorted((c.score, name, c) for name, c in self.scores().items() if c.score < 9 and c.fix)
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
