# Setup, Running and Debugging

How to install, run and debug the bedtime storyteller.

## Requirements

- Python 3.10 or newer (developed on Python 3.14)
- An Anthropic API key

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure the API key

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_WORKSPACE_ID=
```

- `ANTHROPIC_API_KEY` is required.
- `ANTHROPIC_WORKSPACE_ID` is only needed for **user-level keys** (prefix `sk-ant-usr`), which can act on several workspaces. Find the ID (`wrkspc_...`) in the Claude Console under *Settings → Workspaces*. Leave it empty for a standard workspace key (prefix `sk-ant-api`). If you do set it with a standard key, it must match that key's workspace.

`.env` is git-ignored. Never commit it.

## Run

```bash
python main.py
```

1. Type a story request, for example `A story about a shy dragon who wants to make friends`. Press Enter on an empty line to use the built-in example request.
2. Progress appears while the agents work: the request category, then a score line for each judge round.
3. The finished story is printed. Feedback can be provided to make changes (for example `make it funnier` or `add a sleepy owl`) and the story is revised and judged again. Press Enter on an empty line to finish.

Each story takes 4 to 8 model calls: one each for the analyzer and planner, then one write and one judge call per round, up to `MAX_ROUNDS` rounds.

### Command-line options

| Flag | Effect |
| --- | --- |
| `-v`, `--verbose` | Also print the story outline and the judge's revision notes for each round |
| `--graph` | Print the agent graph as a Mermaid diagram and exit (no API key needed) |
| `-h`, `--help` | Show usage help |

Progress goes to **stderr** and the story to **stdout**. To hide progress, run `python main.py 2>/dev/null`. To view the graph diagram, paste the output of `python main.py --graph` into [mermaid.live](https://mermaid.live). A rendered copy is in [flow-graph.png](flow-graph.png).

## Project layout

```text
main.py            CLI: reads the request, runs the graph, loops on user feedback
core/
  configs.py       Model, per-agent max_tokens/effort, judge-loop thresholds
  graph.py         LangGraph flow: analyze → plan → write ⇄ judge → present (feedback) → write …
  agents.py        The four agents; each is one model call
  prompts.py       System prompts, user-message templates, per-category strategies
  schemas.py       Pydantic structured outputs (StoryBrief, StoryPlan, Verdict) and the pass rule
  llm.py           Claude API wrapper: client, structured outputs, stop-reason checks
```

## Tuning

All settings are in [core/configs.py](core/configs.py):

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `claude-haiku-5-5` | Model used by every agent |
| `ANALYZER`, `PLANNER`, `STORYTELLER`, `JUDGE` | see file | Per-agent `max_tokens` (includes the model's reasoning) and `effort` (`low`/`medium`/`high`) |
| `MAX_ROUNDS` | `3` | Judge rounds per story or per change request |
| `PASS_SCORE` | `8` | Every rubric criterion must reach this, and safety must pass, for a draft to be accepted |
| `POLISH_SCORE` | `9` | Criteria scoring below this get a concrete fix from the judge, even if they pass |

`temperature` is not configurable: Claude Haiku 5.5 rejects non-default values. `effort` is used to tune each role instead.

Prompt wording, including the target length of 500–700 words and the 5–10 age range, is in [core/prompts.py](core/prompts.py).

## Debugging

### See what the agents decided

Run with `-v` to print the outline and every revision note the judge sends to the storyteller:

```text
Writing...
  judge round 1: safe | request_fit 8, story_arc 7, engagement 7, language_level 8, bedtime_tone 9 | avg 7.8
  revision notes:
- [story_arc, 7/10] Show why the dragon's second try fails before the owl helps...
  revising...
```

If stories keep hitting the round limit, the notes show which criterion is holding them back.

### Common errors

| Message | Cause and fix |
| --- | --- |
| `This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header` | You are using a user-level key (`sk-ant-usr...`). Set `ANTHROPIC_WORKSPACE_ID` in `.env`, or use a workspace key (`sk-ant-api...`). With a user-level key this error can appear even if the key itself is invalid. |
| `ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.` | No key was loaded, or it is empty. Check that `.env` exists in the repository root and sets `ANTHROPIC_API_KEY`. The app checks this at startup, before asking for a story. A variable already set in your shell takes precedence over `.env`, so an empty `ANTHROPIC_API_KEY` exported in the shell also triggers this. |
| `API error: Error code: 401 ...` | The API key is wrong or revoked. |
| `Sorry, I couldn't finish the story: The response was cut off (max_tokens reached).` | An agent ran out of tokens, usually because its reasoning was long. Raise that agent's `max_tokens` in `core/configs.py`. |
| `Sorry, I couldn't finish the story: The model declined this request.` | A safety classifier declined the request. Rephrase it. |
| `400 ... temperature` | Something is passing a sampling parameter. Claude Haiku 5.5 only accepts the defaults. |
| `ModuleNotFoundError: No module named 'core'` | `core` module was not found. Run commands from the repository root, with the virtualenv activated. |
