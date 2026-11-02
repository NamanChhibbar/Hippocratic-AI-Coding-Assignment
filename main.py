'''
Command-line entry point: ask for a story request, tell the story, then take change requests.

Run `python main.py` (add `-v` for the outline and judge notes, `--graph` for the flow diagram).
Progress is logged to stderr; the story itself is printed to stdout.
'''

import argparse
import logging
import sys

import anthropic
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from bedtime.graph import build_graph
from bedtime.llm import ModelError

'''
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

'''

# Used when the user just presses Enter at the first prompt
example_requests = 'A story about a girl named Alice and her best friend Bob, who happens to be a cat.'


def main():
    '''Parse arguments, run the story graph, and loop on user feedback until an empty reply.'''
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Bedtime stories for ages 5-10, refined by an LLM judge.')
    parser.add_argument('-v', '--verbose', action='store_true', help="show the outline and the judge's revision notes")
    parser.add_argument('--graph', action='store_true', help='print the agent graph as a Mermaid diagram and exit')
    args = parser.parse_args()
    # Build the flow graph
    graph = build_graph()
    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return
    # Progress goes to stderr so stdout holds only the story
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(message)s', stream=sys.stderr)
    for noisy in ('anthropic', 'httpx', 'httpx2', 'httpcore', 'httpcore2'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()  # Load .env
    try:
        user_input = input('What kind of story do you want to hear? ').strip() or example_requests
        # `config` passes the thread_id to the graph to restore state across interrupts
        config = RunnableConfig({'configurable': {'thread_id': 'session0'}})
        state = graph.invoke({'request': user_input, 'feedback': []}, config)
        # The graph pauses in `present` with the story; resume it with the user's feedback
        while interrupts := state.get('__interrupt__'):
            print(f'\n{interrupts[0].value}\n')
            feedback = input('Would you like any changes? (press Enter to finish) ')
            state = graph.invoke(Command(resume=feedback), config)
    except (KeyboardInterrupt, EOFError):
        print()
    except ModelError as e:
        sys.exit(f"Sorry, I couldn't finish the story: {e}")
    except anthropic.AnthropicError as e:
        sys.exit(f'API error: {e}\nCheck that ANTHROPIC_API_KEY is set in .env.')
    # Ending note
    print('Sweet dreams!')


if __name__ == '__main__':
    main()
