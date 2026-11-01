import anthropic
from dotenv import load_dotenv

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()


def call_model(prompt: str, max_tokens=3000) -> str:
    # temperature is not configurable on claude-haiku-5-5 (non-default values return a 400)
    resp = client.messages.create(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    # the response may also contain thinking blocks; return only the text
    return "".join(block.text for block in resp.content if block.type == "text")

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."


def main():
    user_input = input("What kind of story do you want to hear? ")
    response = call_model(user_input)
    print(response)


if __name__ == "__main__":
    main()
