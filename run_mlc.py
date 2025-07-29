from mlc_llm import MLCEngine


def format_prompt_with_template(system_prompt, user_prompt, document, request):
    formatted_user_prompt = user_prompt.replace("{{document}}", document).replace("{{request}}", request)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt}
    ]


system_prompt = """You are an AI assistant for document analysis, performing summarization, question suggestion, and question answering. For each task:

1. Analyze the given document
2. Determine the task (summarization, question suggestion, or question answering)
3. Perform the requested task

Respond using this format:

<intent>: [summarization|question_suggestion|question_answering]
<response>
[Task-specific response here]
</response>

Now, analyze the following document and respond to the request:

"""

user_prompt = """<document>
{{document}}
</document>

<request>
{{request}}
</request>
"""

document = """
Chelsea are world champions! An outstanding performance against Paris Saint-Germain saw the Blues record an emphatic 3-0 victory to win the Club World Cup!
From start to finish at MetLife Stadium, New Jersey, Enzo Maresca’s side was flawless. PSG, who won the Champions League so impressively in May, were simply blown away by the Blues.
We broke the deadlock in the 22nd minute through Cole Palmer; the England international guiding home a shot from the edge of the box after being teed up by Malo Gusto.
Our new No.10 then repeated the trick on the half-hour mark. The finish was a carbon copy of his first, as he breezed into the penalty area and again found the far corner.

It got even better for the Blues before half time as, with two minutes to play, Palmer slipped the ball through to Joao Pedro, who coolly lifted the ball beyond Gianluigi Donnarumma.
In the second period, Robert Sanchez displayed his quality with two brilliant saves to keep Paris Saint-Germain at bay. We also had further chances to add to our lead – Donnarumma produced two excellent stops to deny substitute Liam Delap.
The damage had well and truly been done in the opening 45 minutes, though. There was little PSG could muster in response, and anything that was created was repelled by the Blues. And at full-time, there was a release of emotion from everyone wearing blue inside MetLife Stadium.
"""

request = "Summarize the given document."
# request = "Which team was mentioned in the document?"
# request = "Suggest 3 questions about the document to help me understand it better."

# Create engine
model = "models/SlimLM-125M-DocAssist-q0f16-MLC/"
model_lib = f"models/SlimLM-125M-DocAssist-q0f16-MLC/model_lib-metal.so"

engine = MLCEngine(model=model, model_lib=model_lib)

# Use the new function to format prompts based on model type
messages = format_prompt_with_template(model, system_prompt, user_prompt, document, request)

print("\nMESSAGES:\n")
print(messages)


print("\nRESPONSE:\n")

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=messages,
    model=model,
    stream=True,
    max_tokens=256,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)

print("\n")

# Uncomment the code below for non-streaming mode
# response = engine.chat.completions.create( 
#     # messages=[{"role": "user", "content": "What is the meaning of life?"}],
#     messages=messages,
#     model=model,
#     stream=False,
#     max_tokens=256,
# )
# print("\nRESPONSE:\n")
# print(response)

engine.terminate()

