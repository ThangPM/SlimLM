import torch 
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, pipeline, MptForCausalLM


torch.random.manual_seed(0) 


PROMPT_TEMPLATES = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "system": "<|system|>{system_prompt}<|end|>",
        "user": "<|user|>{user_prompt}<|end|>",
        "assistant": "<|assistant|>",
    },
    "HuggingFaceTB/SmolLM-135M-Instruct": {
        "system": "<|im_start|>system\n{system_prompt}<|im_end|>",
        "user": "<|im_start|>user\n{user_prompt}<|im_end|>",
        "assistant": "<|im_start|>assistant\n",
    },
    "default": {
        "system": "{system_prompt}",
        "user": "{user_prompt}",
        "assistant": "",
    },
    # Add more model-specific templates here
}


def get_prompt_template(model_path):
    return PROMPT_TEMPLATES.get(model_path, PROMPT_TEMPLATES["default"])


def format_prompt(model_path, system_prompt, user_prompt):
    template = get_prompt_template(model_path)
    formatted_prompt = (
        template["system"].format(system_prompt=system_prompt) + " " +
        template["user"].format(user_prompt=user_prompt) + " " +
        template["assistant"]
    )
    return formatted_prompt.strip()


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


def load_model_and_tokenizer(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_device = 'cpu'

    model = AutoModelForCausalLM.from_pretrained( 
        model_path,  
        config=config,
        device_map="cpu",  
        torch_dtype="auto",  
        trust_remote_code=True,  
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


if __name__ == "__main__":
    model_path = "models/SlimLM-125M-DocAssist-HF"

    model, tokenizer = load_model_and_tokenizer(model_path)

    formatted_user_prompt = user_prompt.replace("{{document}}", document).replace("{{request}}", request)
    formatted_prompt = format_prompt(model_path, system_prompt, formatted_user_prompt)

    print(f"\033[92m")
    print(formatted_prompt)
    print(f"\033[0m")

    prompt_tokenized=tokenizer(formatted_prompt, return_tensors="pt")

    output_tokenized = model.generate(
        input_ids=prompt_tokenized["input_ids"], 
        max_new_tokens=256,    
        output_scores=True,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.1,
        )

    output = tokenizer.decode(output_tokenized[0][prompt_tokenized["input_ids"].shape[1]:]).strip().replace(tokenizer.eos_token, "")

    print(f"\033[93m")
    print("RESPONSE:\n")
    print(output)
    print(f"\033[0m")


