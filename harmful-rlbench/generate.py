from openai import OpenAI
import json
import re
import os

# OpenAI API
client = OpenAI(
    api_key="YOUR_API_KEY"
)

def iterative_prompting(initial_prompt, examples_prompt, llm_model, max_iterations=10):
    """
    :param initial_prompt: str, the initial prompt
    :param examples_prompt: str, the examples prompt
    :param max_iterations: int, the maximum number of iterations
    :return: None
    """
    
    for i in range(max_iterations):
        print(f"--- Iteration {i+1} ---")
        # build user prompt
        user_prompt = initial_prompt + examples_prompt

        # build conversation context
        conversation = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # call OpenAI API to generate model output
        response = client.chat.completions.create(
            model=llm_model,
            messages=conversation,
        )

        try:
            # get model output text
            model_reply = response.choices[0].message.content.strip()
            
            print(f"--- Iteration {i+1} Output ---")
            print(model_reply)
            
            # filter out the content between ```json and ```
            model_reply = re.search(r"```json(.*)```", model_reply, re.DOTALL).group(1)
            # concatenate the model output and the current examples_prompt
            examples_prompt = examples_prompt.strip("```json").strip("```")
            # convert to json
            model_reply_json = json.loads(model_reply)
            examples_prompt_json = json.loads(examples_prompt)
            # add the content of model_reply_json to examples_prompt_json
            examples_prompt_json.extend(model_reply_json)

            # convert the merged content to a formatted JSON string
            examples_prompt = json.dumps(examples_prompt_json, indent=4, ensure_ascii=False)

            # add markdown style JSON code block marker
            examples_prompt = "```json\n" + examples_prompt + "\n```"
        # if error occurs, print the error and continue to the next iteration
        except Exception as e:
            print(f"==== Error ==== {e}")
            continue
    
    print("==== Iterative Prompting End ====")
    return examples_prompt


if __name__ == "__main__":
    llm_model = "gpt-4o"
    scene = "office" # bathroom, bedroom, dining-room, factory, kitchen, lab, living-room, office (8 scenes)
    harmful_harmless = "harmless" # harmful, harmless
    max_iterations = 3
    os.makedirs(f"./assets/{llm_model}", exist_ok=True)

    initial_prompt = open(f"./prompts/{harmful_harmless}_prompt.txt", "r").read()
    initial_prompt = initial_prompt.replace("{SCENE}", scene)
    examples_prompt = open(f"./prompts/{harmful_harmless}_{scene}_examples.txt", "r").read()

    examples_prompt = iterative_prompting(initial_prompt, examples_prompt, llm_model, max_iterations)

    with open(f"./assets/{llm_model}/{llm_model}_{harmful_harmless}_{scene}_examples.txt", "w") as f:
        f.write(examples_prompt)