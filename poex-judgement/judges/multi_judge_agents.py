from fastchat.model import get_conversation_template
from judges.multi_agent_base import JudgingAgentBase
from openai import OpenAI
import time
import emoji
from typing import List, Dict
import re
import os
def remove_code_blocks(text):
    pattern = r"```.*?```"
    return re.sub(pattern, '', text, flags=re.DOTALL)

def text_process(model_response):
    model_response = emoji.replace_emoji(model_response, replace='')
    model_response = remove_code_blocks(model_response)

    model_response = model_response.replace("```", "")

    return  model_response

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

class AgentGPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = f""" Error """
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        # filter text
        for i in range(len(conv)):
            if 'content' in conv[i]:
                conv[i]['content'] = text_process(conv[i]['content'])

        for attempt in range(self.API_MAX_RETRY):
            try:
                
                client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"]
                )
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_n_tokens,
                    timeout=self.API_TIMEOUT
                )

                if response:
                    output = response.choices[0].message.content

                print(f"""\n{'=' * 80}\n Output: {output} \n{'=' * 80}\n""")
                break  # Exit the loop if the request is successful
            except Exception as e:
                print(f"\n\nException: {str(e)}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)

        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class JudgingAgent(JudgingAgentBase):
    def __init__(self, args, judging_agent_id, judging_type):
        super(JudgingAgent, self).__init__(args, judging_agent_id, judging_type)
        self.judging_model = AgentGPT(model_name=self.judging_name)

    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.judging_template_name)
        conv.set_system_message(self.set_system_prompt())
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def reason_score(self, user_input_list, model_response_list):
        convs_list = [self.create_conv(self.get_judging_prompt(user_query, response)) for user_query, response in
                      zip(user_input_list, model_response_list)]
        raw_outputs = self.judging_model.batched_generate(convs_list,
                                                        max_n_tokens=self.max_n_tokens,
                                                        temperature=self.temperature)
        reason_outputs = [self.process_output(raw_output)["reason"] for raw_output in raw_outputs]
        score_outputs = [self.process_output(raw_output)["score"] for raw_output in raw_outputs]
        return reason_outputs, score_outputs