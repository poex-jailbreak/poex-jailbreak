import re
from judges.agent_templates import get_judging_instruct, get_system_prompt

class JudgingAgentBase:
    def __init__(self, args, judging_agent_id, judging_type):
        self.id = judging_agent_id
        self.judging_type = judging_type  # 'rejection', 'harmful', 'logical', 'redundant'
        self.max_n_tokens = args.judging_max_n_tokens
        self.temperature = args.judging_temperature
        self.judging_name = args.judging_model
        self.judging_template_name = args.judging_template_name
        self.judging_top_p = args.judging_top_p

    def set_system_prompt(self):
        system_prompt = get_system_prompt(self.judging_type)
        return system_prompt


    def get_judging_prompt(self, user_input, model_output):
        judging_prompt = get_judging_instruct(user_input, model_output, self.judging_type)
        return judging_prompt
    


    def process_output(self, raw_output):
        try:
            reason_answer_pattern = r'<reason>(.*?)</reason>'
            reason_text = re.findall(reason_answer_pattern, raw_output)
            if reason_text:
                reason_text = reason_text[-1]
            else:
                reason_text = "Error"
        except Exception as e:
            reason_text = "Error"
            print(f"An reason error occurred: {e}")

        try:
            score_answer_pattern = r'<score>(.*?)</score>'
            score_text = re.findall(score_answer_pattern, raw_output)
            if score_text:
                score_text = score_text[-1]
            else:
                score_text = "-1"
        except Exception as e:
            score_text = "-1"
            print(f"An score error occurred: {e}")
        
        extracted_dict = {"reason": reason_text,
                          "score": int(score_text)}
        return extracted_dict