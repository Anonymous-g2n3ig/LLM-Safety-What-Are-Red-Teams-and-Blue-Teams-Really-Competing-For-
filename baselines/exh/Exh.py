import gc
import torch
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from .utils import PromptGenerator, MctsPromptGenerator, UniversalMctsPromptGenerator
import time

def running_time_record(start_time):
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    return elapsed_time, elapsed_seconds

class Exh(SingleBehaviorRedTeamingMethod):
    def __init__(self, 
                 judge_model,
                 target_model,
                 batch_size=128,
                 top_p=0.99,
                 top_k=100,
                 beam_width=1000000,
                 threshold=1000000,
                 num_workers=0,
                 response_length=128,
                 prompt_length_threshold=1000,
                 use_chat_template=False,
                 **kwargs):
        # ===== Init Judge Model =====
        # self.judge_model, allocated_gpus = load_judge_model(**judge_model)
        self.judge_modle_config = judge_model
        # ===== Init Target Model =====
        # self.model_name_or_path = target_model['model_name_or_path']
        # # manually allocate target model into unallocated gpus
        # num_gpus = target_model.get('num_gpus', 1)
        # max_memory = {i: torch.cuda.mem_get_info(i)[0] if i >= allocated_gpus else 0 for i in range(0, allocated_gpus + num_gpus)}
        # self.model, self.tokenizer = load_model_and_tokenizer(**target_model, max_memory=max_memory)
        self.target_model_config = target_model
        self.target_model_config['num_gpus'] = kwargs.get('num_gpus', 1)

        self.chat_template = get_template(target_model['model_name_or_path'], chat_template=target_model['chat_template'])
        self.batch_size = batch_size
        self.top_p = top_p
        self.top_k = top_k
        self.beam_width = beam_width
        self.threshold = threshold
        self.num_workers = num_workers
        self.response_length = response_length
        self.prompt_length_threshold = prompt_length_threshold
        self.use_chat_template = use_chat_template

        # Initialization
        self.top_p = 100
        self.top_k = 15
        self.beam_width = 15
        self.prompt_generator = PromptGenerator(self.judge_modle_config, self.target_model_config, self.response_length, self.prompt_length_threshold, self.threshold, self.chat_template, self.use_chat_template)
        # self.prompt_generator = MctsPromptGenerator(self.judge_modle_config, self.pjudge_model_config, self.target_model_config, self.response_length, self.prompt_length_threshold, self.threshold, self.chat_template, self.use_chat_template)
        # self.prompt_generator = UniversalMctsPromptGenerator(self.judge_modle_config, self.pjudge_model_config, self.target_model_config, self.response_length, self.prompt_length_threshold, self.threshold, self.chat_template, self.use_chat_template)
        # self.prompt_generator = TestPromptGenerator(self.judge_modle_config, self.pjudge_model_config, self.target_model_config, self.response_length, self.prompt_length_threshold, self.threshold, self.chat_template, self.use_chat_template)

    def generate_test_cases_single_behavior(self, behavior_dict, verbose, **kwargs):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        batch_size = self.batch_size

        # ========== Behavior meta data ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']

        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id}')

        logs = []
        test_case = behavior

        # ==== Generate prompt =====
        start_time = time.time()

        test_case, response, number_of_tried_prompts, successful_attack = self.prompt_generator.generate_adv_prompt(test_case, batch_size, self.top_p, self.top_k, beam_width=self.beam_width, behavior_dict=behavior_dict)

        running_time, running_seconds = running_time_record(start_time)

        logs.append(dict(test_case=test_case, number_of_tried_prompts=number_of_tried_prompts, successful_attack=successful_attack, running_time=running_time, running_seconds=running_seconds, response=response))

        torch.cuda.empty_cache()
        gc.collect()

        return test_case, logs
