import os
os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

import logging
from datetime import datetime, timedelta
from typing import cast, List, Union

from huggingface_hub import hf_hub_download, snapshot_download

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    apply_mistral_chat_template,
    ChatCompletionMessageParam,
    parse_chat_messages,
)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.utils import is_list_of


MODEL_NAME = "mistralai/Pixtral-Large-Instruct-2411"
TENSOR_PARALLEL_SIZE = 16
IMAGE_PER_PROMPT = 4

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO Delete this after fixing Mistral model's checkpoint loading issue.
snapshot_download(MODEL_NAME)


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, 'r') as file:
        system_prompt = file.read()
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


def get_model_config():
    engine_args = EngineArgs(
        model=MODEL_NAME,
        tokenizer=None,
        tokenizer_mode="mistral",
        skip_tokenizer_init=False,
        trust_remote_code=False,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        quantization=None,
        revision=None,
        tokenizer_revision=None,
        seed=0,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cpu_offload_gb=0,
        enforce_eager=None,
        max_context_len_to_capture=None,
        max_seq_len_to_capture=8192,
        disable_custom_all_reduce=False,
        disable_async_output_proc=False,
        mm_processor_kwargs=None,
        # kwargs
        config_format="mistral",
        load_format="mistral",
        limit_mm_per_prompt={"image": IMAGE_PER_PROMPT},
        disable_log_stats=True,
    )
    engine_config = engine_args.create_engine_config()
    model_config = engine_config.model_config
    return model_config


def get_prompt():
    # Set raw prompts.
    SYSTEM_PROMPT = load_system_prompt(MODEL_NAME, "SYSTEM_PROMPT.txt")
    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Which of the depicted countries has the best food? Which the second and third and fourth? Name the country, its color on the map and one its city that is visible on the map, but is not the capital. Make absolutely sure to only name a city that can be seen on the map.",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    # Get tokenized prompts.
    tokenizer = MistralTokenizer.from_pretrained(MODEL_NAME, revision=None)
    prompts: List[Union[TokensPrompt, TextPrompt]] = []

    list_of_messages: List[List[ChatCompletionMessageParam]]
    # Handle multi and single conversations
    if is_list_of(messages, list):
        # messages is List[List[...]]
        list_of_messages = cast(
            List[List[ChatCompletionMessageParam]],
            messages,
        )
    else:
        # messages is List[...]
        list_of_messages = [
            cast(List[ChatCompletionMessageParam], messages)
        ]

    for msgs in list_of_messages:
        model_config = get_model_config()
        _, mm_data = parse_chat_messages(msgs, model_config, tokenizer)
        prompt_data = apply_mistral_chat_template(
            tokenizer,
            messages=msgs,
            chat_template=None,
            add_generation_prompt=True,
            tools=None,
        )

        prompt: Union[TokensPrompt, TextPrompt]
        if is_list_of(prompt_data, int):
            prompt = TokensPrompt(prompt_token_ids=prompt_data)
        else:
            prompt = TextPrompt(prompt=prompt_data)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        prompts.append(prompt)

    return prompts


def main():
    # Initialize LLM.
    llm = LLM(
        model=MODEL_NAME,
        max_num_seqs=1,
        # max_model_len=32,
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        limit_mm_per_prompt={"image": IMAGE_PER_PROMPT},
        device="neuron",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )

    # Get tokenized prompts and sampling parameters.
    prompts = get_prompt()
    sampling_params = SamplingParams(max_tokens=32)

    # Generate Output.
    outputs = llm.generate(prompts,sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
