# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import torch

COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def create_llama_chat_prompt(
    instruction,
    system_prompt,
    start_header="<|start_header_id|>",
    end_header="<|end_header_id|>\n\n",
    eot_id="<|eot_id|>",
):
    """
    Create the instruction and system prompt format for Llama 3 models as described here:
    https://github.com/meta-llama/llama-recipes
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3
    https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L222
    """
    return f"""\
{start_header}system{end_header}\
{system_prompt}{eot_id}{start_header}user{end_header}\
{instruction}{eot_id}{start_header}assistant{end_header}\
"""


def create_mistral_inst(
    instruction,
    b_inst="[INST]",
    e_inst="[/INST]",
):
    return f"{b_inst} {instruction.strip()} {e_inst}"


class FormatExtractor:
    @staticmethod
    def qa_mc(
        sample,
        is_few_shot=False,
        return_dict=False,
    ):
        """
        Filter and clean BioQA and MedQA multiple choice task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def qa_yn(
        sample,
        is_few_shot=False,
        return_dict=False,
    ):
        """
        Filter and clean common sense QA yes-no task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def recipe(
        sample,
        is_few_shot=False,
        return_dict=False,
    ):
        """
        Filter and clean recipe task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction = sample["Instruction"].strip()
            output_parts = sample["Output"].split("Steps:")
            out_dict = {
                "instruction": instruction,
                "ingredients": [
                    ing.strip() for ing in output_parts[0].split("\n")[1:] if ing
                ],
                "steps": [step.strip() for step in output_parts[1].split("\n") if step],
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def summarization(
        sample,
        is_few_shot=False,
        return_dict=False,
    ):
        """
        Filter and clean summarization task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = [
                part.strip() for part in sample["Instruction"].split("\n") if part
            ]
            out_dict = {
                "instruction": instruction_parts[0],
                "summary": sample["Output"].strip(),
                "long_but_clean_text": "\n".join(instruction_parts[1:]),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        out_json = json.dumps(out_dict)

        return out_json


class QuestionLengthError(Exception):
    pass


class OptionsNumberError(Exception):
    pass


class AnswerFormatError(Exception):
    pass
