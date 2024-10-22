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

import random
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.craft.utils import FormatExtractor

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns

# Taken from https://github.com/ziegler-ingo/CRAFT/blob/main/craft/generation/meta_instructions.py


class TaskType(Enum):
    """Enum representing different task types for CRAFT generation."""

    RECIPEGEN = "recipegen"
    SUMMARIZE = "summarize"
    MCQ = "mcq"
    YN_S = "yn_s"
    YN_Q = "yn_q"


class PromptInstructions:
    # used for BioQA, MedQA
    QA_MC_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one question along with four answer choices designated as A, B, C, and D based on the provided text. \
Then, respond to the question with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"question": "<question here>", "options": ["A. <option A here>", "B. <option B here>", "C. <option C here>", "D. <option D here>"], "answer": "<letter label of correct answer here>"}\
"""

    # used for common-sense QA
    QA_YN_INSTRUCTION_Q = """\
Please carefully read the text below. \
Then, generate exactly one question that is answerable with yes or no based on the provided text. \
Then, respond to the question with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"question": "<question here>", "options": ["A. Yes", "B. No"], "answer": "<letter label of correct answer here>"}\
"""

    # used for common-sense QA
    QA_YN_INSTRUCTION_S = """\
Please carefully read the text below. \
Then, generate exactly one statement that is answerable with yes or no based on the provided text. \
Then, respond to the statement with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"statement": "<statement here>", "options": ["A. Yes", "B. No"], "answer": "<letter label of correct answer here>"}\
"""

    RECIPEGEN_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one short one-sentence instruction to prepare the dish named in the text. \
Then, generate a detailed recipe for the dish by listing the required ingredients and steps. \
Return the output only as a JSON structure in this format: \
{"instruction": "<short one-sentence cooking instruction here>", "ingredients": ["<ingredient one here>", "<ingredient two here>", "< ... >", "<continue until end>"], "steps": ["<step one here>", "<step two here>", "< ... >", "<continue until end>"]}\
"""

    SUMMARIZATION_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one instruction for summarizing the text below. \
Then, generate the summary. \
Afterwards, extract a long but clean version of the initial text that encompasses the summary, but keeps almost all details from the initial text. \
Return the output only as a JSON structure in this format: \
{"instruction": "<summary instruction here>", "summary": "<summary here>", "long_but_clean_text": "<long but clean text that encompasses the summary but keeps almost all details from the initial text here>"} \
"""


class CraftGenerator(Task):
    """Generate task samples from corpus and few-shot examples.


    References:
        - [CRAFT Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation](https://arxiv.org/abs/2409.02098)
        - [CRAFT](https://github.com/ziegler-ingo/CRAFT)

    Examples:
        Generate without structured output (original implementation):

        ```python
        from distilabel.steps.tasks import CraftGenerator
        from distilabel.llms import vllm

        llm=vllm(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
        )
        craftgen = CraftGenerator(
            num_shots=2,
            task="recipegen", # or "summarization" or "mcq" or "yn"
            llm=llm
        )
        craftgen.load()

        res = next(
            craftgen.process(
                [
                    {
                        "few_shots": [],
                        "corpus": "What is the binary sum of 10010 and 11101?"
                    }
                ]
            )
        )
        res
        #{}

        ```

    """

    prompt_instructions: str
    corpus: List[str] = []
    task: TaskType

    def load(self) -> None:
        # Based on the task type, set the meta instructions or jinja template
        if self.task == TaskType.RECIPEGEN:
            self.prompt_instructions = PromptInstructions.RECIPEGEN_INSTRUCTION
        elif self.task == TaskType.SUMMARIZE:
            self.prompt_instructions = PromptInstructions.SUMMARIZATION_INSTRUCTION
        elif self.task == TaskType.MCQ:
            self.prompt_instructions = PromptInstructions.QA_MC_INSTRUCTION
        elif self.task == TaskType.YN_S:
            self.prompt_instructions = PromptInstructions.QA_YN_INSTRUCTION_S
        elif self.task == TaskType.YN_Q:
            self.prompt_instructions = PromptInstructions.QA_YN_INSTRUCTION_Q
        super().load()

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task."""
        return {
            "few_shots": True,
            "corpus": True,
        }

    @property
    def outputs(self) -> "StepColumns":
        """The output for the task are the queries and corresponding answers."""
        return ["corpus_sample", "task_sample"]

    def format_input(self, input: Dict[str, Any]):
        """
        Format the input for the task.
        """
        return [
            self._generate_few_shots(
                prompt_instruction=self.prompt_instructions,
                corpus_example=sample,
                few_shots=input.few_shots,
                task=self.task,
                num_shots=self.num_shots,
            )
            for sample in input.corpus
        ]

    def _generate_few_shots(
        self,
        prompt_instruction,
        corpus_example,
        few_shots,
        task,
        num_shots=3,
        b_inst="[INST]",
        e_inst="[/INST]",
        eos="</s>",
    ):
        """
        Create a few-shot prompt in the format described at:
        https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
        """
        out = ""  # tokenizer adds bos token at the start

        # CSQA: we sample either question or statement instruction
        if type(prompt_instruction) is list:
            prompt_instruction = random.choice(prompt_instruction)
        indices = random.sample(range(len(few_shots)), num_shots)
        for idx in indices:
            shot = few_shots[idx]
            text = shot["Text"].strip()
            if task == TaskType.MCQ:
                json_dict = FormatExtractor.qa_mc(shot, is_few_shot=True)
            elif task == TaskType.YN_S or task == TaskType.YN_Q:
                json_dict = FormatExtractor.qa_yn(shot, is_few_shot=True)
            elif task == TaskType.RECIPEGEN:
                json_dict = FormatExtractor.recipe(shot, is_few_shot=True)
            elif task == TaskType.SUMMARIZE:
                json_dict = FormatExtractor.summarization(shot, is_few_shot=True)
            else:
                raise ValueError("Unknown task.")

        formatted = f"{b_inst} {self.prompt_instructions} \n\n___________\nText: {text} {e_inst} "
        formatted += f"{json_dict}{eos} "
        out += formatted

        out += f"{b_inst} {self.prompt_instructions} \n\n___________\nText: {corpus_example['text'].strip()} {e_inst} "

        return out
