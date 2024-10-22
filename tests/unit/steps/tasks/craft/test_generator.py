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
from typing import TYPE_CHECKING, List

import pytest

from distilabel.steps.tasks.craft.generator import CraftGenerator, TaskType
from tests.unit.conftest import DummyLLM

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput

import json


class DummyCraftGenLLM(DummyLLM):
    def generate(
        self, inputs: List["FormattedInput"], num_generations: int = 1
    ) -> "GenerateOutput":
        query_answers = [
            {
                "query": "What information can be obtained about the Maine Coon cat breed?",
                "answers": [
                    {
                        "name": "get_breed_information",
                        "arguments": {"breed": "Maine Coon"},
                    }
                ]
                * self.number,
            }
        ]
        if self.use_structured_output:
            query_answers = {"pairs": query_answers}
        return [
            [json.dumps(query_answers) for _ in range(num_generations)]
            for _ in range(len(inputs))
        ]


class TestCraftGenerator:
    def test_format_input(
        self,
        corpus_sample: str,
    ) -> None:
        random.seed(42)
        task = CraftGenerator(
            llm=DummyLLM(),
            task=TaskType.MCQ,
        )
        task.load()

    def test_mcq_generation(
        self, corpus_sample: str, mcq_gen_few_shots: List[str]
    ) -> None:
        """
        Test the multiple-choice question generation functionality.

        This test ensures that the generated task_sample has the correct structure
        for a multiple-choice question.
        """
        random.seed(42)
        task = CraftGenerator(
            llm=DummyCraftGenLLM(),
            task=TaskType.MCQ,
            corpus_sample=corpus_sample,
        )
        task.load()
        result = next(task.process(mcq_gen_few_shots))[0]
        assert "task_sample" in result
        task_sample = result["task_sample"]
        assert isinstance(task_sample, str)

        # Parse the task_sample JSON
        try:
            mcq_data = json.loads(task_sample)
        except json.JSONDecodeError:
            pytest.fail("task_sample is not a valid JSON string")

        # Assert the structure of the MCQ data
        assert isinstance(mcq_data, dict), "task_sample should be a dictionary"
        assert "question" in mcq_data, "task_sample should contain a 'question' key"
        assert isinstance(mcq_data["question"], str), "question should be a string"

        assert "options" in mcq_data, "task_sample should contain an 'options' key"
        assert isinstance(mcq_data["options"], list), "options should be a list"
        assert len(mcq_data["options"]) == 4, "options should contain exactly 4 items"
        for option in mcq_data["options"]:
            assert isinstance(option, str), "each option should be a string"
            assert option.startswith(
                ("A. ", "B. ", "C. ", "D. ")
            ), "options should start with A., B., C., or D."

        assert "answer" in mcq_data, "task_sample should contain an 'answer' key"
        assert isinstance(mcq_data["answer"], str), "answer should be a string"
        assert mcq_data["answer"] in [
            "A",
            "B",
            "C",
            "D",
        ], "answer should be A, B, C, or D"

    def test_q_yn_generation(
        self, corpus_sample: str, yn_gen_few_shots: List[str]
    ) -> None:
        """
        Test the yes/no question generation functionality.

        This test ensures that the generated task_sample has the correct structure
        for a yes/no question.
        """
        random.seed(42)
        task = CraftGenerator(
            llm=DummyCraftGenLLM(),
            corpus_sample=corpus_sample,
            task=TaskType.YN_Q,
        )
        task.load()
        result = next(task.process(yn_gen_few_shots))[0]
        assert "task_sample" in result
        task_sample = result["task_sample"]
        assert isinstance(task_sample, str)

        # Parse the task_sample JSON
        try:
            yn_data = json.loads(task_sample)
        except json.JSONDecodeError:
            pytest.fail("task_sample is not a valid JSON string")

        # Assert the structure of the yes/no question data
        assert isinstance(yn_data, dict), "task_sample should be a dictionary"
        assert "question" in yn_data, "task_sample should contain a 'question' key"
        assert isinstance(yn_data["question"], str), "question should be a string"

        assert "options" in yn_data, "task_sample should contain an 'options' key"
        assert isinstance(yn_data["options"], list), "options should be a list"
        assert len(yn_data["options"]) == 2, "options should contain exactly 2 items"
        assert yn_data["options"] == [
            "A. Yes",
            "B. No",
        ], "options should be ['A. Yes', 'B. No']"

        assert "answer" in yn_data, "task_sample should contain an 'answer' key"
        assert isinstance(yn_data["answer"], str), "answer should be a string"
        assert yn_data["answer"] in ["A", "B"], "answer should be either 'A' or 'B'"

    def test_s_yn_generation(
        self, corpus_sample: str, yn_gen_few_shots: List[str]
    ) -> None:
        """
        Test the statement-based yes/no generation functionality.

        This test ensures that the generated task_sample has the correct structure
        for a statement-based yes/no question.
        """
        random.seed(42)
        task = CraftGenerator(
            llm=DummyCraftGenLLM(),
            corpus_sample=corpus_sample,
        )
        task.load()
        result = next(task.process(yn_gen_few_shots))[0]
        assert "task_sample" in result
        task_sample = result["task_sample"]
        assert isinstance(task_sample, str)

        # Parse the task_sample JSON
        try:
            s_yn_data = json.loads(task_sample)
        except json.JSONDecodeError:
            pytest.fail("task_sample is not a valid JSON string")

        # Assert the structure of the statement-based yes/no data
        assert isinstance(s_yn_data, dict), "task_sample should be a dictionary"
        assert "statement" in s_yn_data, "task_sample should contain a 'statement' key"
        assert isinstance(s_yn_data["statement"], str), "statement should be a string"

        assert "options" in s_yn_data, "task_sample should contain an 'options' key"
        assert isinstance(s_yn_data["options"], list), "options should be a list"
        assert len(s_yn_data["options"]) == 2, "options should contain exactly 2 items"
        assert s_yn_data["options"] == [
            "A. Yes",
            "B. No",
        ], "options should be ['A. Yes', 'B. No']"

        assert "answer" in s_yn_data, "task_sample should contain an 'answer' key"
        assert isinstance(s_yn_data["answer"], str), "answer should be a string"
        assert s_yn_data["answer"] in ["A", "B"], "answer should be either 'A' or 'B'"

    def test_recipe_generation(
        self, corpus_sample: str, recipe_generation_few_shots: List[str]
    ) -> None:
        """
        Test the recipe generation functionality.

        This test ensures that the generated task_sample has the correct structure
        for a recipe.
        """
        random.seed(42)
        task = CraftGenerator(
            llm=DummyCraftGenLLM(),
            corpus_sample=corpus_sample,
        )
        task.load()
        result = next(task.process(recipe_generation_few_shots))[0]
        assert "task_sample" in result
        task_sample = result["task_sample"]
        assert isinstance(task_sample, str)

        # Parse the task_sample JSON
        try:
            recipe_data = json.loads(task_sample)
        except json.JSONDecodeError:
            pytest.fail("task_sample is not a valid JSON string")

        # Assert the structure of the recipe data
        assert isinstance(recipe_data, dict), "task_sample should be a dictionary"

        assert (
            "instruction" in recipe_data
        ), "task_sample should contain an 'instruction' key"
        assert isinstance(
            recipe_data["instruction"], str
        ), "instruction should be a string"

        assert (
            "ingredients" in recipe_data
        ), "task_sample should contain an 'ingredients' key"
        assert isinstance(
            recipe_data["ingredients"], list
        ), "ingredients should be a list"
        assert (
            len(recipe_data["ingredients"]) > 0
        ), "ingredients list should not be empty"
        for ingredient in recipe_data["ingredients"]:
            assert isinstance(ingredient, str), "each ingredient should be a string"

        assert "steps" in recipe_data, "task_sample should contain a 'steps' key"
        assert isinstance(recipe_data["steps"], list), "steps should be a list"
        assert len(recipe_data["steps"]) > 0, "steps list should not be empty"
        for step in recipe_data["steps"]:
            assert isinstance(step, str), "each step should be a string"

    def test_summarization_generation(
        self, corpus_sample: str, summarization_generation_few_shots: List[str]
    ) -> None:
        """
        Test the summarization generation functionality.

        This test ensures that the generated task_sample has the correct structure
        for a summarization task.
        """
        random.seed(42)
        task = CraftGenerator(
            llm=DummyCraftGenLLM(),
            corpus_sample=corpus_sample,
        )
        task.load()
        result = next(task.process(summarization_generation_few_shots))[0]
        assert "task_sample" in result
        task_sample = result["task_sample"]
        assert isinstance(task_sample, str)

        # Parse the task_sample JSON
        try:
            summarization_data = json.loads(task_sample)
        except json.JSONDecodeError:
            pytest.fail("task_sample is not a valid JSON string")

        # Assert the structure of the summarization data
        assert isinstance(
            summarization_data, dict
        ), "task_sample should be a dictionary"

        assert (
            "instruction" in summarization_data
        ), "task_sample should contain an 'instruction' key"
        assert isinstance(
            summarization_data["instruction"], str
        ), "instruction should be a string"

        assert (
            "summary" in summarization_data
        ), "task_sample should contain a 'summary' key"
        assert isinstance(
            summarization_data["summary"], str
        ), "summary should be a string"

        assert (
            "long_but_clean_text" in summarization_data
        ), "task_sample should contain a 'long_but_clean_text' key"
        assert isinstance(
            summarization_data["long_but_clean_text"], str
        ), "long_but_clean_text should be a string"

        # Additional checks to ensure the summary is shorter than the long text
        assert len(summarization_data["summary"]) < len(
            summarization_data["long_but_clean_text"]
        ), "summary should be shorter than the long_but_clean_text"
