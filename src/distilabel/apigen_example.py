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

from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import ApiGenGenerator

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    generation_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 1024,
    },
)
apigen = ApiGenGenerator(use_default_structured_output=False, llm=llm)
apigen.load()

res = next(
    apigen.process(
        [
            {
                "examples": 'QUERY:\nWhat is the binary sum of 10010 and 11101?\nANSWER:\n[{"name": "binary_addition", "arguments": {"a": "10010", "b": "11101"}}]',
                "func_name": "getrandommovie",
                "func_desc": "Returns a list of random movies from a database by calling an external API.",
            }
        ]
    )
)
