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

import numpy as np

from distilabel.embeddings.llamacpp import LlamaCppEmbeddings


class TestLlamaCppEmbeddings:
    model_name = "all-MiniLM-L6-v2-Q2_K.gguf"
    repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"

    def test_model_name(self) -> None:
        """
        Test if the model name is correctly set.
        """
        embeddings = LlamaCppEmbeddings(model=self.model_name)
        assert embeddings.model_name == self.model_name

    def test_encode(self, local_llamacpp_model_path) -> None:
        """
        Test if the model can generate embeddings.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """
        embeddings = LlamaCppEmbeddings(model=local_llamacpp_model_path)
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        embeddings.load()
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_load_model_from_local(self, local_llamacpp_model_path):
        """
        Test if the model can be loaded from a local file and generate embeddings.

        Args:
            local_llamacpp_model_path (str): Fixture providing the local model path.
        """
        embeddings = LlamaCppEmbeddings(model=local_llamacpp_model_path)
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]
        embeddings.load()
        # Test if the model is loaded by generating an embedding
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_load_model_from_repo(self):
        """
        Test if the model can be loaded from a Hugging Face repository.
        """
        embeddings = LlamaCppEmbeddings(
            hub_repository_id=self.repo_id,
            model=self.model_name,
            normalize_embeddings=True,
        )
        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        embeddings.load()
        # Test if the model is loaded by generating an embedding
        results = embeddings.encode(inputs=inputs)

        for result in results:
            assert len(result) == 384

    def test_normalize_embeddings_true(self, local_llamacpp_model_path):
        """
        Test if embeddings are normalized when normalize_embeddings is True.
        """
        embeddings = LlamaCppEmbeddings(
            model=local_llamacpp_model_path, normalize_embeddings=True
        )
        embeddings.load()

        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        results = embeddings.encode(inputs=inputs)

        for result in results:
            # Check if the embedding is normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(result)
            assert np.isclose(
                norm, 1.0, atol=1e-6
            ), f"Norm is {norm}, expected close to 1.0"

    def test_normalize_embeddings_false(self, local_llamacpp_model_path):
        """
        Test if embeddings are not normalized when normalize_embeddings is False.
        """
        embeddings = LlamaCppEmbeddings(
            model=local_llamacpp_model_path, normalize_embeddings=False
        )
        embeddings.load()

        inputs = [
            "Hello, how are you?",
            "What a nice day!",
            "I hear that llamas are very popular now.",
        ]

        results = embeddings.encode(inputs=inputs)

        for result in results:
            # Check if the embedding is not normalized (L2 norm should not be close to 1)
            norm = np.linalg.norm(result)
            assert not np.isclose(
                norm, 1.0, atol=1e-6
            ), f"Norm is {norm}, expected not close to 1.0"

        # Additional check: ensure that at least one embedding has a norm significantly different from 1
        norms = [np.linalg.norm(result) for result in results]
        assert any(
            not np.isclose(norm, 1.0, atol=0.1) for norm in norms
        ), "Expected at least one embedding with norm not close to 1.0"
