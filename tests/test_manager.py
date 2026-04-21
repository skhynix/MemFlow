# Copyright 2026 SK hynix Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemFlowManager."""

import os
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from memflow.manager import MemFlowManager, _load_env_file
from memflow.models import Procedure, Step, TaskPlan, StepResult, RunResult, StepType
from memflow.store import EmulatedStore, FileStore, MemMachineStore, MemMachineBypass


class TestMemFlowManagerInit:
    """Tests for MemFlowManager initialization."""

    def test_init_with_llm_and_store(self, fake_llm):
        """Test initialization with LLM and store."""
        store = EmulatedStore()
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        assert manager.llm == fake_llm
        assert manager.store == store

    def test_init_with_llm_only(self, fake_llm):
        """Test initialization creates default EmulatedStore."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        assert isinstance(manager.store, EmulatedStore)

    def test_init_with_bypass(self, fake_llm):
        """Test initialization with bypass."""
        bypass = MemMachineBypass()
        manager = MemFlowManager(llm=fake_llm, bypass=bypass, use_env=False)

        # Note: No public getter for bypass, so we verify it was accepted
        # by checking the manager was created successfully with bypass
        assert manager is not None
        assert manager.llm is fake_llm

    def test_planner_lazy_initialization(self, fake_llm):
        """Test that planner is lazily initialized on first use."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # Trigger planner creation by calling plan() with mocked LLMPlanner
        with patch("memflow.manager.LLMPlanner") as mock_planner_cls:
            mock_planner_cls.return_value.plan.return_value = TaskPlan(task="Test", steps=[])
            result = manager.plan("Test task")

        # Verify planner was called with correct kwargs
        mock_planner_cls.assert_called_once()
        call_kwargs = mock_planner_cls.call_args.kwargs
        assert call_kwargs["llm"] is fake_llm
        assert isinstance(result, TaskPlan)

    def test_executor_lazy_initialization(self, fake_llm):
        """Test that executor is lazily initialized on first use."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # Trigger executor creation by calling run() with mocked dependencies
        with patch("memflow.manager.LLMPlanner"), \
             patch("memflow.manager.ToolRegistry") as mock_registry_cls, \
             patch("memflow.manager.Learner"):
            mock_registry_cls.return_value.execute_step.return_value = StepResult(
                step_id="step-1", success=True, output="test"
            )
            result = manager.run("Test task")

        # Verify executor was called with correct kwargs
        mock_registry_cls.assert_called_once()
        call_kwargs = mock_registry_cls.call_args.kwargs
        assert call_kwargs["llm"] is fake_llm
        assert isinstance(result, RunResult)

    def test_learner_lazy_initialization(self, fake_llm):
        """Test that learner is lazily initialized on first use."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # Trigger learner creation by calling run() with mocked dependencies
        with patch("memflow.manager.LLMPlanner"), \
             patch("memflow.manager.ToolRegistry"), \
             patch("memflow.manager.Learner") as mock_learner_cls:
            mock_learner = MagicMock()
            mock_learner.extract.return_value = Procedure(
                title="Test", content="1. Step"
            )
            mock_learner_cls.return_value = mock_learner

            # Use multi_stage=False for simpler single-shot execution path
            result = manager.run("Test task", multi_stage=False)

        # Verify learner was instantiated and extract was called
        mock_learner_cls.assert_called_once()
        mock_learner.extract.assert_called_once()
        assert isinstance(result, RunResult)


class TestMemFlowManagerFromEnv:
    """Tests for MemFlowManager.from_env() - now via use_env=True."""

    @patch("memflow.manager.LLMFactory")
    def test_from_env_defaults(self, mock_factory, clean_env):
        """Test initialization with use_env=True uses default values."""
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        manager = MemFlowManager(use_env=True)

        # Verify LLMFactory.create was called with ollama provider
        mock_factory.create.assert_called_once()
        call_args = mock_factory.create.call_args
        assert call_args.args[0] == "ollama"
        assert call_args.kwargs.get("api_base") == "http://localhost:11434"
        assert isinstance(manager.store, EmulatedStore)

    @patch("memflow.manager.LLMFactory")
    def test_from_env_with_ollama(self, mock_factory, clean_env):
        """Test initialization with use_env=True with ollama provider."""
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_MODEL"] = "llama3.2"

        manager = MemFlowManager(use_env=True)

        assert isinstance(manager.store, EmulatedStore)

    @patch("memflow.manager.LLMFactory")
    def test_from_env_with_file_backend(self, mock_factory, clean_env):
        """Test initialization with use_env=True with file backend."""
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        os.environ["MEMFLOW_BACKEND"] = "file"
        os.environ["MEMFLOW_DATA_DIR"] = "/tmp/test_data"

        manager = MemFlowManager(use_env=True)

        assert isinstance(manager.store, FileStore)

    @patch("memflow.manager.LLMFactory")
    def test_from_env_with_memmachine_backend(self, mock_factory, clean_env):
        """Test initialization with use_env=True with memmachine backend."""
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        os.environ["MEMFLOW_BACKEND"] = "memmachine"
        os.environ["MEMMACHINE_BASE_URL"] = "http://test:8080"
        os.environ["MEMMACHINE_ORG_ID"] = "test-org"
        os.environ["MEMMACHINE_PROJECT"] = "test-project"

        manager = MemFlowManager(use_env=True)

        assert isinstance(manager.store, MemMachineStore)
        # Note: No public getter for bypass, verify manager was created successfully
        assert manager is not None

    @patch("memflow.manager.LLMFactory")
    def test_from_env_with_openai_provider(self, mock_factory, clean_env):
        """Test initialization with use_env=True with openai-compatible provider."""
        mock_llm = MagicMock()
        mock_factory.create.return_value = mock_llm

        os.environ["LLM_PROVIDER"] = "openai-compatible"
        os.environ["LLM_MODEL"] = "gpt-4"
        os.environ["LLM_API_BASE"] = "http://vllm:8000/v1"
        os.environ["LLM_API_KEY"] = "test-key"

        manager = MemFlowManager(use_env=True)

        mock_factory.create.assert_called_once_with(
            "openai-compatible",
            model="gpt-4",
            api_base="http://vllm:8000/v1",
            api_key="test-key"
        )


class TestMemFlowManagerAdd:
    """Tests for MemFlowManager.add()."""

    def test_add_procedure_direct(self, fake_llm):
        """Test adding a procedure directly."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)
        proc = Procedure(title="Test", content="1. Step")

        result = manager.add(procedure=proc)

        assert result["event"] == "ADD"
        assert result["id"] == proc.id
        assert proc in [p for p in manager.store.list_all()]

    def test_add_procedure_requires_either_messages_or_procedure(self, fake_llm):
        """Test that add requires either messages or procedure."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        with pytest.raises(ValueError, match="Either 'messages' or 'procedure'"):
            manager.add()

    def test_add_with_string_messages_keyword_detection(self, fake_llm):
        """Test add with string messages - classification is now LLM-based."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # Set LLM to classify as "none" (non-procedural)
        fake_llm.set_response('{"type": "none"}')
        result = manager.add(messages="This has no procedural keywords")

        # LLM-based classification now - should be skipped as "classified as none"
        assert result.get("skipped") == "classified as none"

    def test_add_with_procedural_keywords(self, fake_llm):
        """Test add with procedural keywords triggers extraction."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # LLM will classify as procedural and extract
        result = manager.add(messages="how to deploy step 1 step 2")

        # Should have attempted extraction
        assert "results" in result


class TestMemFlowManagerSearch:
    """Tests for MemFlowManager.search()."""

    def test_search_delegates_to_store(self, fake_llm):
        """Test that search delegates to store."""
        store = EmulatedStore()
        store.add(Procedure(title="Deploy guide", content="1. Deploy"))
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        results = manager.search("deploy")

        assert len(results) == 1
        assert results[0].procedure.title == "Deploy guide"

    def test_search_with_user_id(self, fake_llm):
        """Test search filters by user_id."""
        store = EmulatedStore()
        store.add(Procedure(title="User1 deploy", content="1. Deploy", user_id="user1"))
        store.add(Procedure(title="User2 deploy", content="1. Deploy", user_id="user2"))
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        results = manager.search("deploy", user_id="user1")

        assert len(results) == 1
        assert results[0].procedure.user_id == "user1"

    def test_search_with_top_k(self, fake_llm):
        """Test search limits results to top_k."""
        store = EmulatedStore()
        for i in range(10):
            store.add(Procedure(title=f"Deploy {i}", content=f"1. Deploy {i}"))
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        results = manager.search("deploy", top_k=3)

        assert len(results) == 3


class TestMemFlowManagerChat:
    """Tests for MemFlowManager.chat()."""

    def test_chat_with_procedures(self, fake_llm):
        """Test chat with retrieved procedures."""
        store = EmulatedStore()
        # Add procedure with content that will match the search query
        store.add(Procedure(title="How to deploy", content="1. Run deploy.sh to deploy the application"))
        # First call: intent classification -> SEARCH
        # Second call: chat response generation (SEARCH handler uses LLM for response)
        fake_llm.set_response('{"intents": ["SEARCH"], "primary": "SEARCH"}')
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        # First verify search works directly
        search_results = manager.search("How to deploy?")
        assert len(search_results) > 0, "Search should find the procedure"

        result = manager.chat("How to deploy?")

        # chat() returns dict with response key
        assert "response" in result
        # The SEARCH handler returns procedures text
        assert "How to deploy" in result["response"] or "deploy.sh" in result["response"]
        assert len(fake_llm.generate_calls) >= 1

    def test_chat_without_procedures(self, fake_llm):
        """Test chat when no procedures found."""
        store = EmulatedStore()
        fake_llm.set_response('{"intents": ["CONVERSATION"], "primary": "CONVERSATION"}')
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        result = manager.chat("How to deploy?")

        assert len(fake_llm.generate_calls) > 0
        assert "response" in result

    def test_chat_auto_learn_async(self, fake_llm):
        """Test that chat with ADD intent triggers learning via handler."""
        store = EmulatedStore()

        # Configure LLM to respond to intent classification and extraction
        # First call: intent classification, Second call: extraction
        fake_llm.response = '{"intents": ["ADD"], "primary": "ADD"}'
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        result = manager.chat("Test query")

        # Verify ADD intent was processed
        assert "ADD" in result.get("intents", [])
        # Check that handler_results contains ADD handler result
        assert "ADD" in result.get("handler_results", {})


class TestMemFlowManagerPlan:
    """Tests for MemFlowManager.plan()."""

    @patch("memflow.manager.LLMPlanner")
    def test_plan_retrieves_context(self, mock_planner_cls, fake_llm):
        """Test that plan retrieves relevant procedures as context."""
        store = EmulatedStore()
        # Use title and content that will match the search query "deploy app"
        store.add(Procedure(title="Deploy Guide", content="1. Run deploy.sh"))

        mock_plan = TaskPlan(task="Deploy", steps=[])
        mock_planner_cls.return_value.plan.return_value = mock_plan

        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)
        plan = manager.plan("deploy app")

        # Verify planner was called with context containing procedure
        mock_planner_cls.return_value.plan.assert_called_once()
        call_kwargs = mock_planner_cls.return_value.plan.call_args.kwargs
        context = call_kwargs.get("context", "")
        # Context should contain the retrieved procedure title and content
        assert "Deploy Guide" in context
        assert "deploy.sh" in context

    @patch("memflow.manager.LLMPlanner")
    def test_plan_creates_planner_if_needed(self, mock_planner_cls, fake_llm):
        """Test that plan creates planner if not exists."""
        fake_llm.set_response('{"type": "procedural"}')
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        mock_plan = TaskPlan(task="Test", steps=[])
        mock_planner_cls.return_value.plan.return_value = mock_plan

        manager.plan("Test task")

        # Verify planner was called with correct kwargs
        mock_planner_cls.assert_called_once()
        call_kwargs = mock_planner_cls.call_args.kwargs
        assert call_kwargs["llm"] is fake_llm

    @patch("memflow.manager.LLMPlanner")
    def test_plan_returns_task_plan(self, mock_planner_cls, fake_llm):
        """Test that plan returns TaskPlan."""
        fake_llm.set_response('{"type": "procedural"}')
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        mock_plan = TaskPlan(
            task="Deploy",
            steps=[Step(id="step-1", goal="Deploy", type=StepType.TOOL, tool_name="bash")]
        )
        mock_planner_cls.return_value.plan.return_value = mock_plan

        plan = manager.plan("Deploy app")

        assert isinstance(plan, TaskPlan)
        assert plan.task == "Deploy"


class TestMemFlowManagerRun:
    """Tests for MemFlowManager.run()."""

    @patch("memflow.manager.LLMPlanner")
    @patch("memflow.manager.ToolRegistry")
    @patch("memflow.manager.Learner")
    def test_run_full_pipeline(self, mock_learner_cls, mock_registry_cls, mock_planner_cls, fake_llm):
        """Test full run: plan -> execute -> learn."""
        fake_llm.set_response('{"type": "procedural"}')
        store = EmulatedStore()
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        # Mock plan
        mock_plan = TaskPlan(
            task="Deploy",
            steps=[Step(id="step-1", goal="Deploy", type=StepType.TOOL, tool_name="bash", args={"command": "./deploy.sh"})]
        )
        mock_planner_cls.return_value.plan.return_value = mock_plan

        # Mock executor
        mock_registry = MagicMock()
        mock_registry.execute_step.return_value = StepResult(
            step_id="step-1",
            success=True,
            output="Deployed"
        )
        mock_registry_cls.return_value = mock_registry

        # Mock learner
        mock_learned = Procedure(title="Learned Deploy", content="1. Run deploy.sh")
        mock_learner = MagicMock()
        mock_learner.extract.return_value = mock_learned
        mock_learner_cls.return_value = mock_learner

        result = manager.run("Deploy app")

        assert isinstance(result, RunResult)
        assert result.plan == mock_plan
        assert len(result.step_results) == 1
        assert result.learned == mock_learned

    @patch("memflow.manager.LLMPlanner")
    @patch("memflow.manager.ToolRegistry")
    @patch("memflow.manager.Learner")
    def test_run_with_custom_tools(self, mock_learner_cls, mock_registry_cls, mock_planner_cls, fake_llm):
        """Test run with custom tools."""
        fake_llm.set_response('{"type": "procedural"}')
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        mock_plan = TaskPlan(task="Test", steps=[])
        mock_planner_cls.return_value.plan.return_value = mock_plan

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        def custom_tool(arg: str) -> str:
            return f"Custom: {arg}"

        manager.run("Test", tools={"custom": custom_tool})

        # Verify custom tool was registered
        mock_registry.register.assert_called_with("custom", custom_tool)

    @patch("memflow.manager.LLMPlanner")
    @patch("memflow.manager.ToolRegistry")
    @patch("memflow.manager.Learner")
    def test_run_stores_learned_procedure(self, mock_learner_cls, mock_registry_cls, mock_planner_cls, fake_llm):
        """Test that run stores the learned procedure."""
        store = EmulatedStore()
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        mock_plan = TaskPlan(task="Test", steps=[])
        mock_planner_cls.return_value.plan.return_value = mock_plan

        mock_registry = MagicMock()
        mock_registry_cls.return_value = mock_registry

        mock_learner = MagicMock()
        mock_learner_cls.return_value = mock_learner

        learned_proc = Procedure(title="Learned", content="1. Step")
        mock_learner.extract.return_value = learned_proc

        initial_count = len(store.list_all())
        # Use multi_stage=False for simpler execution path
        manager.run("Test task", multi_stage=False)
        final_count = len(store.list_all())

        assert final_count == initial_count + 1


class TestMemFlowManagerUtils:
    """Tests for MemFlowManager utility methods via public API."""

    def test_add_detects_procedural_keywords(self, fake_llm):
        """Test that add() detects procedural keywords and triggers extraction."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # When input contains procedural keywords, extraction should be attempted
        result = manager.add(messages="how to deploy step by step")

        # Verify extraction was attempted (result should have 'results' key)
        assert "results" in result

    def test_add_skips_non_procedural_content(self, fake_llm):
        """Test that add() skips content classified as non-procedural."""
        manager = MemFlowManager(llm=fake_llm, use_env=False)

        # Set LLM to classify as "none" (non-procedural)
        fake_llm.set_response('{"type": "none"}')
        result = manager.add(messages="what is the capital of France")

        # Verify extraction was skipped
        assert result.get("skipped") == "classified as none"

    def test_classify_memory_type_via_chat(self, fake_llm):
        """Test memory type classification via chat() public API."""
        store = EmulatedStore()
        fake_llm.set_response('{"type": "procedural", "title": "Test", "content": "1. Step"}')
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        # Chat should trigger auto-learn for procedural content
        manager.chat("how to deploy step by step")

        # Verify LLM was called for classification (check generate_calls)
        assert len(fake_llm.generate_calls) > 0

    def test_classify_memory_type_on_error(self, fake_llm):
        """Test classification falls back safely on error."""
        store = EmulatedStore()
        fake_llm.set_response("invalid json response")
        manager = MemFlowManager(llm=fake_llm, store=store, use_env=False)

        # Should not raise even with invalid LLM response
        result = manager.chat("test content")

        # Verify LLM was called
        assert len(fake_llm.generate_calls) > 0
        # chat() returns a dict with response key
        assert isinstance(result, dict)
        assert "response" in result


class TestLoadEnvFile:
    """Tests for _load_env_file() function including inline comment handling."""

    def test_inline_comment_stripped(self):
        """Test that inline comments are stripped from values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TEST_VAR=http://example.com # This is a comment\n",
                encoding="utf-8"
            )

            # Clear env var before test
            os.environ.pop("TEST_VAR", None)

            _load_env_file(str(env_file))

            assert os.environ.get("TEST_VAR") == "http://example.com"

    def test_quoted_value_preserves_hash(self):
        """Test that quoted values preserve # character."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                'QUOTED_HASH="value#with#hash"\n',
                encoding="utf-8"
            )

            os.environ.pop("QUOTED_HASH", None)
            _load_env_file(str(env_file))

            assert os.environ.get("QUOTED_HASH") == "value#with#hash"

    def test_single_quoted_value_preserves_hash(self):
        """Test that single-quoted values preserve # character."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "SINGLE_QUOTED_HASH='another#value'\n",
                encoding="utf-8"
            )

            os.environ.pop("SINGLE_QUOTED_HASH", None)
            _load_env_file(str(env_file))

            assert os.environ.get("SINGLE_QUOTED_HASH") == "another#value"

    def test_url_with_inline_comment(self):
        """Test that URL with inline comment is properly stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "MEMFLOW_EMBEDDING_API_BASE=http://10.78.59.136:8001/v1 # Required\n",
                encoding="utf-8"
            )

            os.environ.pop("MEMFLOW_EMBEDDING_API_BASE", None)
            _load_env_file(str(env_file))

            assert os.environ.get("MEMFLOW_EMBEDDING_API_BASE") == "http://10.78.59.136:8001/v1"

    def test_full_line_comment_ignored(self):
        """Test that full line comments are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "# This is a comment\n"
                "VALID_VAR=value\n"
                "# Another comment\n",
                encoding="utf-8"
            )

            os.environ.pop("VALID_VAR", None)
            _load_env_file(str(env_file))

            assert os.environ.get("VALID_VAR") == "value"

    def test_value_with_space_before_comment(self):
        """Test that value with space before comment is stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "TEST_VAR=test_value   # comment after spaces\n",
                encoding="utf-8"
            )

            os.environ.pop("TEST_VAR", None)
            _load_env_file(str(env_file))

            assert os.environ.get("TEST_VAR") == "test_value"

    def test_nonexistent_file_no_error(self):
        """Test that nonexistent .env file does not raise error."""
        # Should not raise any exception
        _load_env_file("/nonexistent/path/.env")

        # No exception means test passed
        assert True
