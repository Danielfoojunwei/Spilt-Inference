"""Tests for privacy budget optimizer and device profiles."""

import pytest

from src.compiler.device_profiles import DEVICE_PROFILES, auto_detect_profile, get_profile
from src.compiler.privacy_budget import PrivacyBudgetOptimizer, PrivacyBudgetResult
from src.compiler.split_compiler import SplitCompiler


class TestDeviceProfiles:
    """Test device profile configuration."""

    def test_all_profiles_exist(self):
        """All standard profiles are defined."""
        expected = {"phone", "laptop", "workstation", "server", "server_tee"}
        assert set(DEVICE_PROFILES.keys()) == expected

    def test_phone_profile(self):
        """Phone profile has minimal resources."""
        p = get_profile("phone")
        assert p.num_client_layers == 1
        assert p.epsilon == 1.0  # Strong privacy
        assert p.lora_rank == 4
        assert p.max_client_ram_gb <= 2.0

    def test_server_tee_profile(self):
        """Server with TEE has no DP noise (infinite epsilon)."""
        p = get_profile("server_tee")
        assert p.has_tee is True
        assert p.epsilon == float("inf")
        assert p.num_client_layers == 0  # TEE handles everything

    def test_privacy_ordering(self):
        """Stronger devices get weaker DP (higher epsilon)."""
        phone = get_profile("phone")
        laptop = get_profile("laptop")
        workstation = get_profile("workstation")
        server = get_profile("server")

        assert phone.epsilon < laptop.epsilon < workstation.epsilon < server.epsilon

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown device type"):
            get_profile("refrigerator")

    def test_auto_detect_phone(self):
        """Low RAM → phone profile."""
        p = auto_detect_profile(1.0)
        assert p.name == "phone"

    def test_auto_detect_workstation(self):
        """Medium RAM → workstation profile."""
        p = auto_detect_profile(6.0)
        assert p.name == "workstation"

    def test_auto_detect_tee(self):
        """TEE → server_tee profile."""
        p = auto_detect_profile(32.0, has_tee=True)
        assert p.name == "server_tee"


class TestPrivacyBudgetOptimizer:
    """Test joint (epsilon, K, rank) optimization."""

    def test_optimize_phone(self):
        """Phone optimization produces valid result."""
        opt = PrivacyBudgetOptimizer(total_model_layers=16)
        result = opt.optimize(get_profile("phone"))

        assert result.epsilon == 1.0
        assert result.num_client_layers >= 0
        assert result.lora_rank == 4
        assert result.estimated_throughput_tps > 0
        assert 0 <= result.privacy_score <= 1

    def test_optimize_workstation(self):
        """Workstation has weaker privacy than phone (higher epsilon)."""
        opt = PrivacyBudgetOptimizer(total_model_layers=16)

        phone_result = opt.optimize(get_profile("phone"))
        ws_result = opt.optimize(get_profile("workstation"))

        # Workstation has weaker DP privacy (higher epsilon)
        assert ws_result.privacy_score < phone_result.privacy_score

        # Workstation has better quality (lower quality loss due to less noise + higher rank)
        assert ws_result.estimated_quality_loss_pct < phone_result.estimated_quality_loss_pct

    def test_privacy_score_range(self):
        """Privacy score is always in [0, 1]."""
        opt = PrivacyBudgetOptimizer(total_model_layers=32)

        for name, profile in DEVICE_PROFILES.items():
            result = opt.optimize(profile)
            assert 0 <= result.privacy_score <= 1, f"Profile {name}: score={result.privacy_score}"

    def test_throughput_positive(self):
        """Throughput is always positive."""
        opt = PrivacyBudgetOptimizer(total_model_layers=32)

        for name, profile in DEVICE_PROFILES.items():
            result = opt.optimize(profile)
            assert result.estimated_throughput_tps > 0, f"Profile {name}"

    def test_generate_config(self):
        """Generate full config from profile."""
        opt = PrivacyBudgetOptimizer(total_model_layers=16)
        profile = get_profile("laptop")

        config = opt.generate_config(profile, model_id="test-model")

        assert config.model_id == "test-model"
        assert config.total_layers == 16
        assert config.num_client_layers + config.num_server_layers == 16
        assert config.privacy.epsilon == profile.epsilon
        assert config.lora_rank == profile.lora_rank
        assert config.adapter_only_encryption is True
        assert config.parallel_adapter is True
        assert config.fused_round_trip is True


class TestSplitCompiler:
    """Test split inference compiler."""

    def test_compile_laptop(self):
        """Compile for laptop profile."""
        compiler = SplitCompiler(
            model_id="test-model",
            total_layers=16,
            hidden_size=2048,
        )
        report = compiler.compile("laptop")

        assert report.config.model_id == "test-model"
        assert report.profile.name == "laptop"
        assert report.budget.estimated_throughput_tps > 0
        assert isinstance(report.warnings, list)

    def test_compile_all_profiles(self):
        """All profiles compile without error."""
        compiler = SplitCompiler(total_layers=16, hidden_size=2048)

        for device_type in DEVICE_PROFILES:
            report = compiler.compile(device_type)
            assert report.config is not None
            assert report.budget is not None

    def test_compile_auto(self):
        """Auto-detect compilation works."""
        compiler = SplitCompiler(total_layers=16, hidden_size=2048)
        report = compiler.compile_auto(available_ram_gb=4.0)

        assert report.profile is not None
        assert report.config is not None
