from pathlib import Path
import pytest
from src.training.config import load_config, WalkForwardConfig


def test_load_config_parses_walk_forward(tmp_path):
    yaml_text = """
training:
  epochs: 5
walk_forward:
  test_years: [2018, 2019]
  val_months: 2
seeds: [42]
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_text)
    config = load_config(cfg_file)
    assert isinstance(config.walk_forward, WalkForwardConfig)
    assert config.walk_forward.test_years == [2018, 2019]
    assert config.walk_forward.val_months == 2


def test_load_config_walk_forward_defaults(tmp_path):
    yaml_text = "training:\n  epochs: 5\nseeds: [42]\n"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_text)
    config = load_config(cfg_file)
    assert config.walk_forward.test_years == [2018, 2019, 2020, 2021, 2022, 2023]
    assert config.walk_forward.val_months == 2
