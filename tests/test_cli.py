import json

from mnemos.cli import main


def test_budget_command(capsys, tmp_path):
    main(["budget", "--config", "configs/tiny_pkm.yaml", "--out", str(tmp_path / "b.json")])
    out = json.loads(capsys.readouterr().out)
    assert out["params_from_memory"] > 0
    assert json.loads((tmp_path / "b.json").read_text())["param_overhead_x"] > 1


def test_list_command(capsys):
    main(["list"])
    text = capsys.readouterr().out
    for kind in ["product_key", "slot", "surprise", "knn"]:
        assert kind in text


def test_train_command_with_overrides(capsys, tmp_path):
    main(["train", "--config", "configs/tiny_slot.yaml", "--set",
          "train.steps=3", "train.batch_size=4", "train.eval_batches=1",
          "train.device=cpu", f"train.out_dir={tmp_path}/cli"])
    assert (tmp_path / "cli" / "final.json").exists()


def test_eval_command_roundtrips_a_checkpoint(capsys, tmp_path):
    main(["train", "--config", "configs/tiny_knn.yaml", "--set",
          "train.steps=2", "train.batch_size=4", "train.eval_batches=1",
          "train.device=cpu", f"train.out_dir={tmp_path}/ck"])
    capsys.readouterr()
    main(["eval", "--ckpt", f"{tmp_path}/ck/ckpt.pt", "--batches", "2", "--device", "cpu"])
    out = json.loads(capsys.readouterr().out)
    assert out["val_loss"] > 0 and 0.0 <= out["val_acc"] <= 1.0


def test_ablate_command_writes_a_full_report(tmp_path):
    main(["ablate", "--config", "configs/tiny_pkm.yaml", "--match", "params", "--set",
          "train.steps=2", "train.batch_size=4", "train.eval_batches=1",
          "train.device=cpu", f"train.out_dir={tmp_path}/ab",
          "--out", f"{tmp_path}/report.json"])
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["budget"]["params_from_memory"] > 0
    assert rep["read_ablation"]["memory_on"]["scored_tokens"] > 0
    assert "gain_over_params_matched" in rep
    assert "baseline_flops" not in rep, "--match params must not train the FLOP baseline"


def test_train_out_flag_is_honoured(tmp_path):
    dest = tmp_path / "elsewhere" / "run.json"
    main(["train", "--config", "configs/tiny_none.yaml", "--set",
          "train.steps=2", "train.batch_size=4", "train.eval_batches=1",
          "train.device=cpu", f"train.out_dir={tmp_path}/rd", "--out", str(dest)])
    assert dest.exists(), "--out was accepted and then ignored"
    assert (tmp_path / "rd" / "final.json").exists(), "run dir must stay self-contained"
    assert json.loads(dest.read_text())["config"] == "tiny-none"
