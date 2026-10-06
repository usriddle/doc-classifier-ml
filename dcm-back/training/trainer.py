"""
QLoRA 학습 - LoRA 부착, 학습 루프, 실시간 그래프, 중간 점검, 저장.

설계 메모
  - 베이스 모델은 llm_engine.load_base_model() 로 올립니다. 추론과 양자화·dtype 이 같습니다.
  - loss 는 정답(어시스턴트 출력) 토큰에만 겁니다. 긴 고정 프리픽스는 loss 에서 빠집니다.
  - 배치 크기는 1 로 고정하고 gradient accumulation 으로 묶습니다.
    그 대신 정답 구간의 로짓만 계산(logits_to_keep)해서, 어휘 수 x 전체 길이 크기의
    로짓을 만들지 않습니다. T4 메모리를 가장 크게 아끼는 부분입니다.
  - 어댑터는 비전·오디오 인코더를 빼고 언어 모델의 어텐션·MLP 선형층(q/k/v/o, gate/up/down)에만 붙입니다.
    (Gemma 4 E4B 의 층별 임베딩(PLE) 투영층은 건드리지 않습니다)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.config import settings
from app.services import llm_engine, prompts, runtime
from training.samples import IGNORE, Sample

KST = timezone(timedelta(hours=9))


# =============================================================================
# 설정
# =============================================================================
@dataclass
class TrainConfig:
    # --- LoRA ---
    lora_r: int = 16               # 어댑터 용량. 올리면 더 많이 배우지만 과적합·기존 능력 손상 위험도 커짐
    lora_alpha: int = 32           # 어댑터 영향력. 보통 r 의 2배
    lora_dropout: float = 0.1      # Gemma 4 v1 에서 검증 loss 가 절반 지점 뒤로 다시 올라 0.05 -> 0.1
    # --- 학습 ---
    learning_rate: float = 1e-4    # 가장 민감한 값. Gemma 4 v1(2e-4)은 높은 학습률 구간에서 loss 가 다시 튀어 1e-4 로
    epochs: float = 1.0
    grad_accum: int = 8            # 배치 1 x 8 = 실효 배치 8
    warmup_ratio: float = 0.1      # 0.05(약 13스텝)는 짧아 최고 학습률에서 흔들림 -> 0.1
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 4096         # 이보다 긴 샘플은 버림 (정답이 잘린 채 학습되지 않게)
    evals_per_epoch: int = 8       # 한 epoch 에 검증 loss 를 몇 번 잴지 (4번이면 최저점을 62스텝 간격으로만 봄)
    # 검증 loss 가 이 횟수만큼 연속으로 최저 기록을 못 깨면 학습을 멈춥니다. (0 이면 끝까지)
    # 최저점 체크포인트는 load_best_model_at_end 로 어차피 고르므로, 남은 스텝 시간을 아끼는 용도입니다.
    early_stopping_patience: int = 3
    logging_steps: int = 1
    save_total_limit: int = 2
    seed: int = 42
    full_kbit_prep: bool = False   # True 면 비양자화 층을 float32 로 (메모리 더 씀, 문제 해결용)
    # 중간 계산을 버렸다가 역전파 때 다시 계산해 메모리를 아낌. 끄면 약 20~30% 빨라지지만 메모리를 훨씬 더 씀.
    # Gemma 4 는 어휘가 26만 개·층별 임베딩(PLE)이 16bit 로 남아 메모리를 많이 씁니다. A100 에서만 끄기를 시도.
    gradient_checkpointing: bool = True
    # --- 중간 점검 ---
    spot_check_every: int = 2      # 검증 n 번마다 몇 건을 실제로 생성해 봄 (0 이면 안 함)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# LoRA 부착
# =============================================================================
_SKIP_PARTS = ("visual", "vision", "lm_head", "embed", "merger", "mm_projector", "audio", "per_layer")
# 표준 어텐션·MLP 투영층 이름. 이 이름이 있으면 여기에만 붙이고, 없는 구조면 언어 모델 선형층 전체에 붙입니다.
_STANDARD_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def pick_lora_targets(model) -> list[str]:
    """
    언어 모델 쪽 선형층 이름. (비전·오디오 인코더·출력층·임베딩·PLE 투영 제외)

    Gemma 4 E4B 는 q/k/v/o_proj, gate/up/down_proj 를 씁니다. 비전·오디오 인코더에도 같은 이름이
    있으므로 이름 끝만 보지 않고 경로에 vision/audio 가 들어간 것은 뺍니다.
    """
    linear = []
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls not in ("Linear", "Linear4bit", "Linear8bitLt"):
            continue
        lowered = name.lower()
        if any(part in lowered for part in _SKIP_PARTS):
            continue
        linear.append(name)
    standard = [n for n in linear if n.rsplit(".", 1)[-1] in _STANDARD_TARGETS]
    names = standard or linear
    if not names:
        raise RuntimeError("LoRA 를 붙일 선형층을 찾지 못했습니다.")
    return names


def summarize_targets(names: list[str]) -> dict[str, int]:
    """층 종류(이름 끝부분)별 개수 - 어디에 붙었는지 눈으로 확인하는 용도."""
    counts: dict[str, int] = {}
    for n in names:
        key = n.rsplit(".", 1)[-1]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def attach_lora(model, cfg: TrainConfig):
    """
    4bit 베이스에 LoRA 를 붙여 PeftModel 을 돌려줍니다.

    cfg.gradient_checkpointing 이 False 면 중간 계산을 저장해 두고 재계산하지 않습니다. (빠르지만 메모리를 더 씀)

    준비 방식 (cfg.full_kbit_prep)
      False(기본) : gradient checkpointing + 입력 grad 만 켭니다. 베이스 가중치의 dtype 은 그대로라
                    추론 때와 같고, T4 메모리를 아낍니다. LoRA 가중치만 float32 로 학습됩니다.
      True        : peft 의 prepare_model_for_kbit_training - 양자화 안 된 층(임베딩·정규화·출력층)을
                    전부 float32 로 올립니다. 더 안정적일 수 있지만 수 GB 를 더 씁니다.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.full_kbit_prep and runtime.use_4bit():
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = False   # 학습 중에는 캐시를 안 씀 (판정·생성은 호출할 때 직접 켬)

    targets = pick_lora_targets(model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    return peft_model, targets


def trainable_report(model) -> dict[str, object]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # 4bit 가중치는 두 값을 한 칸에 묶어 저장해 개수가 실제의 약 절반으로 세어집니다. 비율은 참고용.
    return {"학습되는 파라미터": trainable, "전체 파라미터(4bit 는 절반으로 셈)": total,
            "비율(%, 참고)": round(100 * trainable / max(total, 1), 3)}


# =============================================================================
# 데이터 / 손실
# =============================================================================
class SampleDataset:
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i].to_dict()


def make_collator(pad_id: int):
    import torch

    def collate(batch: list[dict]) -> dict:
        longest = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}

    return collate


def answer_only_loss(model, inputs):
    """
    정답 구간만 로짓을 계산해 cross-entropy 를 구합니다. (배치 1 전제)

    input:  [프리픽스 ...... 요청 구간 ...... | 정답 토큰들]
    위치 t 의 로짓은 t+1 번째 토큰을 예측하므로, 정답 첫 토큰 바로 앞부터 끝까지만 남깁니다.
    """
    import torch.nn.functional as F

    input_ids, labels = inputs["input_ids"], inputs["labels"]
    if input_ids.shape[0] != 1:
        raise ValueError("answer_only_loss 는 배치 크기 1 만 지원합니다.")
    first = int((labels[0] != IGNORE).nonzero()[0].item())
    keep = input_ids.shape[1] - first + 1
    # 폴백(전체 로짓 계산)은 두지 않습니다. 길이 x 어휘 수 크기의 로짓은 T4 에서 메모리가 터집니다.
    out = model(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        use_cache=False,
        logits_to_keep=keep,
    )
    logits = out.logits[:, -keep:, :]
    pred = logits[:, :-1, :].float()
    target = labels[:, first:]
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1), ignore_index=IGNORE)
    return loss, pred


def _trainer_class():
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        """정답 구간 로짓만 계산하는 Trainer."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # compute_loss 가 샘플별 평균 loss 를 돌려주므로, gradient accumulation 나눗셈은
            # Trainer 가 하도록 합니다. (True 로 두면 나눗셈을 건너뛰어 기울기·로그 loss 가 accum 배가 됨)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, pred = answer_only_loss(model, inputs)
            return (loss, {"logits": pred}) if return_outputs else loss

    return AnswerOnlyTrainer


# =============================================================================
# 실시간 모니터 (그래프 + 상태 + 중간 점검)
# =============================================================================
def _callback_base():
    from transformers import TrainerCallback

    return TrainerCallback


class LiveMonitor:
    """
    학습 중 노트북 출력 칸에 그래프와 상태를 제자리에서 갱신합니다.

      - 학습 loss(원값 + 이동평균), 검증 loss, 학습률
      - 최저 검증 loss 와 그때의 스텝
      - 경고: nan/inf, 검증 loss 연속 상승(과적합 신호), 학습 loss 정체
      - spot_records 가 있으면 검증 몇 번마다 실제 출력을 뽑아 표로 보여 줌
    """

    def __init__(self, spot_records=None, cands=None, spot_every: int = 2):
        self.history: dict[str, list] = {"train": [], "eval": [], "lr": [], "spot": []}
        self.warnings: list[str] = []
        self.spot_records = spot_records or []
        self.cands = cands or {}
        self.spot_every = spot_every
        self._evals = 0
        self._overfit_warned = False
        self._started = None
        self._fig_handle = None
        self._text_handle = None
        self._spot_handle = None

    # --- TrainerCallback 으로 변환 ---
    def callback(self):
        monitor = self
        Base = _callback_base()

        class _Cb(Base):
            def on_train_begin(self, args, state, control, **kw):
                monitor._started = time.time()
                monitor._render(state)

            def on_log(self, args, state, control, logs=None, **kw):
                monitor._on_log(state, logs or {})

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                monitor._on_eval(state, metrics or {})

            def on_train_end(self, args, state, control, **kw):
                monitor._render(state, final=True)

        return _Cb()

    # --- 기록 ---
    def _on_log(self, state, logs: dict) -> None:
        if "loss" in logs:
            value = float(logs["loss"])
            self.history["train"].append((state.global_step, value))
            if not math.isfinite(value):
                self._warn(f"step {state.global_step}: 학습 loss 가 {value} 입니다. 학습률을 낮추세요.")
        if "learning_rate" in logs:
            self.history["lr"].append((state.global_step, float(logs["learning_rate"])))
        self._render(state)

    def _on_eval(self, state, metrics: dict) -> None:
        if "eval_loss" not in metrics:
            return
        self.history["eval"].append((state.global_step, float(metrics["eval_loss"])))
        evals = [v for _, v in self.history["eval"]]
        if len(evals) >= 3 and evals[-1] > evals[-2] > evals[-3] and not self._overfit_warned:
            self._overfit_warned = True
            self._warn(
                f"step {state.global_step}: 검증 loss 가 세 번 연속 올랐습니다 - 과적합 신호입니다. "
                "가장 좋았던 체크포인트가 자동으로 선택됩니다. 다음 학습은 learning_rate 를 낮추거나 epoch 을 줄여 보세요."
            )
        self._evals += 1
        if self.spot_records and self.spot_every and self._evals % self.spot_every == 0:
            self._spot(state)
        self._render(state)

    def _spot(self, state) -> None:
        from training import evaluate

        try:
            items = evaluate.spot_check(self.spot_records, self.cands)
        except Exception as exc:   # 점검 실패가 학습을 멈추면 안 됨
            self._warn(f"중간 점검 실패(학습은 계속): {type(exc).__name__}: {exc}")
            return
        self.history["spot"].append({"step": state.global_step, "items": items})
        try:
            import pandas as pd
            from IPython.display import display

            df = pd.DataFrame(items)
            title = f"중간 점검 - step {state.global_step}"
            if self._spot_handle is None:
                self._spot_handle = display(df.style.set_caption(title), display_id=True)
            else:
                self._spot_handle.update(df.style.set_caption(title))
        except Exception:
            print(f"[중간 점검 step {state.global_step}]", json.dumps(items, ensure_ascii=False)[:1500])

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    # --- 그리기 ---
    def _status_text(self, state, final: bool = False) -> str:
        step, total = state.global_step, state.max_steps or 0
        elapsed = time.time() - (self._started or time.time())
        eta = (elapsed / step * (total - step)) if step and total else None
        train = self.history["train"]
        evals = self.history["eval"]
        best = min(evals, key=lambda x: x[1]) if evals else None
        lines = [
            f"{'학습 종료' if final else '학습 중'} | step {step}/{total} | epoch {state.epoch or 0:.2f} "
            f"| 경과 {elapsed / 60:.1f}분" + (f" | 남은 시간 약 {eta / 60:.1f}분" if eta else ""),
            f"학습 loss(최근) {train[-1][1]:.4f}" if train else "학습 loss -",
            (f"검증 loss(최근) {evals[-1][1]:.4f} | 최저 {best[1]:.4f} @ step {best[0]}" if best else "검증 loss -"),
        ]
        if len(train) >= 30:
            # 중앙값으로 비교합니다. (한두 스텝 튄 값 때문에 경고가 잘못 뜨지 않게)
            import statistics
            first = statistics.median(v for _, v in train[:10])
            last = statistics.median(v for _, v in train[-20:])
            if last > first * 0.9 and step > 30:
                self._warn("학습 loss 가 거의 줄지 않습니다. 데이터 형식이나 학습률(너무 낮음)을 확인하세요.")
        if self.warnings:
            lines.append("경고:")
            lines += [f"  - {w}" for w in self.warnings[-5:]]
        return "\n".join(lines)

    def _render(self, state, final: bool = False) -> None:
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display
        except ImportError:
            print(self._status_text(state, final))
            return

        # 그래프 글자는 영어로 둡니다. (Colab 기본 글꼴에 한글이 없어 네모로 깨짐)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
        train = self.history["train"]
        if train:
            xs, ys = zip(*train)
            ax1.plot(xs, ys, color="#9aa7b8", linewidth=0.8, label="train loss")
            if len(ys) >= 5:
                w = max(3, len(ys) // 15)
                smooth = [sum(ys[max(0, i - w + 1): i + 1]) / len(ys[max(0, i - w + 1): i + 1])
                          for i in range(len(ys))]
                ax1.plot(xs, smooth, color="#2c6fbb", linewidth=1.8, label="train loss (moving avg)")
        if self.history["eval"]:
            xs, ys = zip(*self.history["eval"])
            ax1.plot(xs, ys, "o-", color="#d9822b", linewidth=1.8, label="eval loss")
        ax1.set_xlabel("step")
        ax1.set_title("loss")
        ax1.grid(alpha=0.3)
        if train or self.history["eval"]:
            ax1.legend(loc="upper right", fontsize=8)
        if self.history["lr"]:
            xs, ys = zip(*self.history["lr"])
            ax2.plot(xs, ys, color="#5b8c5a")
        ax2.set_xlabel("step")
        ax2.set_title("learning rate")
        ax2.grid(alpha=0.3)
        fig.tight_layout()

        text = self._status_text(state, final)
        if self._fig_handle is None:
            self._text_handle = display({"text/plain": text}, raw=True, display_id=True)
            self._fig_handle = display(fig, display_id=True)
        else:
            self._text_handle.update({"text/plain": text}, raw=True)
            self._fig_handle.update(fig)
        plt.close(fig)

    def save_figure(self, path: str | Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        if self.history["train"]:
            ax.plot(*zip(*self.history["train"]), color="#9aa7b8", label="train loss")
        if self.history["eval"]:
            ax.plot(*zip(*self.history["eval"]), "o-", color="#d9822b", label="eval loss")
        ax.set_xlabel("step")
        if self.history["train"] or self.history["eval"]:
            ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


# =============================================================================
# 학습 실행
# =============================================================================
def build_trainer(model, tokenizer, train_samples, val_samples, cfg: TrainConfig,
                  checkpoint_dir: str | Path, monitor: LiveMonitor | None = None):
    from transformers import TrainingArguments

    steps_per_epoch = max(1, math.ceil(len(train_samples) / cfg.grad_accum))
    eval_steps = max(1, steps_per_epoch // max(cfg.evals_per_epoch, 1))
    bf16 = runtime.supports_bf16()
    kwargs = dict(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,       # 검증 loss 가 가장 낮았던 체크포인트로 끝남
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=not bf16 and runtime.is_cuda(),   # T4 는 bf16 이 없어 fp16
        optim="paged_adamw_8bit" if runtime.use_4bit() else "adamw_torch",
        gradient_checkpointing=False,      # attach_lora 에서 cfg.gradient_checkpointing 대로 켜거나 끔
        report_to="none",
        remove_unused_columns=False,
        prediction_loss_only=True,
        dataloader_num_workers=0,
        label_names=["labels"],
        seed=cfg.seed,
    )
    try:
        args = TrainingArguments(warmup_ratio=cfg.warmup_ratio, **kwargs)
    except TypeError:
        # 최신 transformers 는 warmup_ratio 대신 warmup_steps 에 0~1 비율을 받습니다.
        args = TrainingArguments(warmup_steps=cfg.warmup_ratio, **kwargs)
    callbacks = [monitor.callback()] if monitor else []
    if cfg.early_stopping_patience and cfg.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = _trainer_class()(
        model=model,
        args=args,
        train_dataset=SampleDataset(train_samples),
        eval_dataset=SampleDataset(val_samples),
        data_collator=make_collator(pad_id),
        callbacks=callbacks or None,
    )
    # 노트북 기본 진행 표시(표)는 그래프와 겹치므로 끕니다. 진행 상황은 LiveMonitor 가 보여 줌.
    try:
        from transformers.utils.notebook import NotebookProgressCallback

        trainer.remove_callback(NotebookProgressCallback)
    except Exception:
        pass
    return trainer, {"steps_per_epoch": steps_per_epoch, "eval_steps": eval_steps,
                     "total_steps": math.ceil(steps_per_epoch * cfg.epochs), "bf16": bf16,
                     "early_stopping_patience": cfg.early_stopping_patience}


def finish_training(trainer):
    """
    학습이 끝난 모델을 평가·저장용으로 정리합니다.

    fp16/bf16 학습 때 accelerate 가 모델 forward 를 autocast 로 감싸 둔 것을 풀어서,
    학습 후 평가가 학습 전 기준선·서버와 같은 계산 경로로 돌게 합니다.
    """
    model = trainer.model
    try:
        from accelerate.utils import extract_model_from_parallel

        model = extract_model_from_parallel(model, keep_fp32_wrapper=False)
    except Exception:
        original = model.__dict__.pop("_original_forward", None)
        if original is not None:
            model.forward = original
    model.eval()
    return model


def check_lora_dtype(model) -> list[str]:
    """학습되는 파라미터가 전부 float32 인지 (fp16 학습에서 GradScaler 오류를 막는 조건)."""
    import torch

    return [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]


def latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


# =============================================================================
# 저장 / 버전 관리
# =============================================================================
def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def save_run(model, run_dir: str | Path, info: dict) -> Path:
    """어댑터 + run_info.json 을 저장합니다. 서버는 run_info 로 학습 조건을 대조합니다."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(run_dir))
    base = {
        "created_at": now_kst(),
        "base_model": settings.LLM_MODEL,
        "prompt_fingerprint": prompts.prompt_fingerprint(),
        "enable_thinking": settings.LLM_ENABLE_THINKING,
        "load_4bit": runtime.use_4bit(),
        "compute_dtype": str(runtime.compute_dtype()).replace("torch.", ""),
    }
    base.update(info)
    (run_dir / "run_info.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return run_dir


def list_runs(adapters_dir: str | Path) -> list[dict]:
    """저장된 어댑터 버전들의 요약. (노트북 '버전 비교' 표)"""
    rows = []
    for info_file in sorted(Path(adapters_dir).glob("*/run_info.json")):
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = info.get("metrics_after") or {}
        rows.append({
            "run": info.get("run_name", info_file.parent.name),
            "만든 시각": info.get("created_at"),
            "판정": info.get("verdict"),
            "베이스 모델": info.get("base_model"),
            "학습 샘플": (info.get("sample_counts") or {}).get("train"),
            "r": (info.get("train_config") or {}).get("lora_r"),
            "lr": (info.get("train_config") or {}).get("learning_rate"),
            "epochs": (info.get("train_config") or {}).get("epochs"),
            "최저 검증 loss": info.get("best_eval_loss"),
            "location 정확": (m.get("location 정확 일치") or {}).get("value"),
            "location 지어냄": (m.get("location 지어냄 비율") or {}).get("value"),
            "content 유사도": (m.get("content 유사도") or {}).get("value"),
            "JSON 파싱": (m.get("JSON 파싱 성공률") or {}).get("value"),
            "의도 정확도": (m.get("의도 정확도") or {}).get("value"),
            "프롬프트 일치": info.get("prompt_fingerprint") == prompts.prompt_fingerprint(),
            "지금 모델과 같음": info.get("base_model") == settings.LLM_MODEL,
        })
    return rows
