"""DRaFT-K reward trainer for Krea 2 (FedorAiToolkit).

Stage-2 process: loads the Krea2 base (quantized per config), attaches the
SFT-trained LoRA / LoKr network (resume via ``network.pretrained_lora_path``
or the process's own save folder), then optimizes the adapter directly on
differentiable rewards (face identity + body geometry) computed on images
sampled from the model during training.

The differentiable sampling loop is ported from
https://github.com/KONAKONA666/krea-2 (Apache-2.0) ``objectives.draft_sample_images``
onto ai-toolkit's Krea2 pipeline helpers: no grad for the first
``steps - draft_k`` denoising steps, gradient through the last K steps and a
checkpointed VAE decode, plus optional DRaFT-LV variance-reduction samples.

Config (all under the process dict):

  draft:
    steps: 12               # sampling steps per reward update
    draft_k: 1              # grad flows through the last K steps
    guidance_scale: 4.5     # ai-toolkit-style CFG (internally uses scale - 1)
    width: 512
    height: 512
    lv_samples: 0           # extra DRaFT-LV re-noised samples
    high_noise_shift: 0.5   # additive mu shift toward noisier states
    seed: 42                # per-step generator seed = seed + step
    checkpoint_vae: true    # activation-checkpoint the VAE decode
    train_modules: qkvo     # qkvo = only attention wq/wk/wv/wo adapter
                            # tensors receive optimizer updates; all = every
                            # adapter tensor (LoRA and LoKr alike)
    save_images_every: 10   # dump reward images to <save>/draft_step_images
    prompts:                # explicit prompt list, or...
      - "GB playing chess in the park"
    prompts_path: null      # ...a txt file (one prompt per line), or fall
                            # back to dataset captions next to the reward
                            # reference images
    reward:
      reference_images: "path/to/reference/images"
      face_weight: 1.0
      body_weight: 0.5
      face: {}              # extra FaceSimilarityReward kwargs
      body: {}              # extra BodyGeometryReward kwargs
"""

import glob
import os
import random
from collections import OrderedDict
from typing import List, Union

import torch
from torch.utils.checkpoint import checkpoint

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from extensions_built_in.diffusion_models.krea2.src.pipeline import (
    pad_text_features,
    predict_velocity,
    timesteps as krea2_timesteps,
)
from toolkit.basic import flush
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds
from toolkit.rewards import build_reward_from_config
from toolkit.accelerator import unwrap_model
from toolkit.print import print_acc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
QKVO_TAILS = {"wq", "wk", "wv", "wo"}


def _is_qkvo_module(lora_name: str) -> bool:
    """True when the adapter wraps an attention q/k/v/o projection.

    lora module names join the original module path with ``_`` or ``$$``
    (e.g. ``lora_transformer$$blocks$$0$$attn$$wq``). Matching the trailing
    projection name covers both DiT-block and text-fusion attention; SwiGLU
    (gate/up/down), attention gates and modulation stay frozen, mirroring the
    source repo's QKVO-only DRaFT restriction.
    """
    tokens = lora_name.replace("$$", "_").split("_")
    return len(tokens) > 0 and tokens[-1] in QKVO_TAILS


class Krea2DraftTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        if (self.model_config.arch or "").lower() != "krea2":
            raise ValueError(
                f"krea2_draft_trainer requires model.arch: krea2, got "
                f"{self.model_config.arch!r}"
            )

        draft = self.get_conf("draft", {}) or {}
        self.draft_steps = int(draft.get("steps", 12))
        self.draft_k = int(draft.get("draft_k", 1))
        self.draft_guidance_scale = float(draft.get("guidance_scale", 4.5))
        self.draft_width = int(draft.get("width", 512))
        self.draft_height = int(draft.get("height", 512))
        self.draft_lv_samples = int(draft.get("lv_samples", 0))
        self.draft_high_noise_shift = float(draft.get("high_noise_shift", 0.5))
        self.draft_seed = int(draft.get("seed", 42))
        self.draft_checkpoint_vae = bool(draft.get("checkpoint_vae", True))
        self.draft_train_modules = str(draft.get("train_modules", "qkvo")).lower()
        if self.draft_train_modules not in {"qkvo", "all"}:
            raise ValueError("draft.train_modules must be 'qkvo' or 'all'")
        self.draft_save_images_every = int(draft.get("save_images_every", 10))
        self._draft_prompts_conf = draft.get("prompts", None)
        self._draft_prompts_path = draft.get("prompts_path", None)
        self._draft_reward_conf = draft.get("reward", {}) or {}
        if self.draft_k < 1 or self.draft_k > self.draft_steps:
            raise ValueError("draft.draft_k must be in [1, draft.steps]")

        self.reward_fn = None
        self.draft_prompts: List[str] = []
        self._draft_embeds: List[torch.Tensor] = []
        self._prompt_cursor = 0

        # all prompts are pre-encoded in hook_before_train_loop, so the text
        # encoder (Qwen3-VL, ~8 GB quantized) never needs to be resident
        # during the reward loop. The base train loop applies this preset
        # right before stepping, which would otherwise move the TE back to
        # the GPU and push the reward models into shared-memory spill.
        self.train_device_state_preset["text_encoder"]["device"] = "cpu"

    def run(self):
        self._release_sibling_processes()
        super().run()

    def _release_sibling_processes(self):
        """Free models loaded by earlier processes in the same job.

        When SFT + DRaFT run as sequential processes in one config, ai-toolkit
        keeps the finished stage's quantized base model resident. Loading a
        second copy overflows the GPU into Windows shared memory, which slows
        CUDA to a crawl instead of OOMing. Processes run sequentially, so the
        finished stage's model is safe to drop before we load ours.
        """
        job = getattr(self, "job", None)
        released = False
        for proc in getattr(job, "process", None) or []:
            if proc is self:
                continue
            for attr in ("sd", "network", "optimizer", "lr_scheduler", "ema",
                         "data_loader", "data_loader_reg", "adapter",
                         "embedding", "modules_being_trained", "params"):
                if getattr(proc, attr, None) is not None:
                    setattr(proc, attr, None)
                    released = True
        if released:
            # the global Accelerator keeps its own references to every model
            # passed through accelerator.prepare() (stage 1's unet/vae/network
            # live in accelerator._models); free_memory() drops them all and
            # empties the CUDA cache. Nothing of ours is prepared yet.
            try:
                self.accelerator.free_memory()
            except Exception as e:
                print_acc(f"draft: accelerator.free_memory() failed: {e}")
            # gc first so dropped models actually free their CUDA blocks,
            # then release the now-empty cache back to the driver
            flush(garbage_collect=True)
            flush(garbage_collect=False)
            allocated = torch.cuda.memory_allocated() / 1e9
            print_acc(
                "draft: released prior process model(s) before loading the "
                f"DRaFT base ({allocated:.2f} GB still allocated)"
            )

    # ------------------------------------------------------------------
    # QKVO / all parameter selection (LoRA and LoKr)
    # ------------------------------------------------------------------
    def hook_add_extra_train_params(self, params):
        params = super().hook_add_extra_train_params(params)
        if self.network is None or self.draft_train_modules != "qkvo":
            return params

        network = unwrap_model(self.network)
        if hasattr(network, "get_all_modules"):
            modules = list(network.get_all_modules())
        else:
            modules = list(getattr(network, "unet_loras", [])) + list(
                getattr(network, "text_encoder_loras", [])
            )

        keep_ids = set()
        frozen = kept = 0
        for module in modules:
            name = getattr(module, "lora_name", "")
            if _is_qkvo_module(name):
                kept += 1
                for p in module.parameters():
                    keep_ids.add(id(p))
            else:
                frozen += 1
                for p in module.parameters():
                    p.requires_grad_(False)

        filtered = []
        dropped_params = 0
        for group in params:
            if isinstance(group, dict):
                group_params = [p for p in group["params"] if id(p) in keep_ids]
                dropped_params += len(group["params"]) - len(group_params)
                if group_params:
                    new_group = dict(group)
                    new_group["params"] = group_params
                    filtered.append(new_group)
            else:
                if id(group) in keep_ids:
                    filtered.append(group)
                else:
                    dropped_params += 1
        print_acc(
            f"draft train_modules=qkvo: training {kept} attention adapter "
            f"module(s), froze {frozen} module(s) ({dropped_params} tensors "
            "excluded from the optimizer)"
        )
        if not filtered:
            raise RuntimeError("qkvo filter removed every trainable parameter")
        return filtered

    # ------------------------------------------------------------------
    # Prompt + reward setup
    # ------------------------------------------------------------------
    def _resolve_draft_prompts(self) -> List[str]:
        prompts: List[str] = []
        if self._draft_prompts_conf:
            prompts = [str(p) for p in self._draft_prompts_conf]
        elif self._draft_prompts_path and os.path.exists(self._draft_prompts_path):
            with open(self._draft_prompts_path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # fall back to dataset captions living next to the reference images
            ref = self._draft_reward_conf.get("reference_images", None)
            if ref and os.path.isdir(str(ref)):
                for txt in sorted(glob.glob(os.path.join(str(ref), "*.txt"))):
                    with open(txt, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                    if caption:
                        prompts.append(caption)
        if not prompts:
            raise ValueError(
                "no DRaFT prompts: set draft.prompts, draft.prompts_path, or put "
                ".txt captions next to draft.reward.reference_images"
            )
        if self.trigger_word is not None:
            prompts = [p.replace("[trigger]", self.trigger_word) for p in prompts]
        return prompts

    def hook_before_train_loop(self):
        # caches self.unconditional_embeds (train.unconditional_prompt) and
        # handles the usual vae / noise-scheduler device shuffling
        super().hook_before_train_loop()

        self.draft_prompts = self._resolve_draft_prompts()
        print_acc(f"draft: {len(self.draft_prompts)} reward prompt(s)")

        # encode every DRaFT prompt once; each entry is a (L, F) cpu tensor in
        # the krea2 flattened stacked-layer format (see pad_text_features)
        self._draft_embeds = []
        with torch.no_grad():
            self.sd.text_encoder_to(self.device_torch)
            for prompt in self.draft_prompts:
                embeds: PromptEmbeds = self.sd.encode_prompt([prompt])
                self._draft_embeds.append(
                    embeds.text_embeds[0].detach().to("cpu")
                )
            # blank/negative embeds were cached by super(); free the encoder
            self.sd.text_encoder_to("cpu")
        flush()

        print_acc("draft: building reward model(s)")
        self.reward_fn = build_reward_from_config(
            self._draft_reward_conf, device=self.device_torch
        )

        # generated latents must decode with grad; keep the VAE resident
        self.sd.vae.to(self.device_torch)
        self.sd.vae.eval()
        self.sd.vae.requires_grad_(False)

        # shuffle prompts deterministically so short lists don't always pair
        # the same prompts into a batch
        rng = random.Random(self.draft_seed)
        order = list(range(len(self.draft_prompts)))
        rng.shuffle(order)
        self.draft_prompts = [self.draft_prompts[i] for i in order]
        self._draft_embeds = [self._draft_embeds[i] for i in order]

    # ------------------------------------------------------------------
    # Differentiable sampling (ported draft_sample_images)
    # ------------------------------------------------------------------
    def _schedule(self, latent_h: int, latent_w: int) -> List[float]:
        model = unwrap_model(self.sd.unet)
        patch = model.config.patch
        align = self.sd.vae_scale_factor * patch  # 16
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        seq_len = (latent_h // patch) * (latent_w // patch)
        mkw = self.model_config.model_kwargs
        y1 = float(mkw.get("schedule_y1", 0.5))
        y2 = float(mkw.get("schedule_y2", 1.15))
        mu = mkw.get("schedule_mu", None)
        if mu is None:
            slope = (y2 - y1) / (x2 - x1)
            mu = slope * seq_len + (y1 - slope * x1)
        # bias the deterministic integration grid toward noisier states while
        # preserving both endpoints (source repo's high_noise_schedule_mu)
        mu = float(mu) + self.draft_high_noise_shift
        return krea2_timesteps(seq_len, self.draft_steps, x1, x2, y1=y1, y2=y2, mu=mu)

    def _cfg_velocity(self, latents, t, cond, cond_mask, uncond, uncond_mask, guidance):
        model = unwrap_model(self.sd.unet)
        dtype = self.sd.torch_dtype
        v_cond = predict_velocity(model, latents.to(dtype), t, cond, cond_mask)
        if guidance <= 0 or uncond is None:
            return v_cond
        with torch.no_grad():
            v_uncond = predict_velocity(model, latents.to(dtype), t, uncond, uncond_mask)
        return v_cond + guidance * (v_cond - v_uncond.detach())

    def _decode_latents_checkpointed(self, latents: torch.Tensor) -> torch.Tensor:
        def run(z):
            return self.sd.decode_latents(
                z.to(self.sd.torch_dtype),
                device=self.device_torch,
                dtype=self.sd.torch_dtype,
            )

        if self.draft_checkpoint_vae:
            return checkpoint(
                run, latents, use_reentrant=False, preserve_rng_state=False
            )
        return run(latents)

    def draft_sample_images(
        self, embeds_list: List[torch.Tensor], seed: int
    ) -> torch.Tensor:
        """Sample images with grad through the last K steps + VAE decode.

        Returns (B * (1 + lv_samples), 3, H, W) images in [-1, 1].
        """
        device = self.device_torch
        dtype = self.sd.torch_dtype
        b = len(embeds_list)
        latent_h = self.draft_height // self.sd.vae_scale_factor
        latent_w = self.draft_width // self.sd.vae_scale_factor

        gen = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            b, 16, latent_h, latent_w, device=device, dtype=torch.float32, generator=gen
        )

        cond, cond_mask = pad_text_features(embeds_list, device, dtype)
        # ai-toolkit-style CFG scale; krea2's internal multiplier is scale - 1
        guidance = max(0.0, self.draft_guidance_scale - 1.0)
        uncond = uncond_mask = None
        if guidance > 0 and self.unconditional_embeds is not None:
            un = self.unconditional_embeds.text_embeds
            un = list(un) * b if len(un) == 1 else list(un)
            uncond, uncond_mask = pad_text_features(un[:b], device, dtype)

        ts = self._schedule(latent_h, latent_w)
        grad_start = max(0, self.draft_steps - self.draft_k)

        for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
            t = torch.full((b,), tcurr, dtype=dtype, device=device)
            delta = tprev - tcurr
            if i < grad_start:
                with torch.no_grad():
                    v = self._cfg_velocity(
                        latents, t, cond, cond_mask, uncond, uncond_mask, guidance
                    )
                latents = (latents + delta * v.to(torch.float32)).detach()
            else:
                v = self._cfg_velocity(
                    latents, t, cond, cond_mask, uncond, uncond_mask, guidance
                )
                latents = latents + delta * v.to(torch.float32)

        outputs = [latents]
        if self.draft_lv_samples:
            # DRaFT-LV: re-noise the (detached) sample at the last grid time and
            # take single-step denoised estimates for variance reduction
            last_t = float(ts[-2])
            t = torch.full((b,), last_t, dtype=dtype, device=device)
            for _ in range(self.draft_lv_samples):
                noise = torch.randn(
                    latents.shape, device=device, dtype=torch.float32, generator=gen
                )
                noised = last_t * noise + (1.0 - last_t) * latents.detach()
                v = self._cfg_velocity(
                    noised, t, cond, cond_mask, uncond, uncond_mask, guidance
                )
                outputs.append(noised - last_t * v.to(torch.float32))

        images = []
        for z in outputs:
            images.append(self._decode_latents_checkpointed(z).clamp(-1, 1))
        return torch.cat(images, dim=0)

    # ------------------------------------------------------------------
    # Reward loss (ported reward_loss)
    # ------------------------------------------------------------------
    def _reward_loss(self, images: torch.Tensor, prompts: List[str]):
        values = []
        for image, prompt in zip(images, prompts):
            value = self.reward_fn(image.unsqueeze(0).float(), prompt)
            value = value.to(device=image.device, dtype=torch.float32).mean()
            if not torch.isfinite(value):
                raise ValueError(f"reward returned a non-finite value for {prompt!r}")
            if not value.requires_grad:
                raise ValueError(
                    "reward output is not connected to the generated image; "
                    "DRaFT-K requires a differentiable reward"
                )
            values.append(value)
        rewards = torch.stack(values)
        return -rewards.mean(), rewards.detach()

    def _save_step_images(self, images: torch.Tensor, prompts: List[str]):
        import time as _time

        from torchvision.transforms import functional as TF

        sample_dir = os.path.join(self.save_root, "draft_step_images")
        os.makedirs(sample_dir, exist_ok=True)
        # mirror into the job's samples folder using ai-toolkit's
        # {ms}__{step:09d}_{idx} naming so the UI sample view picks them up
        ui_sample_dir = None
        if getattr(self, "is_ui_trainer", False):
            ui_sample_dir = os.path.join(self.save_root, "samples")
            os.makedirs(ui_sample_dir, exist_ok=True)
        ms_now = int(_time.time() * 1000)
        images_cpu = ((images.detach().float().clamp(-1, 1).cpu() + 1.0) * 0.5).clamp(0, 1)
        for idx, image in enumerate(images_cpu):
            pil = TF.to_pil_image(image)
            stem = f"step_{self.step_num:06d}_{idx:02d}"
            pil.save(os.path.join(sample_dir, f"{stem}.jpg"))
            with open(os.path.join(sample_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                f.write((prompts[idx] if idx < len(prompts) else "") + "\n")
            if ui_sample_dir is not None:
                pil.save(
                    os.path.join(
                        ui_sample_dir,
                        f"{ms_now}__{str(self.step_num).zfill(9)}_{idx}.jpg",
                    )
                )

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def hook_train_loop(
        self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO], None] = None
    ):
        # the DRaFT stage generates its own data -- dataset batches (if any
        # were configured) are intentionally ignored
        self.optimizer.zero_grad()

        network = unwrap_model(self.network) if self.network is not None else None
        if network is not None:
            network.is_active = True
            network.multiplier = 1.0
            network._update_torch_multiplier()

        bsz = max(1, self.train_config.batch_size)
        n = len(self.draft_prompts)
        idxs = [(self._prompt_cursor + i) % n for i in range(bsz)]
        self._prompt_cursor = (self._prompt_cursor + bsz) % n
        prompts = [self.draft_prompts[i] for i in idxs]
        embeds = [self._draft_embeds[i] for i in idxs]

        with self.timer("draft_sample"):
            images = self.draft_sample_images(
                embeds, seed=self.draft_seed + self.step_num
            )
        # DRaFT-LV extras reuse the same prompt list
        reps = images.shape[0] // len(prompts)
        full_prompts = prompts * reps

        with self.timer("draft_reward"):
            loss, rewards = self._reward_loss(images, full_prompts)

        with self.timer("draft_backward"):
            self.accelerator.backward(loss)

        if not self.is_grad_accumulation_step:
            if self.train_config.optimizer != "adafactor":
                if len(self.params) > 0 and isinstance(self.params[0], dict):
                    for group in self.params:
                        self.accelerator.clip_grad_norm_(
                            group["params"], self.train_config.max_grad_norm
                        )
                else:
                    self.accelerator.clip_grad_norm_(
                        self.params, self.train_config.max_grad_norm
                    )
            with self.timer("optimizer_step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                with self.timer("ema_update"):
                    self.ema.update()

        with self.timer("scheduler_step"):
            self.lr_scheduler.step()

        if (
            self.draft_save_images_every > 0
            and self.step_num % self.draft_save_images_every == 0
        ):
            self._save_step_images(images, full_prompts)

        loss_dict = OrderedDict(
            {
                "loss": loss.item(),
                "reward": rewards.mean().item(),
            }
        )
        components = getattr(self.reward_fn, "last_components", None)
        if components:
            for key, value in components.items():
                loss_dict[f"rw_{key}"] = value

        del images
        flush()

        self.end_of_training_loop()
        return loss_dict
