import importlib
import argparse
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
# from diffusers import DDPMScheduler # Not used for SmolVLM
from library import deepspeed_utils # model_util might be less relevant
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig # Added BitsAndBytesConfig

import library.train_util as train_util
# from library.train_util import DreamBoothDataset # Replaced by SmolVLMDataset
import library.config_util as config_util
# from library.config_util import ( # May not need these if dataset config is simpler
#     ConfigSanitizer,
#     BlueprintGenerator,
# )
import library.huggingface_util as huggingface_util
# import library.custom_train_functions as custom_train_functions # Most are SD specific
from library.utils import setup_logging, add_logging_arguments
from accelerate.utils import gather_object, gather

# Import SmolVLM specific utilities
from library.smolvlm_train_util import SmolVLMDataset #, calculate_smolvlm_bucket_resolution, parse_toml_caption, generate_qas_from_patterns


setup_logging()
import logging

logger = logging.getLogger(__name__)


class SmolVLMTrainer: # Renamed from NetworkTrainer
    def __init__(self):
        # self.vae_scale_factor = 0.18215 # Not applicable to SmolVLM
        self.is_sdxl = False # Not applicable, but keep if some shared utils use it

    # TODO: Review if this log generation is still fully applicable
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        # train_dataset_group.verify_bucket_reso_steps(64) # May not be needed if dataset handles its own validation
        pass

    def load_target_model(self, args, accelerator, weight_dtype): # Added weight_dtype
        # TODO: Add BitsAndBytesConfig for quantization if specified in args
        # For now, standard loading
        model = AutoModelForVision2Seq.from_pretrained(
            args.smolvlm_model_id,
            torch_dtype=weight_dtype, # Apply dtype during loading
            # low_cpu_mem_usage=True # Option for large models
        )
        return model

    def load_processor(self, args): # Renamed from load_tokenizer
        processor = AutoProcessor.from_pretrained(args.smolvlm_model_id)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # Ensure <image> token is handled if not already part of vocab
        # This was done in SmolVLMDataset example, good to ensure processor's tokenizer is ready
        if "<image>" not in processor.tokenizer.get_vocab():
            # Model will be resized later if new tokens added here affect its vocab size
            current_vocab_size = len(processor.tokenizer)
            processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
            if len(processor.tokenizer) > current_vocab_size:
                logger.info(f"<image> token added to processor's tokenizer. Old vocab size: {current_vocab_size}, New: {len(processor.tokenizer)}")


        return processor

    # Removed SD specific methods:
    # is_text_encoder_outputs_cached
    # is_train_text_encoder
    # cache_text_encoder_outputs_if_needed
    # get_text_cond
    # call_unet
    # all_reduce_network (Accelerator handles gradient reduction)
    
    # sample_images will be removed/re-implemented later for SmolVLM
    def sample_images(self, accelerator, args, epoch, global_step, device, model_processor_tuple, example_tuple=None):
        logger.info("SmolVLM sampling/evaluation not yet implemented.")
        pass

    def train(self, args):
        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            '''
            # Code to force sync seeds. Conduct testing with normal random seeds on separate processes.
            seed = [random.randint(0, 2**32)]
            seed = gather_object(seed)
            args.seed = seed[0]
            '''
            args.seed = random.randint(0, 2**32)
            logger.info(f"Seed for this run is: {args.seed}.")
        set_seed(args.seed)

        # Load processor (tokenizer + image_processor)
        processor = self.load_processor(args)
        
        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)

        # モデルを読み込む
        model = self.load_target_model(args, accelerator, weight_dtype)
        
        # Resize token embeddings if new tokens were added to tokenizer by processor
        if len(processor.tokenizer) > model.config.vocab_size:
            logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(processor.tokenizer)}.")
            model.resize_token_embeddings(len(processor.tokenizer))
            # After resizing, the model should be moved to device again by accelerator.prepare
            # or ensure it's on the correct device before optimizer creation.


        # データセットを準備する
        is_dreambooth_mode = hasattr(args, 'is_dreambooth_mode') and args.is_dreambooth_mode
        if not hasattr(args, 'is_dreambooth_mode') and args.reg_data_dir is not None: # Assuming reg_data_dir implies Dreambooth
            logger.info("`reg_data_dir` is provided, and `is_dreambooth_mode` is not set. Enabling Dreambooth mode for dataset.")
            is_dreambooth_mode = True
            
        smolvlm_max_reso = args.smolvlm_longest_edge_n * 384
        dataset_repeats = args.dataset_repeats if hasattr(args, 'dataset_repeats') else 1

        train_dataset = SmolVLMDataset(
            image_data_dir=args.train_data_dir,
            toml_captions_dir=args.toml_captions_dir,
            processor=processor,
            is_dreambooth_mode=is_dreambooth_mode, 
            smolvlm_longest_edge_n=args.smolvlm_longest_edge_n,
            batch_size=args.train_batch_size,
            tokenizer=processor.tokenizer, # Pass the tokenizer part of the processor
            max_token_length=args.max_token_length if args.max_token_length else processor.tokenizer.model_max_length,
            resolution=(smolvlm_max_reso, smolvlm_max_reso), 
            enable_bucket=True, 
            min_bucket_reso=args.min_bucket_reso if hasattr(args, 'min_bucket_reso') else 256,
            max_bucket_reso=args.max_bucket_reso if hasattr(args, 'max_bucket_reso') else smolvlm_max_reso,
            bucket_reso_steps=args.bucket_reso_steps if hasattr(args, 'bucket_reso_steps') else 64,
            debug_dataset=args.debug_dataset,
            num_repeats=dataset_repeats,
            num_qas_per_image_min=args.num_qas_per_image_min if hasattr(args, 'num_qas_per_image_min') else 1,
            num_qas_per_image_max=args.num_qas_per_image_max if hasattr(args, 'num_qas_per_image_max') else 3,
            color_aug=args.color_aug,
            flip_aug=args.flip_aug,
            random_crop=args.random_crop,
            shuffle_caption=args.shuffle_caption, # Less relevant but part of BaseDataset args
        )
        
        # Simplified dataset handling for SmolVLM as it doesn't use DatasetGroup in the same complex way as SD scripts
        # train_dataset_group = train_util.DatasetGroup([train_dataset]) # This might be overly complex now

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        
        if args.debug_dataset:
            logger.info("Dataset debugging enabled. SmolVLMDataset has internal logging.")
            # For a quick check:
            # if len(train_dataset) > 0:
            #     logger.info(f"First item from dataset: {train_dataset[0]}") # Might be large to log
            # else:
            #     logger.warning("Dataset is empty after initialization.")

        if len(train_dataset) == 0:
            logger.error("No data found by SmolVLMDataset. Please check image_data_dir and toml_captions_dir arguments.")
            return
            
        # `cache_latents` is not used for SmolVLM.
        # `assert_extra_args` might not be needed or needs adaptation.
        # self.assert_extra_args(args, train_dataset) # If adapted for single dataset

        # XFormers and memory efficient attention are typically handled by the model's config.
        # If an explicit call is needed and supported by the SmolVLM model:
        if args.xformers: 
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                model.enable_xformers_memory_efficient_attention()
                logger.info("Enabled XFormers memory efficient attention for the model.")
            # else:
            #     logger.warning("XFormers requested, but model does not support `enable_xformers_memory_efficient_attention`.")

        # Gradient checkpointing
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for the model.")

        # Model requires_grad_ setup (ensure all params are trainable for full fine-tune)
        model.requires_grad_(True)
        logger.info("Set model parameters to requires_grad=True for fine-tuning.")

        # 学習に必要なクラスを準備する
        accelerator.print("Prepare optimizer, data loader etc.")

        # For SmolVLM, trainable_params will be model.parameters()
        trainable_params = model.parameters() 
        # lr_descriptions might not be needed unless we have different LRs for parts of SmolVLM
        lr_descriptions = None 

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # dataloaderを準備する
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())

        # For SmolVLM, dataset items are dicts. Default collate should work if batch_size=1.
        # If batch_size > 1, a custom collator (like one using processor.tokenizer.pad) would be needed.
        # For now, assume batch_size=1 from DataLoader, and batching is handled by gradient_accumulation_steps.
        # Or, if dataset returns items that can be stacked by default collate_fn.
        # SmolVLMDataset returns individual processed items, so default collator is fine for batch_size=1.
        # The `batch_size` arg in DataLoader is how many items from dataset to group into one batch for model.
        # Let's use args.train_batch_size for the DataLoader.
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, # Use the single dataset instance
            batch_size=args.train_batch_size, # This is per-device batch size
            shuffle=True,
            # collate_fn=collator, # Default collator should work for dicts of tensors
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers if n_workers > 0 else False,
        )

        # 学習ステップ数を計算する
        # Original: num_of_steps = len(train_dataset_group)
        # For SmolVLM, it's len(train_dataset) // (args.train_batch_size * num_processes * grad_accum_steps) per epoch
        # Let Accelerate handle the dataloader length for num_update_steps_per_epoch.
        # len(train_dataloader) will give num batches per process per epoch.
        
        if args.max_train_epochs is not None:
            # This calculation needs to be after accelerator.prepare(train_dataloader)
            # to get the correct length of the sharded dataloader.
            # For now, we'll calculate it later.
            pass
        
        # lr schedulerを用意する
        # Placeholder for lr_scheduler, will be prepared with accelerator
        lr_scheduler = None

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16: # This moves the model to fp16. Accelerator handles mixed precision.
            assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16'"
            logger.info("Model will be cast to fp16 for full fp16 training (handled by Accelerator).")
            # model.to(weight_dtype) # Accelerator handles this
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", "full_bf16 requires mixed precision='bf16'"
            logger.info("Model will be cast to bf16 for full bf16 training (handled by Accelerator).")
            # model.to(weight_dtype) # Accelerator handles this

        # FP8 is experimental and support depends on model and hardware.
        if args.fp8_base:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0"
            assert args.mixed_precision != "no", "fp8_base requires mixed precision='fp16' or 'bf16'"
            logger.info("FP8 training for base model requested. Model will be cast to fp8 (handled by Accelerator).")
            # model.to(torch.float8_e4m3fn) # Accelerator might handle this if configured.

        # Prepare model, optimizer, dataloader, and lr_scheduler with accelerator
        # lr_scheduler is created inside accelerator.prepare if not provided.
        # We need to create it before if we want to use a specific one from train_util.get_scheduler_fix
        # Let's create optimizer first, then pass it to get_scheduler_fix, then to accelerator.prepare
        
        # Calculate num_train_steps for scheduler if not explicitly provided in args and max_train_epochs is set
        # This calculation is tricky before knowing the true length of the dataloader after sharding.
        # Accelerator's prepare(dataloader) will give the sharded one.
        # For now, if max_train_steps is None, scheduler might be infinite or based on epochs.
        # train_util.get_scheduler_fix handles this.
        
        # Prepare Optimizer without lr_scheduler first
        # training_model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        
        # Create LR scheduler after we know the dataloader length from accelerator
        # For now, pass optimizer directly. Accelerator will create a default scheduler if lr_scheduler is None.
        # Or, create scheduler after preparing dataloader (more accurate steps).
        # Let's stick to creating it before, assuming max_train_steps is the primary guide.
        
        # If max_train_steps is not set, calculate from epochs and dataloader length (pre-accelerator)
        # This is an estimate, as accelerator might change dataloader length.
        if args.max_train_steps is None and args.max_train_epochs is not None:
            estimated_num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            args.max_train_steps = args.max_train_epochs * estimated_num_update_steps_per_epoch
            logger.info(f"max_train_steps estimated to {args.max_train_steps} for {args.max_train_epochs} epochs.")

        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        training_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        # model is now potentially a new object (e.g. DDPStatedModel), use training_model
        
        # Update num_train_epochs if it was derived from max_train_steps and an estimated dataloader length
        if args.max_train_epochs is not None : # Re-calculate based on potentially sharded dataloader length
             num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
             if num_update_steps_per_epoch > 0 : # Avoid division by zero if dataloader is empty
                num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
             else:
                num_train_epochs = 0
             logger.info(f"Re-calculated num_train_epochs = {num_train_epochs} based on sharded dataloader length and max_train_steps.")
        else:
            num_train_epochs = 1 # Default if no epoch/step limit, should be controlled by max_train_steps


        if args.gradient_checkpointing: # Ensure it's enabled on the potentially wrapped model
            if hasattr(training_model, "gradient_checkpointing_enable"):
                 training_model.gradient_checkpointing_enable()
            elif hasattr(accelerator.unwrap_model(training_model), "gradient_checkpointing_enable"):
                 accelerator.unwrap_model(training_model).gradient_checkpointing_enable()


        # Save/load hooks need to operate on the unwrapped model type
        # The type check should be against the original model class, not the wrapped one.
        original_model_class = type(accelerator.unwrap_model(training_model))

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, m in enumerate(models):
                    if not isinstance(m, original_model_class): # Check against original model type
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i: # Check if weights list is long enough
                        weights.pop(i)
            
            train_state_file = os.path.join(output_dir, "train_state.json")
            logger.info(f"Save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None
        epoch_from_state = None

        def load_model_hook(models, input_dir):
            nonlocal steps_from_state, epoch_from_state
            remove_indices = []
            for i, m in enumerate(models):
                if not isinstance(m, original_model_class): # Check against original model type
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if models: # Check if models list is not empty before popping
                     models.pop(i)
            
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                epoch_from_state = data["current_epoch"]
                logger.info(f"Load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        train_util.resume_from_local_or_hf_if_specified(accelerator, args)
        
        # epoch数を計算する (re-calculate after accelerator.prepare)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if num_update_steps_per_epoch == 0 : num_update_steps_per_epoch = 1 # Prevent div by zero if dataloader is empty
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 0
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0) and num_train_epochs > 0:
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
        else:
            args.save_every_n_epochs = None # Ensure it's None if not applicable


        # 学習する
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("Running training...")
        logger.info(f"  Num examples = {len(train_dataset)}") # num_train_images from dataset
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
        # Metadata for saving
        # TODO: Adapt metadata for SmolVLM, remove SD specific fields
        model_version_str = f"SmolVLM_{args.smolvlm_model_id.replace('/','_')}" # Example
        metadata = {
            "ss_session_id": session_id,
            "ss_training_started_at": training_started_at,
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            # "ss_text_encoder_lr": args.text_encoder_lr, # Not applicable for SmolVLM single LR
            # "ss_unet_lr": args.unet_lr, # Not applicable
            "ss_num_train_images": len(train_dataset), # Adjusted
            # "ss_num_reg_images": train_dataset_group.num_reg_images, # Not applicable
            "ss_num_batches_per_epoch": num_update_steps_per_epoch, # Adjusted
            "ss_num_epochs": num_train_epochs, # Adjusted
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            # "ss_network_module": args.network_module, # TODO: Remove for SmolVLM
            # "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim # TODO: Remove for SmolVLM
            # "ss_network_alpha": args.network_alpha,  # some networks may not have alpha # TODO: Remove for SmolVLM
            # "ss_network_dropout": args.network_dropout,  # some networks may not have dropout # TODO: Remove for SmolVLM
            "ss_mixed_precision": args.mixed_precision,
            "ss_smolvlm_model_id": args.smolvlm_model_id,
            "ss_smolvlm_longest_edge_n": args.smolvlm_longest_edge_n,
            "ss_toml_captions_dir": args.toml_captions_dir,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version_str, # Adjusted
            # "ss_clip_skip": args.clip_skip, # Not applicable
            "ss_max_token_length": args.max_token_length if args.max_token_length else processor.tokenizer.model_max_length,
            # "ss_cache_latents": bool(args.cache_latents), # Not applicable
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            # "ss_prior_loss_weight": args.prior_loss_weight, # Not directly applicable
            # "ss_min_snr_gamma": args.min_snr_gamma, # SD specific
            # "ss_scale_weight_norms": args.scale_weight_norms, # LoRA specific
            # "ss_ip_noise_gamma": args.ip_noise_gamma, # SD specific
            # "ss_debiased_estimation": bool(args.debiased_estimation_loss), # SD specific
            # "ss_noise_offset_random_strength": args.noise_offset_random_strength, # SD specific
            # "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength, # SD specific
            # "ss_loss_type": args.loss_type, # SmolVLM uses model's internal loss
            # "ss_huber_schedule": args.huber_schedule, # Related to SD loss types
            # "ss_huber_c": args.huber_c, # Related to SD loss types
        }
        
        # Dataset metadata (simplified for single SmolVLMDataset)
        # The complex dataset metadata structure from SD scripts might not be fully applicable.
        # We can store basic info about the SmolVLMDataset if needed.
        dataset_meta = {
            "is_dreambooth": is_dreambooth_mode,
            "num_train_images": len(train_dataset), # Number of unique images
            "num_repeats": dataset_repeats,
            "batch_size_per_device": args.train_batch_size,
            "smolvlm_longest_edge_n": args.smolvlm_longest_edge_n,
            "min_bucket_reso": train_dataset.min_bucket_reso,
            "max_bucket_reso": train_dataset.max_bucket_reso,
            "bucket_info": train_dataset.bucket_manager.get_bucket_info(),
        }
        metadata["ss_dataset_info"] = json.dumps(dataset_meta)


        # Remove LoRA network args from metadata
        # metadata.pop("ss_network_module", None)
        # metadata.pop("ss_network_dim", None)
        # metadata.pop("ss_network_alpha", None)
        # metadata.pop("ss_network_dropout", None)
        # metadata.pop("ss_network_args", None)
        
        # Remove SD specific model/vae hashes
        # metadata.pop("ss_sd_model_hash", None)
        # metadata.pop("ss_new_sd_model_hash", None)
        # metadata.pop("ss_sd_model_name", None) # Keep if pretrained_model_name_or_path is still used for base
        # metadata.pop("ss_vae_hash", None)
        # metadata.pop("ss_new_vae_hash", None)
        # metadata.pop("ss_vae_name", None)


        metadata = {k: str(v) for k, v in metadata.items()} # Ensure all values are strings for saving

        minimum_metadata = {} # Keep a minimal set if args.no_metadata is not used
        for key in train_util.SS_METADATA_MINIMUM_KEYS: # This list might need review for SmolVLM
            if key in metadata and metadata[key] is not None:
                minimum_metadata[key] = metadata[key]
        # Add SmolVLM specific min metadata
        minimum_metadata["ss_smolvlm_model_id"] = metadata.get("ss_smolvlm_model_id")
        minimum_metadata["ss_smolvlm_longest_edge_n"] = metadata.get("ss_smolvlm_longest_edge_n")


        # calculate steps to skip for resuming
        initial_step = accelerator.state.num_steps # Accelerator handles step resumption
        epoch_to_start = accelerator.state.completed_epochs if hasattr(accelerator.state, "completed_epochs") else 0
        global_step = initial_step # global_step tracks optimizer steps
        
        # If resuming from a specific epoch/step from train_state.json (if accelerator didn't handle it fully)
        if steps_from_state is not None and initial_step < steps_from_state :
            if args.skip_until_initial_step: # This arg might be re-interpreted for SmolVLM
                 logger.info(f"Resuming: Will attempt to skip dataloader to step {steps_from_state} if possible.")
                 # This is complex with sharded dataloaders; accelerator.skip_first_batches is preferred.
                 # For now, we rely on accelerator's state or a simple loop skip.
                 initial_step = steps_from_state # This will be used to skip batches if skip_until_initial_step is true
                 if epoch_from_state is not None:
                     epoch_to_start = epoch_from_state
            else: # Only adjust epoch, let steps start from accelerator's resumed step
                 if epoch_from_state is not None:
                     epoch_to_start = epoch_from_state
                 initial_step = accelerator.state.num_steps # Reset to accelerator's step
        
        logger.info(f"Starting training from epoch {epoch_to_start}, global step {global_step}")


        # Remove SD specific noise scheduler
        # noise_scheduler = DDPMScheduler(...)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_api_key: # Check if wandb_api_key is provided
                init_kwargs["wandb"] = {"entity": args.wandb_entity if hasattr(args, 'wandb_entity') else None , "name": args.wandb_run_name if args.wandb_run_name else "smolvlm-finetune"}
            elif args.wandb_run_name : # Legacy or if API key is set via env
                 init_kwargs["wandb"] = {"name": args.wandb_run_name}

            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        loss_recorder = train_util.LossRecorder()


        # callback for step start
        # TODO: SmolVLM specific callbacks if needed
        # if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        #     on_step_start = accelerator.unwrap_model(network).on_step_start
        # else:
        #     on_step_start = lambda *args, **kwargs: None
        on_step_start = lambda *args, **kwargs: None


        # function for saving/removing
        # TODO: Adapt saving for SmolVLM (model.save_pretrained)
        def save_model(ckpt_name, unwrapped_model, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, ckpt_name) # ckpt_name might be a dir for HF models

            accelerator.print(f"\nsaving checkpoint: {output_path}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            # metadata_to_save = minimum_metadata if args.no_metadata else metadata # TODO: How to save metadata with HF model?
            # sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False) # TODO: Adapt for SmolVLM
            # metadata_to_save.update(sai_metadata)

            # unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save) # Original LoRA save
            unwrapped_model.save_pretrained(output_path, save_function=accelerator.save) # Use accelerator.save for distributed training
            
            # TODO: Save processor/tokenizer state if needed
            # tokenizer.save_pretrained(output_path)


            if args.huggingface_repo_id is not None:
                # huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload) # TODO: Adapt for HF model directory
                huggingface_util.upload_dir(args, output_path, ckpt_name, force_sync_upload=force_sync_upload)


        def remove_model(old_ckpt_name): # TODO: Adapt for HF model directory
            old_ckpt_path = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_path):
                accelerator.print(f"removing old checkpoint: {old_ckpt_path}")
                import shutil
                shutil.rmtree(old_ckpt_path) # Remove directory for HF models
                # os.remove(old_ckpt_file)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        if args.sample_at_first == True:
            self.sample_images(accelerator, args, 0, 0, accelerator.device, vae, tokenizer, text_encoder, unet)
      
        # training loop

        if initial_step > 0:  # only if skip_until_initial_step is specified
            
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                # current_epoch.value = skip_epoch+1
                if args.incremental_reg_reload:
                    train_dataset_group.incremental_reg_load(True)
                initial_step -= num_of_steps
                
            # Caching logic (cache_latents, cache_text_encoder_outputs) is removed as SmolVLMDataset handles its own data prep.
            # Any reloading logic like incremental_reg_reload would need to be implemented in SmolVLMDataset or by re-creating the dataloader.
            # For now, assume dataset is fixed per epoch after initial load.
            
            # if epoch == epoch_to_start or (args.incremental_reg_reload and hasattr(train_dataset, 'incremental_reg_load')):
                 # train_dataset.incremental_reg_load(True) # Example if dataset supports it
                 # Re-prepare dataloader if dataset changes significantly
                 # train_dataloader = accelerator.prepare(torch.utils.data.DataLoader(...))

            logger.info(f"Epoch {epoch+1}/{num_train_epochs}")
            metadata["ss_epoch"] = str(epoch + 1)
            
            # on_epoch_start for model (if any)
            # accelerator.unwrap_model(training_model).on_epoch_start(...) # If model has such a hook

            # Progress bar setup
            # The number of steps in progress_bar should be based on len(train_dataloader)
            progress_bar = tqdm(range(len(train_dataloader)), smoothing=0, disable=not accelerator.is_local_main_process, desc="Steps")

            for step, batch in enumerate(train_dataloader):
                # Skip steps if resuming and initial_step is set by train_state.json
                # This is a simplified batch skipping; accelerator.skip_first_batches is more robust if dataloader state can be saved/restored.
                if global_step < initial_step and args.skip_until_initial_step:
                    if step % args.gradient_accumulation_steps == 0: # only count optimizer steps
                        progress_bar.update(1)
                    if global_step % 100 == 0 : # Log occasionally during skip
                        logger.info(f"Skipping step {global_step}/{initial_step}")
                    if accelerator.sync_gradients: # Optimizer step would have occurred
                         global_step +=1 # Increment global_step as if an optimizer step happened
                    current_step.value = global_step # Keep current_step value up-to-date
                    continue


                with accelerator.accumulate(training_model):
                    # SmolVLM forward pass
                    # Batch items are already on the correct device due to DataLoader + Accelerator
                    try:
                        outputs = training_model(
                            input_ids=batch.get('input_ids'),
                            attention_mask=batch.get('attention_mask'),
                            pixel_values=batch.get('pixel_values'),
                            labels=batch.get('labels')
                        )
                        loss = outputs.loss
                    except Exception as e:
                        logger.error(f"Error during model forward pass at step {global_step}: {e}")
                        logger.error(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
                        if isinstance(batch, dict):
                            logger.error(f"Input IDs shape: {batch.get('input_ids').shape if batch.get('input_ids') is not None else 'None'}")
                            logger.error(f"Pixel values shape: {batch.get('pixel_values').shape if batch.get('pixel_values') is not None else 'None'}")
                        # Decide how to handle this: skip batch, raise error, etc.
                        # For now, let's try to continue if possible, or break if severe.
                        if accelerator.is_main_process:
                            logger.warning("Skipping problematic batch.")
                        accelerator.gradient_state.set_sync_gradients(False) # Don't sync this problematic grad
                        continue # Skip to next batch


                    accelerator.backward(loss)
                    if accelerator.sync_gradients: # Only clip and step if gradients are synced
                        if args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(training_model.parameters(), args.max_grad_norm)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                
                # End of gradient accumulation block
                current_loss = loss.item() # Get scalar loss

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1 # Optimizer step taken
                    current_step.value = global_step # Update shared value

                    # Logging
                    loss_recorder.add(epoch=epoch, step=global_step, loss=current_loss) # Log optimizer steps
                    avr_loss = loss_recorder.moving_average
                    log_data = {"loss/current": current_loss, "loss/average": avr_loss, "lr": lr_scheduler.get_last_lr()[0]}
                    
                    if args.logging_dir is not None:
                        # logs = self.generate_step_logs(args, current_loss, avr_loss, lr_scheduler, lr_descriptions) # keys_scaled etc. are LoRA specific
                        accelerator.log(log_data, step=global_step)
                    
                    progress_bar.set_postfix(log_data)

                    # Saving (model and state)
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "_" + args.save_model_as, global_step)
                            unwrapped_model = accelerator.unwrap_model(training_model)
                            save_model(ckpt_name, unwrapped_model, global_step, epoch +1) # epoch is 0-indexed

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)
                            
                            # Removing old checkpoints (logic might need adjustment for HF model dirs)
                            # remove_step_no = train_util.get_remove_step_no(args, global_step)
                            # if remove_step_no is not None:
                            #     remove_ckpt_name = train_util.get_step_ckpt_name(args, "_" + args.save_model_as, remove_step_no)
                            #     remove_model(remove_ckpt_name) # remove_model needs to handle dirs for HF models
                    
                    # Sampling / Evaluation (placeholder)
                    if args.sample_every_n_steps is not None and global_step % args.sample_every_n_steps == 0:
                        if accelerator.is_main_process:
                            logger.info(f"Sampling/evaluation at step {global_step} (not implemented).")
                            # self.sample_images(...) # Placeholder for SmolVLM eval

                else: # Not a step where gradients are synced (still accumulating)
                    progress_bar.set_postfix({"loss": current_loss})


                if global_step >= args.max_train_steps:
                    logger.info("Max train steps reached. Exiting training loop.")
                    break 
            
            # End of epoch loop
            progress_bar.close()
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                 break # Exit outer epoch loop as well

            # Epoch-based saving
            if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0 :
                if accelerator.is_main_process :
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "_" + args.save_model_as, epoch + 1)
                    unwrapped_model = accelerator.unwrap_model(training_model)
                    save_model(ckpt_name, unwrapped_model, global_step, epoch + 1)
                    
                    # remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    # if remove_epoch_no is not None:
                    #     remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "_" + args.save_model_as, remove_epoch_no)
                    #     remove_model(remove_ckpt_name)
                    
                    if args.save_state:
                         train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
            
            # Epoch-based sampling/evaluation
            if args.sample_every_n_epochs is not None and (epoch + 1) % args.sample_every_n_epochs == 0:
                if accelerator.is_main_process:
                    logger.info(f"Sampling/evaluation at end of epoch {epoch+1} (not implemented).")
                    # self.sample_images(...)

            # Incremental reload (if dataset supports it and it's not the last epoch)
            if hasattr(train_dataset, 'incremental_reg_load') and args.incremental_reg_reload and (epoch + 1 < num_train_epochs):
                logger.info(f"Performing incremental dataset reload for epoch {epoch + 2}")
                train_dataset.incremental_reg_load(is_main_process=accelerator.is_main_process) # Assuming method exists
                # Re-create and re-prepare dataloader as dataset content/length might change
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.train_batch_size, shuffle=True,
                    num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers if n_workers > 0 else False
                )
                train_dataloader = accelerator.prepare(train_dataloader)
                logger.info("Re-prepared DataLoader after incremental load.")

        # End of training loop (epochs)
        accelerator.wait_for_everyone()
        logger.info("Training finished.")

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(training_model)
            # Final save
            ckpt_name = train_util.get_last_ckpt_name(args, "_" + args.save_model_as)
            save_model(ckpt_name, unwrapped_model, global_step, num_train_epochs, force_sync_upload=True)
            logger.info(f"Saved final model to {ckpt_name}")

            if args.save_state or args.save_state_on_train_end:
                train_util.save_state_on_train_end(args, accelerator)
        
        accelerator.end_training()
    
    # Placeholder for SmolVLM specific sampling/evaluation
    # def evaluate_model(self, accelerator, args, model, processor, current_step):
    #    pass

# Example of a simple collator if needed (e.g. if dataset returns lists of items to be batched by processor)
# def smolvlm_collate_fn(batch, processor):
#     texts = [item['text'] for item in batch]
#     images = [item['image'] for item in batch]
#     # Process batch using processor
#     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
#     # Assuming labels are also part of the item and need similar batching/padding
#     # labels = processor.tokenizer(text=[item['labels_text'] for item in batch], return_tensors="pt", padding=True, truncation=True).input_ids
#     # inputs['labels'] = labels 
#     return inputs

# This would be used in DataLoader: collate_fn=lambda b: smolvlm_collate_fn(b, processor)
# However, if SmolVLMDataset already returns fully processed tensors in a dict, default collator is fine.

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args) # Keep for common args
    args = train_util.read_config_from_file(args, parser)

    trainer = SmolVLMTrainer() # Use renamed class
    trainer.train(args)
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if args.sample_every_n_steps is not None and global_step % args.sample_every_n_steps == 0:
                        # example_tuple = (latents.detach().clone(), batch["captions"]) # TODO: SmolVLM sampling
                        accelerator.wait_for_everyone()
                        # self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, example_tuple) # TODO: SmolVLM sampling
                        accelerator.print("TODO: Implement SmolVLM sampling")


                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "_" + args.save_model_as, global_step) # HF model usually saved as dir
                            save_model(ckpt_name, accelerator.unwrap_model(training_model), global_step, epoch) # Changed network to training_model

                            if args.save_state: # TODO: Ensure this works with HF model state
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss, "Global Steps" : global_step}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, keys_scaled, mean_norm, maximum_norm
                    )
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)
            del sharded_dataloader
            accelerator.wait_for_everyone()
            progress_bar.close()

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "_" + args.save_model_as, epoch + 1) # HF model usually saved as dir
                    save_model(ckpt_name, accelerator.unwrap_model(training_model), global_step, epoch + 1) # Changed network to training_model

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "_" + args.save_model_as, remove_epoch_no) # HF model usually saved as dir
                        remove_model(remove_ckpt_name)

                    if args.save_state: # TODO: Ensure this works with HF model state
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            if args.sample_every_n_epochs is not None and (epoch + 1)% args.sample_every_n_epochs == 0:
                # example_tuple = (latents.detach().clone(), batch["captions"]) # TODO: SmolVLM sampling
                accelerator.wait_for_everyone()
                # self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet, example_tuple) # TODO: SmolVLM sampling
                accelerator.print("TODO: Implement SmolVLM sampling")
                
           
            # Reloading reg images here and checking cache before train_dataloader's workers are reinitialized
            # TODO: This logic is for SD, adapt if SmolVLM needs something similar (e.g. dynamic dataset aspects)
            if args.incremental_reg_reload and epoch + 1 < num_train_epochs:
                train_dataset_group.incremental_reg_load(True) # TODO: Adapt for SmolVLMDataset
                '''
                # Leave in place but comment out for testing
                if cache_latents:
                    vae.to(accelerator.device, dtype=vae_dtype)
                    vae.requires_grad_(False)
                    vae.eval()
                    with torch.no_grad():
                        train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
                    vae.to("cpu")
                    clean_memory_on_device(accelerator.device)
        
                    accelerator.wait_for_everyone()
                else:
                    vae.requires_grad_(False)
                    vae.eval()
                    vae.to(accelerator.device, dtype=vae_dtype)

            # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
            # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
                if args.cache_text_encoder_outputs:
                    self.cache_text_encoder_outputs_if_needed(
                        args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
                    )
                accelerator.wait_for_everyone()  # Wait for all processes to finish dataset/caching
                ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
                collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset_group,
                    batch_size=1,
                    shuffle=True,
                    collate_fn=collator,
                    num_workers=n_workers,
                    persistent_workers=args.persistent_data_loader_workers,
                )
        
                sharded_dataloader = accelerator.prepare(train_dataloader)
            '''
            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            # network = accelerator.unwrap_model(network) # Original
            model = accelerator.unwrap_model(training_model) # SmolVLM

        accelerator.end_training()

        if is_main_process and (args.save_state or args.save_state_on_train_end): # TODO: Ensure this works with HF model state
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "_" + args.save_model_as) # HF model usually saved as dir
            save_model(ckpt_name, model, global_step, num_train_epochs, force_sync_upload=True) # Changed network to model

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    # parser.add_argument(
    #     "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    # )
    # parser.add_argument(
    #     "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    # )
    # parser.add_argument(
    #     "--network_dim",
    #     type=int,
    #     default=None,
    #     help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    # )
    # parser.add_argument(
    #     "--network_alpha",
    #     type=float,
    #     default=1,
    #     help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    # )
    # parser.add_argument(
    #     "--network_dropout",
    #     type=float,
    #     default=None,
    #     help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    # )
    # parser.add_argument(
    #     "--network_args",
    #     type=str,
    #     default=None,
    #     nargs="*",
    #     help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    # )
    # parser.add_argument(
    #     "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    # )
    # parser.add_argument(
    #     "--network_train_text_encoder_only",
    #     action="store_true",
    #     help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    # )
    parser.add_argument("--toml_captions_dir", type=str, default=None, help="directory containing TOML caption files")
    parser.add_argument("--smolvlm_model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct", help="SmolVLM model identifier")
    parser.add_argument("--smolvlm_longest_edge_n", type=int, default=4, help="N value for longest_edge: N*384 image sizing")
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--resume_from_epoch",
        action="store_true",
        help="Sets initial epoch to last epoch from state, does not skip initial steps",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )

    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args) # Keep for common args
    args = train_util.read_config_from_file(args, parser)

    trainer = SmolVLMTrainer() # Use renamed class
    trainer.train(args)
