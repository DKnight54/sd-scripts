"""
Main training script for fine-tuning SmolVLM-style Vision-Language Models
on image-caption datasets using Hugging Face Accelerate for distributed training
and PEFT (LoRA/QLoRA) for efficient fine-tuning.

This script handles:
1.  Argument parsing for detailed training configuration.
2.  Distributed training setup using Hugging Face Accelerate.
3.  Dataset creation (`SmolVLMDataset`) for image-caption pairs, including
    image bucketing/resizing and chat-templated prompt construction.
4.  Optional PEFT (LoRA/QLoRA) setup for parameter-efficient fine-tuning,
    including 4-bit and 8-bit quantization.
5.  Model loading (e.g., `HuggingFaceTB/SmolVLM-Instruct`) with support for
    mixed precision (fp16, bf16), Flash Attention 2, and quantization.
6.  Optimizer setup (AdamW).
7.  A training loop with gradient accumulation, gradient clipping, and
    logging (TensorBoard/W&B).
8.  Comprehensive checkpointing: saving and resuming model state (full or adapter),
    optimizer, scheduler, processor, and custom tracker state.
9.  Saving the final model (full or adapter) and processor.
10. Optional sample generation after training.
"""
import argparse
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm 
import random 
import datetime # for ddp_timeout
import json # for tracker_state.json
from torch.optim import AdamW # Use torch.optim.AdamW

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig # For QLoRA
import torch # Ensure torch is imported for BitsAndBytesConfig dtypes

from library.smolvlm_train_util import load_and_preprocess_image, process_image_caption_item

class SmolVLMDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing image-caption pairs for SmolVLM training.

    Attributes:
        min_bucket_reso (int): Minimum resolution for image bucketing.
        max_bucket_reso (int): Maximum resolution for image bucketing.
        processor (AutoProcessor): Hugging Face processor for tokenizing text and processing images.
        max_token_length (int): Maximum sequence length for tokenized inputs.
        data (list): A list of tuples, where each tuple contains (image_path, qa_pair_dict).
    """
    def __init__(self, image_folder, min_bucket_reso, max_bucket_reso, processor, max_token_length, accelerator=None):
        """
        Initializes the SmolVLMDataset.

        Args:
            image_folder (str): Path to the folder containing images.
                                Corresponding .txt files are expected for captions.
            min_bucket_reso (int): Minimum resolution for image bucketing.
            max_bucket_reso (int): Maximum resolution for image bucketing.
            processor (AutoProcessor): The Hugging Face processor.
            max_token_length (int): Max sequence length for tokenized inputs.
            accelerator (Accelerator, optional): Accelerator instance for distributed logging.
        """
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.processor = processor
        self.max_token_length = max_token_length
        self.data = []
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.webp') # Supported image types
        self.accelerator = accelerator

        # Use accelerator.print if available, otherwise fallback to standard print
        self.printf = self.accelerator.print if self.accelerator else print

        self.printf(f"Scanning folder: {image_folder} for images and captions...")
        # Iterate through files in the image_folder
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(self.image_extensions):
                image_path = os.path.join(image_folder, filename)
                # process_image_caption_item handles finding .txt file, loading caption, and creating QA pair
                qa_pair = process_image_caption_item(image_path, log_missing_captions=True, printf=self.printf) # Pass printf
                if qa_pair:
                    self.data.append((image_path, qa_pair))
                else:
                    # Log if a caption or QA pair couldn't be formed for an image
                    self.printf(f"Skipping image {image_path} due to missing or invalid caption/QA pair.")
        self.printf(f"Found {len(self.data)} valid image-caption pairs.")

    def __len__(self):
        """Returns the total number of image-caption pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single image-caption pair for model training.

        This method performs several key steps:
        1.  Loads the image and resizes it according to bucketing parameters using `load_and_preprocess_image`.
        2.  Constructs a conversation `messages` list (user with image+question, assistant with answer)
            suitable for the model's chat template.
        3.  Applies the processor's chat template to get the full prompt string.
        4.  Uses the `self.processor` (Hugging Face AutoProcessor) to:
            a.  Tokenize the prompt string.
            b.  Process the PIL image into the model's expected image format (e.g., pixel values).
            c.  Combine text and image features, pad/truncate to `max_token_length`.
        5.  Creates `labels` for training:
            a.  Labels are initially a copy of `input_ids`.
            b.  The prompt part of the labels (user's question and any template-specific tokens
                before the assistant's response) is masked with -100. This is achieved by
                tokenizing the user-only part of the conversation to determine its length (`prompt_len`).
            c.  Padding tokens in the labels are also masked with -100.
        
        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            dict: A dictionary containing processed model inputs, including `input_ids`,
                  `attention_mask`, `pixel_values`, and masked `labels`. Returns `None` if
                  any critical step like image loading or processing fails.
        """
        image_path, qa_pair = self.data[idx]

        # Load and preprocess the image (bucketing, resizing)
        # Pass self.printf to the image loading function
        pil_image = load_and_preprocess_image(image_path, self.min_bucket_reso, self.max_bucket_reso, printf=self.printf)

        if pil_image is None:
            # self.printf will be used internally by load_and_preprocess_image if an error occurs
            # No need for an additional print here unless it's a different message.
            # self.printf(f"Warning: load_and_preprocess_image returned None for {image_path} (logged internally). Skipping this item.")
            return None # Handled by collate_fn

        question = qa_pair["question"]
        answer = qa_pair["answer"]

        # Construct messages for chat template (training format)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        # Get the full prompt string (text part of the input) using the chat template
        # tokenize=False here because the processor call below will handle tokenization of text and image together
        prompt_text = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=False, # Important for training: model learns to generate assistant's reply
            tokenize=False 
        )

        try:
            # Process combined text and image using the Hugging Face processor
            inputs = self.processor(
                text=prompt_text, 
                images=pil_image, 
                return_tensors="pt", 
                padding="max_length",    # Pad to max_token_length
                truncation=True,         # Truncate if longer than max_token_length
                max_length=self.max_token_length
            )
        except Exception as e:
            current_printf = self.accelerator.print if hasattr(self, 'accelerator') and self.accelerator else print
            current_printf(f"Error during processor call for image {image_path} with prompt '{prompt_text[:100]}...'. Error: {e}")
            return None # Handled by collate_fn

        # Remove the batch dimension that processor might add; DataLoader will add its own.
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # --- Create and Mask Labels for Causal Language Modeling ---
        # Labels are input_ids shifted (for CausalLMs) or specific parts (for Seq2Seq/VLM).
        # Here, we mask the prompt part so loss is only calculated on the assistant's response.
        labels = inputs["input_ids"].clone()

        # Determine the length of the prompt (user part + template tokens before assistant's reply)
        user_messages_for_length_calc = [messages[0]] # Only the user's turn
        
        # Tokenize just the user part to find out how many tokens it occupies
        # This includes user's text, image placeholder, and any template tokens for the user turn
        user_prompt_segments = self.processor.apply_chat_template(
            user_messages_for_length_calc,
            add_generation_prompt=False, # Critical: ensures we only get tokens BEFORE assistant's turn
            tokenize=True,               # Need token IDs to determine length
            return_tensors="pt",
            padding=False,               # Don't pad this segment; we only need its actual length
            truncation=True,             # Truncate if user prompt alone is too long
            max_length=self.max_token_length 
        )
        
        prompt_len = user_prompt_segments["input_ids"].shape[1] # Number of tokens in the user part

        # Mask tokens in `labels` that belong to the prompt, so they don't contribute to loss
        if prompt_len < labels.shape[0]: # Check if there's anything left to be an answer
            labels[:prompt_len] = -100
        else:
            # This implies the prompt itself (user part) filled or exceeded max_token_length.
            # All tokens will be masked, meaning no effective label for this instance.
            current_printf = self.accelerator.print if hasattr(self, 'accelerator') and self.accelerator else print
            current_printf(f"Warning: Prompt length ({prompt_len}) is >= total sequence length ({labels.shape[0]}) for {image_path}. Entire label sequence will be masked.")
            labels[:] = -100

        # Also, explicitly mask any padding tokens in the labels.
        # `inputs["input_ids"]` contains padding tokens if sequence is shorter than `max_token_length`.
        if self.processor.tokenizer.pad_token_id is not None:
            padding_mask = (inputs["input_ids"] == self.processor.tokenizer.pad_token_id)
            labels[padding_mask] = -100
        
        inputs["labels"] = labels
        # --- End Label Creation and Masking ---
            
        return inputs


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Filters out None items from the batch (which can occur if `SmolVLMDataset.__getitem__`
    returns None due to an error in processing a specific item).
    Then, uses the default PyTorch collate function to combine valid items into a batch.
    """
    # Filter out None items (e.g., from failed image loads or processing errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the batch is empty after filtering
    # Use the default collate function for the rest of the items
    return torch.utils.data.dataloader.default_collate(batch)


def main(args):
    """
    Main function to orchestrate the SmolVLM training process.
    Handles:
    - Device setup (CUDA/CPU).
    - Processor and Dataset initialization.
    - DataLoader creation.
    - Model loading (SmolVLM) with specified mixed precision and attention mechanism.
    - Optimizer (AdamW) setup.
    - Training loop over epochs and batches, including:
        - Forward pass, loss calculation.
        - Backward pass, optimizer step.
        - Mixed precision handling (autocast, GradScaler for fp16).
    - Saving the trained model and processor.
    - Generating sample image-caption pairs using the trained model.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        log_with=args.log_with,
        project_dir=args.output_dir, 
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))] 
    )
    set_seed(args.seed)
    if args.log_with: # Initialize trackers if log_with is set
        accelerator.init_trackers("smolvlm_finetune_" + args.output_name, config=vars(args))

    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Initialize Hugging Face processor
    accelerator.print("Initializing processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # Set pad_token if not already defined (common for some models like Llama)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id # Crucially, set token ID too
        accelerator.print(f"Set tokenizer pad_token to eos_token (ID: {processor.tokenizer.eos_token_id}).")

    # Initialize dataset
    accelerator.print("Initializing dataset...")
    train_dataset = SmolVLMDataset(
        args.image_folder,
        args.min_bucket_reso,
        args.max_bucket_reso,
        processor,
        args.max_token_length,
        accelerator=accelerator # Pass accelerator for logging within dataset
    )

    if len(train_dataset) == 0:
        accelerator.print("No training data found. Please check your image_folder and caption files. Exiting.")
        return

    accelerator.print(f"Found {len(train_dataset)} image-caption pairs for training.")

    # Initialize DataLoader
    accelerator.print("Initializing DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn # Use custom collate_fn to handle potential Nones
    )

    # Load model
    accelerator.print("Loading model...")

    quantization_config = None
    model_dtype = torch.float32 # Default for "no" mixed precision or CPU
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    if args.load_in_4bit:
        accelerator.print("Loading base model in 4-bit for QLoRA...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype, # Use determined model_dtype
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        accelerator.print("Loading base model in 8-bit for QLoRA...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    attn_implementation = "eager" # Default attention
    if str(device) == "cuda": # Check device type from accelerator
        # Try to use Flash Attention 2 if available and on CUDA
        try:
            import flash_attn # Check if flash_attn is installed
            attn_implementation = "flash_attention_2"
            accelerator.print("Attempting to use flash_attention_2.")
        except ImportError:
            accelerator.print("flash_attn library not found, defaulting to 'eager' attention implementation.")
    
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype if quantization_config is None else None, # torch_dtype not to be used with BitsAndBytesConfig
        trust_remote_code=True,       # Required for some custom models
        attn_implementation=attn_implementation, # Use Flash Attention 2 if available
        quantization_config=quantization_config
    )
    
    accelerator.print(f"Base model loaded. Using dtype: {model.dtype}, Attention: {attn_implementation}.")
    if quantization_config:
        accelerator.print(f"Quantization config applied: {quantization_config.to_dict()}")


    if args.use_peft:
        accelerator.print("Applying PEFT (LoRA/QLoRA) to the model...")
        if not args.peft_target_modules:
             accelerator.print("Warning: --peft_target_modules is empty or not provided. LoRA may not be applied effectively. Common targets: q_proj, v_proj, k_proj, o_proj, fc1, fc2, etc., within the language_model part of SmolVLM.")

        peft_config = LoraConfig(
            r=args.peft_lora_r,
            lora_alpha=args.peft_lora_alpha,
            lora_dropout=args.peft_lora_dropout,
            target_modules=args.peft_target_modules if args.peft_target_modules else None,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM, # SmolVLM is Vision2Seq
        )
        model = get_peft_model(model, peft_config)
        accelerator.print("PEFT model created. Trainable parameters:")
        model.print_trainable_parameters() # PeftModel has this utility method

    # model.to(device) # Accelerator handles device placement - This line was already commented
    # accelerator.print(f"Model loaded with dtype {model_dtype} and attention: {attn_implementation}.") # Replaced by more specific prints above

    # Setup optimizer
    accelerator.print("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Resuming from Checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            tracker_file = os.path.join(args.resume_from_checkpoint, "tracker_state.json")
            if os.path.exists(tracker_file):
                with open(tracker_file, "r") as f:
                    tracker_state = json.load(f)
                completed_epoch = tracker_state.get("epoch", -1) 
                global_step = tracker_state.get("global_step", 0)
                start_epoch = completed_epoch + 1 # Start from the next epoch
                accelerator.print(f"Resumed from completed epoch {completed_epoch} and global_step {global_step}. Starting epoch {start_epoch}.")
            else:
                accelerator.print(f"tracker_state.json not found in {args.resume_from_checkpoint}. Resuming optimizer/scheduler/model weights only. Epoch/step will start from scratch unless parsed from dir name.")
        else:
            accelerator.print(f"Checkpoint {args.resume_from_checkpoint} not found or not a directory. Starting from scratch.")

    # Training loop
    accelerator.print(f"Starting training from epoch {start_epoch} for {args.num_train_epochs} total epochs...")
    for epoch in range(start_epoch, args.num_train_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        num_batches_processed_this_epoch = 0 # Changed from num_batches
        
        # Progress bar for batches within an epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}", disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: # Skip if collate_fn returned None (empty batch after filtering)
                accelerator.print(f"Skipping empty or invalid batch {batch_idx+1}.")
                continue

            # Move batch data to the target device - REMOVED (handled by Accelerator via prepare)
            
            # optimizer.zero_grad() # Moved into accumulate block
            
            # Mixed precision context managers - REMOVED (handled by Accelerator)
            
            # Training step logic with accelerator.accumulate
            with accelerator.accumulate(model):
                optimizer.zero_grad() # Zero gradients at the start of accumulation
                outputs = model(**batch) 
                loss = outputs.loss

                if loss is not None:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients: # True if this is an optimization step
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                # Loss accumulation and progress bar update (inside `if loss is not None:`)
                if loss is not None:
                    current_loss = loss.item()
                    epoch_loss += current_loss 
                    num_batches_processed_this_epoch += 1 

                    if accelerator.is_local_main_process:
                        progress_bar.set_postfix({"loss": current_loss, "avg_epoch_loss": epoch_loss / num_batches_processed_this_epoch})
                    
                    # Log step loss when gradients are synced (actual optimizer step)
                    if accelerator.sync_gradients:
                         if global_step > 0 and (global_step % 10 == 0 or (args.save_every_n_steps == 1 and global_step > 0)) : # Log every 10 opt steps or if saving every step
                            accelerator.log({"train/step_loss": current_loss}, step=global_step)
            
            # Increment global_step and save periodic checkpoints (after accumulation block)
            if loss is not None: # Only increment step if a valid forward/backward pass happened
                if accelerator.sync_gradients: # This ensures global_step increments only on actual optimization steps
                    global_step += 1
                    if args.save_every_n_steps and global_step % args.save_every_n_steps == 0 and global_step > 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                            accelerator.save_state(save_path)
                            # Save processor with the checkpoint
                            processor.save_pretrained(save_path) 
                            tracker_state = {"epoch": epoch, "global_step": global_step} # Save completed epoch and current step
                            with open(os.path.join(save_path, "tracker_state.json"), "w") as f:
                                json.dump(tracker_state, f)
                            accelerator.print(f"Saved checkpoint, processor, and tracker state to {save_path}")
            
            if loss is None and accelerator.is_local_main_process: # Moved this condition here
                 accelerator.print(f"Warning: Batch {batch_idx+1} in epoch {epoch+1} produced no loss.")


        if num_batches_processed_this_epoch > 0:
            avg_epoch_loss = epoch_loss / num_batches_processed_this_epoch
            accelerator.log({"train/epoch_loss": avg_epoch_loss}, step=epoch) 
            accelerator.print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        else:
            accelerator.print(f"Epoch {epoch+1} completed. No batches processed or no loss recorded.")

    accelerator.print("Training finished.")

    # Save model and processor
    # args.output_dir is already the specific path for this run's outputs (e.g., output_root/output_name_arg)
    model_save_path = args.output_dir 
    accelerator.print(f"Saving model and processor to {model_save_path}...")
    # model.save_pretrained(model_save_path) # Will be replaced by accelerator save
    # processor.save_pretrained(model_save_path) 
    # accelerator.print(f"Model and processor saved successfully.")
    # accelerator.print(f"Note: Actual model file format (e.g., .safetensors) depends on Hugging Face defaults and installed libraries.")
    # accelerator.print(f"The argument --save_model_as ('{args.save_model_as}') is noted, but save_pretrained typically defaults to safetensors if available.")
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_save_path) # Saves adapter if PEFT, full model otherwise
        processor.save_pretrained(model_save_path)
    accelerator.wait_for_everyone()
    accelerator.print(f"Model (adapter if PEFT used) and processor saved successfully to {model_save_path}.")
    if args.log_with:
        accelerator.end_training()


    # --- Sample Generation ---
    if accelerator.is_main_process and args.num_sample_answers > 0 and len(train_dataset.data) > 0:
        accelerator.print("\nStarting sample generation...")
        unwrapped_model = accelerator.unwrap_model(model) # Use unwrapped model
        unwrapped_model.eval() # Set model to evaluation mode for inference
        
        sample_dir = os.path.join(model_save_path, "samples") # Create a "samples" subdir
        os.makedirs(sample_dir, exist_ok=True)
        accelerator.print(f"Generating up to {args.num_sample_answers} samples in {sample_dir}...")

        available_data = train_dataset.data # Use data loaded by the dataset
        num_to_sample = min(args.num_sample_answers, len(available_data))
        selected_samples = random.sample(available_data, num_to_sample)

        # Use the same fixed question as defined for training Q&A pairs
        fixed_question_for_sampling = (
            "Caption this image for stable diffusion training as accurately as possible. "
            "Use \"@@@@,\" to separate the front of the caption which will be kept static "
            "except for keyword substitution. Also for items/actions that may have multiple "
            "keywords, use the format {keyword1|keyword2|keyword3} to allow for substitution "
            "during training. Commas are not allowed inside of curly braces."
        )

        for image_path, _ in tqdm(selected_samples, desc="Generating Samples", disable=not accelerator.is_local_main_process): # original_qa_pair not needed for inference here
            # Call load_and_preprocess_image once, passing accelerator.print for logging
            pil_image = load_and_preprocess_image(image_path, args.min_bucket_reso, args.max_bucket_reso, printf=accelerator.print)
            if pil_image is None:
                # Message will be printed inside load_and_preprocess_image if an error occurs
                continue

            # Prepare input for inference using chat template
            messages_for_inference = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": fixed_question_for_sampling}
                    ]
                }
            ]
            
            # `add_generation_prompt=True` is typically used for inference to signal model to start generating
            prompt_text_for_inference = processor.apply_chat_template(
                messages_for_inference, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            inputs_for_inference = processor(
                text=prompt_text_for_inference, 
                images=pil_image, 
                return_tensors="pt",
                padding=True, # Allow padding for generation if needed
                truncation=True, 
                max_length=args.max_token_length 
            ).to(accelerator.device) # Move inputs to accelerator device

            # Generate text
            with torch.no_grad(): # Disable gradient calculations for inference
                generated_ids = unwrapped_model.generate( # Use unwrapped_model
                    **inputs_for_inference, 
                    max_new_tokens=args.sample_max_new_tokens, 
                    do_sample=False, # Use greedy search for deterministic samples
                    num_beams=1      # Number of beams for beam search (1 for greedy)
                )
            
            # Decode generated tokens, attempting to get only the newly generated part
            input_token_len = inputs_for_inference.input_ids.shape[1]
            newly_generated_ids = generated_ids[:, input_token_len:] # Slice to get only new tokens
            generated_answer = processor.batch_decode(newly_generated_ids, skip_special_tokens=True)[0].strip()
            
            if not generated_answer: # Handle case where no new text is generated
                generated_answer = "Error: No new text generated or empty output after decoding."


            # Save the sample image and generated text
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            try:
                # Save the original (bucketed) PIL image
                pil_image.save(os.path.join(sample_dir, f"{base_filename}.png"))
            except Exception as e:
                accelerator.print(f"Error saving sample image {base_filename}.png: {e}")

            try:
                with open(os.path.join(sample_dir, f"{base_filename}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Question:\n{fixed_question_for_sampling}\n\nGenerated Answer:\n{generated_answer}\n")
                accelerator.print(f"Saved sample for {base_filename}")
            except Exception as e:
                accelerator.print(f"Error saving sample text {base_filename}.txt: {e}")

        accelerator.print("Sample generation finished.")
    # model.train() # Not strictly necessary here as it's the end of main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main training script for fine-tuning SmolVLM models on image-caption datasets with Accelerate and PEFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # --- Core Arguments ---
    # Dataset and Paths
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images and corresponding .txt caption files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save trained models, checkpoints, logs, and samples. A subdirectory named by --output_name will be created here.")
    parser.add_argument("--output_name", type=str, required=True, help="Specific name for this training run's output subdirectory (created within output_dir). Used for organizing outputs and naming logging runs.")
    
    # Image Bucketing and Tokenization
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="Minimum resolution (longest side) for image bucketing during preprocessing.")
    parser.add_argument("--max_bucket_reso", type=int, default=1536, help="Maximum resolution (longest side) for image bucketing during preprocessing.")
    parser.add_argument("--max_token_length", type=int, default=512, help="Maximum token length for the processor (text sequence length). Inputs will be padded/truncated to this length.")

    # Model Parameters
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceTB/SmolVLM-Instruct", help="Hugging Face model name or path to a local model directory.")
    
    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for the AdamW optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device for training. Effective batch size is (train_batch_size * gradient_accumulation_steps * num_processes).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate gradients before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping. Set to 0 or less to disable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across all processes.")

    # Mixed Precision and Hardware
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training type, handled by Hugging Face Accelerate. 'bf16' is recommended for Ampere+ GPUs.")

    # PEFT (LoRA/QLoRA) Arguments
    parser.add_argument("--use_peft", action='store_true', help="Enable PEFT (LoRA or QLoRA if quantization is also enabled) for parameter-efficient fine-tuning.")
    parser.add_argument("--peft_lora_r", type=int, default=8, help="LoRA rank (dimension of the LoRA update matrices).")
    parser.add_argument("--peft_lora_alpha", type=int, default=16, help="LoRA alpha scaling factor.")
    parser.add_argument("--peft_lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers.")
    parser.add_argument("--peft_target_modules", type=str, nargs='*', default=["q_proj", "v_proj"], help="Space-separated list of module names within the base model to apply LoRA to. For SmolVLM, these are typically in the language_model component (e.g., 'language_model.model.layers.0.self_attn.q_proj'). Needs to be specific. Common examples: 'q_proj', 'v_proj', 'k_proj', 'o_proj', 'fc1', 'fc2'.")
    
    # Quantization (QLoRA) Arguments
    parser.add_argument("--load_in_8bit", action='store_true', help="Load the base model in 8-bit precision for QLoRA. Requires bitsandbytes.")
    parser.add_argument("--load_in_4bit", action='store_true', help="Load the base model in 4-bit precision for QLoRA (e.g., NF4). Requires bitsandbytes.")

    # Checkpointing and Logging
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory (created by a previous run of this script) to resume training from. Resumes model, optimizer, scheduler, processor, and tracker state.")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="Save a checkpoint (full training state: model/adapter, optimizer, scheduler, processor, tracker) every N global optimizer steps. Overrides epoch-based saving if set.")
    # parser.add_argument("--save_model_as", type=str, default="safetensors", choices=["ckpt", "pt", "safetensors"], help="Format for saving the full model (if not using PEFT). For PEFT, adapters are saved via save_pretrained in its standard format. `save_pretrained` default behavior is used, typically preferring SafeTensors if available.") # This argument is less relevant now, PEFT saves adapters in its own way. Keeping for full model saving if PEFT not used.
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], help="Logging tool to use (tensorboard, wandb, or all), integrated with Hugging Face Accelerate. Logs will be saved in a subfolder within `output_dir/output_name`. For wandb, ensure you are logged in (`wandb login`).")
    
    # Sample Generation (after training)
    parser.add_argument("--num_sample_answers", type=int, default=5, help="Number of image-answer sample pairs to generate and save after training (on main process only).")
    parser.add_argument("--sample_max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate for each sample answer during inference.")
    
    # Note: --save_model_as is implicitly handled by PEFT saving logic or standard save_pretrained.
    # The argument below is kept for potential non-PEFT full model saving, though PEFT is the primary focus.
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=["ckpt", "pt", "safetensors"], help="Format for saving the full model if not using PEFT. When using PEFT, adapters are saved in their standard format by `save_pretrained`. This argument primarily influences full model saves. SafeTensors is generally preferred.")


    # Unused arguments (placeholders for potential future compatibility)
    parser.add_argument("--incremental_reg_reload", action='store_true', help="Reload regularization images incrementally (Not implemented in this script).")
    parser.add_argument("--randomized_regularization_image", action='store_true', help="Randomize regularization images (Not implemented in this script).")

    args = parser.parse_args()
    
    # Construct the final output path for this specific run
    final_output_dir_for_run = os.path.join(args.output_dir, args.output_name)
    if not os.path.exists(final_output_dir_for_run):
        os.makedirs(final_output_dir_for_run, exist_ok=True) # exist_ok=True in case of multi-process race (though not used here)
        # This print might happen before accelerator is ready, so keep it standard for now or ensure it's main process only.
        if Accelerator().is_main_process: # Temp accelerator for early print
             print(f"Created output directory for this run: {final_output_dir_for_run}")
    
    # Update args.output_dir to be this specific path for convenience within main()
    args.output_dir = final_output_dir_for_run

    main(args)
