"""
Main training script for fine-tuning SmolVLM-style Vision-Language Models
on image-caption datasets.

This script handles:
1.  Argument parsing for training configuration.
2.  Dataset creation (`SmolVLMDataset`) which loads images and prepares
    image-text pairs using a Hugging Face processor. This includes image
    bucketing/resizing and constructing appropriate chat-templated prompts.
3.  Label masking for training, ensuring loss is computed only on the
    assistant's (answer) part of the conversation.
4.  DataLoader setup.
5.  Model loading (e.g., `HuggingFaceTB/SmolVLM-Instruct`) with support for
    mixed precision (fp16, bf16) and Flash Attention 2.
6.  Optimizer setup (AdamW).
7.  A training loop that iterates through epochs and batches, performs
    forward/backward passes, and updates model weights.
8.  Saving the trained model and processor.
9.  Optional sample generation after training to qualitatively assess model performance.
"""
import argparse
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, AdamW
from tqdm import tqdm 
import random 

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
    def __init__(self, image_folder, min_bucket_reso, max_bucket_reso, processor, max_token_length):
        """
        Initializes the SmolVLMDataset.

        Args:
            image_folder (str): Path to the folder containing images.
                                Corresponding .txt files are expected for captions.
            min_bucket_reso (int): Minimum resolution for image bucketing.
            max_bucket_reso (int): Maximum resolution for image bucketing.
            processor (AutoProcessor): The Hugging Face processor.
            max_token_length (int): Max sequence length for tokenized inputs.
        """
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.processor = processor
        self.max_token_length = max_token_length
        self.data = []
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.webp') # Supported image types

        print(f"Scanning folder: {image_folder} for images and captions...")
        # Iterate through files in the image_folder
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(self.image_extensions):
                image_path = os.path.join(image_folder, filename)
                # process_image_caption_item handles finding .txt file, loading caption, and creating QA pair
                qa_pair = process_image_caption_item(image_path, log_missing_captions=True)
                if qa_pair:
                    self.data.append((image_path, qa_pair))
                else:
                    # Log if a caption or QA pair couldn't be formed for an image
                    print(f"Skipping image {image_path} due to missing or invalid caption/QA pair.")
        print(f"Found {len(self.data)} valid image-caption pairs.")

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
        pil_image = load_and_preprocess_image(image_path, self.min_bucket_reso, self.max_bucket_reso)

        if pil_image is None:
            print(f"Warning: load_and_preprocess_image returned None for {image_path}. Skipping this item.")
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
            print(f"Error during processor call for image {image_path} with prompt '{prompt_text[:100]}...'. Error: {e}")
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
            print(f"Warning: Prompt length ({prompt_len}) is >= total sequence length ({labels.shape[0]}) for {image_path}. Entire label sequence will be masked.")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Hugging Face processor
    print("Initializing processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # Set pad_token if not already defined (common for some models like Llama)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id # Crucially, set token ID too
        print(f"Set tokenizer pad_token to eos_token (ID: {processor.tokenizer.eos_token_id}).")

    # Initialize dataset
    print("Initializing dataset...")
    train_dataset = SmolVLMDataset(
        args.image_folder,
        args.min_bucket_reso,
        args.max_bucket_reso,
        processor,
        args.max_token_length 
    )

    if len(train_dataset) == 0:
        print("No training data found. Please check your image_folder and caption files. Exiting.")
        return

    print(f"Found {len(train_dataset)} image-caption pairs for training.")

    # Initialize DataLoader
    print("Initializing DataLoader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn # Use custom collate_fn to handle potential Nones
    )

    # Load model
    print("Loading model...")
    model_dtype = torch.float32 # Default for "no" mixed precision or CPU
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16

    attn_implementation = "eager" # Default attention
    if device.type == "cuda":
        # Try to use Flash Attention 2 if available and on CUDA
        try:
            import flash_attn # Check if flash_attn is installed
            attn_implementation = "flash_attention_2"
            print("Attempting to use flash_attention_2.")
        except ImportError:
            print("flash_attn library not found, defaulting to 'eager' attention implementation.")
    
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype,      # Set dtype for model weights
        trust_remote_code=True,       # Required for some custom models
        attn_implementation=attn_implementation # Use Flash Attention 2 if available
    )
    model.to(device) # Move model to the target device
    print(f"Model loaded on {device} with dtype {model_dtype} and attention: {attn_implementation}.")

    # Setup optimizer
    print("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Setup GradScaler for fp16 mixed precision
    scaler = None
    if args.mixed_precision == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Using GradScaler for fp16 mixed precision.")

    # Training loop
    print(f"Starting training for {args.num_train_epochs} epochs...")
    for epoch in range(args.num_train_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar for batches within an epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: # Skip if collate_fn returned None (empty batch after filtering)
                print(f"Skipping empty or invalid batch {batch_idx+1}.")
                continue

            # Move batch data to the target device
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                # Non-tensor items (if any) are kept as is, though usually all items are tensors
            
            optimizer.zero_grad() # Clear previous gradients

            # Mixed precision context managers
            if args.mixed_precision == "bf16" and device.type == "cuda":
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(**batch_on_device) # Forward pass
                    loss = outputs.loss
                # Backward pass and optimizer step are outside autocast for bf16
                if loss is not None: 
                    loss.backward()
                    optimizer.step()
            elif args.mixed_precision == "fp16" and device.type == "cuda" and scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(**batch_on_device) # Forward pass
                    loss = outputs.loss
                if loss is not None:
                    scaler.scale(loss).backward() # Scale loss for fp16
                    scaler.step(optimizer)       # Optimizer step
                    scaler.update()              # Update scaler
            else: # "no" mixed precision or CPU training
                outputs = model(**batch_on_device) # Forward pass
                loss = outputs.loss
                if loss is not None:
                    loss.backward()
                    optimizer.step()
            
            if loss is not None:
                current_loss = loss.item()
                epoch_loss += current_loss
                num_batches += 1
                # Update progress bar postfix with current loss and running epoch average
                if (batch_idx + 1) % 10 == 0: # Log every 10 steps
                    progress_bar.set_postfix({"loss": current_loss, "avg_epoch_loss": epoch_loss / num_batches})
            else:
                # This might happen if all items in a batch had labels fully masked, or model issue.
                print(f"Warning: Batch {batch_idx+1} in epoch {epoch+1} produced no loss.")


        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # Save model and processor
    # args.output_dir is already the specific path for this run's outputs (e.g., output_root/output_name_arg)
    model_save_path = args.output_dir 
    print(f"Saving model and processor to {model_save_path}...")
    model.save_pretrained(model_save_path) 
    processor.save_pretrained(model_save_path) # Save processor to the same directory
    print(f"Model and processor saved successfully.")
    print(f"Note: Actual model file format (e.g., .safetensors) depends on Hugging Face defaults and installed libraries.")
    print(f"The argument --save_model_as ('{args.save_model_as}') is noted, but save_pretrained typically defaults to safetensors if available.")

    # --- Sample Generation ---
    if args.num_sample_answers > 0 and len(train_dataset.data) > 0:
        print("\nStarting sample generation...")
        model.eval() # Set model to evaluation mode for inference
        
        sample_dir = os.path.join(model_save_path, "samples") # Create a "samples" subdir
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Generating up to {args.num_sample_answers} samples in {sample_dir}...")

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

        for image_path, _ in tqdm(selected_samples, desc="Generating Samples"): # original_qa_pair not needed for inference here
            pil_image = load_and_preprocess_image(image_path, args.min_bucket_reso, args.max_bucket_reso)
            if pil_image is None:
                print(f"Warning: Could not load/preprocess image {image_path} for sampling. Skipping.")
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
            ).to(device)

            # Generate text
            with torch.no_grad(): # Disable gradient calculations for inference
                generated_ids = model.generate(
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
                print(f"Error saving sample image {base_filename}.png: {e}")

            try:
                with open(os.path.join(sample_dir, f"{base_filename}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Question:\n{fixed_question_for_sampling}\n\nGenerated Answer:\n{generated_answer}\n")
                print(f"Saved sample for {base_filename}")
            except Exception as e:
                print(f"Error saving sample text {base_filename}.txt: {e}")

        print("Sample generation finished.")
    # model.train() # Not strictly necessary here as it's the end of main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main training script for fine-tuning SmolVLM models on image-caption datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Dataset and Paths
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images and corresponding .txt caption files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save the trained model, processor, and samples.")
    parser.add_argument("--output_name", type=str, required=True, help="Specific name for this training run's output directory (created within output_dir).")
    
    # Image Bucketing and Tokenization
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="Minimum resolution (longest side) for image bucketing during preprocessing.")
    parser.add_argument("--max_bucket_reso", type=int, default=1536, help="Maximum resolution (longest side) for image bucketing during preprocessing.")
    parser.add_argument("--max_token_length", type=int, default=512, help="Maximum token length for the processor (text sequence length). Inputs will be padded/truncated to this length.")

    # Model and Training Parameters
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceTB/SmolVLM-Instruct", help="Hugging Face model name or path to a local model directory.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device for training.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training type. 'bf16' (recommended for Ampere+ GPUs), 'fp16', or 'no'.")

    # Output and Saving
    parser.add_argument("--save_model_as", type=str, default="safetensors", choices=["ckpt", "pt", "safetensors"], help="Desired format for saving (Note: `save_pretrained` default behavior is used, typically preferring SafeTensors if available).")

    # Sample Generation
    parser.add_argument("--num_sample_answers", type=int, default=5, help="Number of image-answer sample pairs to generate and save after training.")
    parser.add_argument("--sample_max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate for each sample answer during inference.")
    
    # Unused arguments (placeholders for potential future compatibility)
    parser.add_argument("--incremental_reg_reload", action='store_true', help="Reload regularization images incrementally (Not implemented in this script).")
    parser.add_argument("--randomized_regularization_image", action='store_true', help="Randomize regularization images (Not implemented in this script).")

    args = parser.parse_args()
    
    # Construct the final output path for this specific run
    final_output_dir_for_run = os.path.join(args.output_dir, args.output_name)
    if not os.path.exists(final_output_dir_for_run):
        os.makedirs(final_output_dir_for_run, exist_ok=True) # exist_ok=True in case of multi-process race (though not used here)
        print(f"Created output directory for this run: {final_output_dir_for_run}")
    
    # Update args.output_dir to be this specific path for convenience within main()
    args.output_dir = final_output_dir_for_run

    main(args)
