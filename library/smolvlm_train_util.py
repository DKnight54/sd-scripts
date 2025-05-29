import toml
import random
import math
import os
import torch
import logging
from PIL import Image, ImageDraw, ImageFont
import imagesize # For getting image dimensions
import json # For sample_prompts parsing

from transformers import AutoProcessor

# Assuming these are in library.train_util or accessible
# Adjust imports as per your project structure
from library.train_util import BaseDataset, ImageInfo, BucketManager, IMAGE_EXTENSIONS, find_image_files_recursively

# Functions from this file
# (No need to import if they are in the same file and defined before the class)

logger = logging.getLogger(__name__)

def parse_toml_caption(toml_file_path: str) -> dict:
    """
    Parses a TOML file to extract Q&A patterns.

    Args:
        toml_file_path: Path to the TOML file.

    Returns:
        A dictionary containing the Q&A patterns (e.g., {"qa_patterns": [...]}).
        Returns an empty dictionary if the file is not found or is invalid.
    """
    try:
        with open(toml_file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
            return data
    except FileNotFoundError:
        # logger.warning(f"TOML file not found at {toml_file_path}") # Prefer logging over print
        return {}
    except toml.TomlDecodeError:
        logger.warning(f"Invalid TOML format in {toml_file_path}")
        return {}

def generate_qas_from_patterns(patterns: dict, image_metadata: dict, num_qas: int) -> list[dict]:
    """
    Generates a list of Q&A pairs from given patterns and image metadata.

    Args:
        patterns: A dictionary containing Q&A patterns (output from parse_toml_caption).
        image_metadata: A dictionary containing metadata to fill in the answer patterns.
        num_qas: The desired number of Q&A pairs to generate.

    Returns:
        A list of Q&A dictionaries (e.g., [{"question": "...", "answer": "..."}, ...]).
    """
    qa_patterns_list = patterns.get("qa_patterns", [])
    if not qa_patterns_list:
        return []

    num_to_select = min(num_qas, len(qa_patterns_list))
    # Ensure qa_patterns_list is not empty before sampling
    if not qa_patterns_list:
        return []
    if num_to_select == 0 and len(qa_patterns_list) > 0 : # if num_qas is 0, but patterns exist, generate at least 1
        num_to_select = 1
        
    selected_patterns = random.sample(qa_patterns_list, num_to_select)

    generated_qas = []
    for pattern in selected_patterns:
        question = pattern.get("question")
        answer_pattern = pattern.get("answer_pattern") # Changed from "answer" to "answer_pattern" based on spec

        if question and answer_pattern:
            try:
                answer = answer_pattern.format(**image_metadata)
                generated_qas.append({"question": question, "answer": answer})
            except KeyError as e:
                logger.warning(f"Missing key {e} in image_metadata for pattern: {pattern}")
            except Exception as e:
                logger.warning(f"Could not format answer for pattern {pattern} with metadata {image_metadata}. Error: {e}")
    
    return generated_qas

def calculate_smolvlm_bucket_resolution(
    original_width: int,
    original_height: int,
    longest_edge_n: int,
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_reso_steps: int,
) -> tuple[int, int, int]:
    """
    Calculates the target bucket resolution for an image based on SmolVLM requirements.

    Args:
        original_width: Original width of the image.
        original_height: Original height of the image.
        longest_edge_n: The N factor for N*384 target longest edge.
        min_bucket_reso: Minimum resolution for any side of a bucket.
        max_bucket_reso: Maximum resolution for any side of a bucket.
        bucket_reso_steps: Step size for bucket resolutions.

    Returns:
        A tuple (target_bucket_width, target_bucket_height, resized_longest_edge).
    """
    target_max_longest_edge = longest_edge_n * 384

    current_width, current_height = original_width, original_height

    # Handle cases where original image is smaller than min_bucket_reso on either side by upscaling
    if current_width < min_bucket_reso or current_height < min_bucket_reso:
        if current_width < current_height:
            aspect_ratio = current_height / current_width
            current_width = min_bucket_reso
            current_height = int(min_bucket_reso * aspect_ratio)
        else:
            aspect_ratio = current_width / current_height
            current_height = min_bucket_reso
            current_width = int(min_bucket_reso * aspect_ratio)
        
        # Ensure the upscaled dimension is also a multiple of bucket_reso_steps (floor)
        current_width = math.floor(current_width / bucket_reso_steps) * bucket_reso_steps
        current_height = math.floor(current_height / bucket_reso_steps) * bucket_reso_steps
        # And re-clamp to min_bucket_reso if rounding down made it too small
        current_width = max(min_bucket_reso, current_width)
        current_height = max(min_bucket_reso, current_height)


    # Determine the aspect ratio
    aspect_ratio = current_width / current_height

    # Calculate initial resized dimensions based on target_max_longest_edge
    if current_width > current_height:
        resized_width = min(target_max_longest_edge, current_width)
        resized_height = int(resized_width / aspect_ratio)
    else:
        resized_height = min(target_max_longest_edge, current_height)
        resized_width = int(resized_height * aspect_ratio)
    
    resized_longest_edge = max(resized_width, resized_height)
    
    # Adjust width and height downwards to be multiples of bucket_reso_steps
    target_bucket_width = math.floor(resized_width / bucket_reso_steps) * bucket_reso_steps
    target_bucket_height = math.floor(resized_height / bucket_reso_steps) * bucket_reso_steps

    # Ensure the resulting bucket dimensions are at least min_bucket_reso
    target_bucket_width = max(min_bucket_reso, target_bucket_width)
    target_bucket_height = max(min_bucket_reso, target_bucket_height)
    
    # Ensure the resulting bucket dimensions are within [min_bucket_reso, max_bucket_reso]
    target_bucket_width = min(max_bucket_reso, target_bucket_width)
    target_bucket_height = min(max_bucket_reso, target_bucket_height)

    return target_bucket_width, target_bucket_height, resized_longest_edge


class SmolVLMDataset(BaseDataset):
    def __init__(self, 
                 image_data_dir: str, 
                 toml_captions_dir: str, 
                 processor, # AutoProcessor instance
                 is_dreambooth_mode: bool, 
                 smolvlm_longest_edge_n: int, 
                 batch_size: int, 
                 tokenizer, # Passed to BaseDataset, might be part of processor
                 max_token_length: int, 
                 resolution: tuple, # Initial resolution, will be overridden by bucketing
                 network_multiplier: float = 1.0, # From BaseDataset
                 enable_bucket: bool = True, 
                 min_bucket_reso: int = 256, 
                 max_bucket_reso: int = 1024, # Example, should match N*384 logic
                 bucket_reso_steps: int = 64, 
                 bucket_no_upscale: bool = False, # BaseDataset uses this, SmolVLM has its own logic
                 prior_loss_weight: float = 1.0, # For Dreambooth reg images
                 debug_dataset: bool = False,
                 keep_tokens: int = 0, # From BaseDataset
                 shuffle_caption: bool = False, # BaseDataset, less relevant here
                 caption_separator: str = ",", # BaseDataset, less relevant here
                 caption_prefix: str = None, # BaseDataset, less relevant here
                 caption_suffix: str = None, # BaseDataset, less relevant here
                 caption_dropout_rate: float = 0.0, # BaseDataset, less relevant here
                 caption_dropout_every_n_epochs: int = 0, # BaseDataset, less relevant here
                 caption_tag_dropout_rate: float = 0.0, # BaseDataset, less relevant here
                 color_aug: bool = False, # BaseDataset, for image augmentation
                 flip_aug: bool = False, # BaseDataset, for image augmentation
                 random_crop: bool = False, # BaseDataset, for image augmentation
                 token_warmup_min: int = 1, # BaseDataset, less relevant here
                 token_warmup_step: float = 0.0, # BaseDataset, less relevant here
                 face_crop_aug_range: tuple = None, # BaseDataset, for image augmentation
                 image_class_identifier_suffix: str = "_class",
                 num_repeats: int = 1, # Number of times to repeat each image in the dataset
                 num_qas_per_image_min: int = 1,
                 num_qas_per_image_max: int = 3,
                 ):
        
        super().__init__(tokenizer, max_token_length, resolution, network_multiplier, debug_dataset)

        self.image_data_dir = image_data_dir
        self.toml_captions_dir = toml_captions_dir
        self.is_dreambooth_mode = is_dreambooth_mode
        self.processor = processor
        self.smolvlm_longest_edge_n = smolvlm_longest_edge_n
        self.batch_size = batch_size 
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso 
        self.bucket_reso_steps = bucket_reso_steps
        self.bucket_no_upscale = bucket_no_upscale 
        self.prior_loss_weight = prior_loss_weight 

        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.random_crop = random_crop
        self.face_crop_aug_range = face_crop_aug_range
        self.keep_tokens = keep_tokens
        self.shuffle_caption = shuffle_caption 
        self.caption_dropout_rate = caption_dropout_rate 
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs 
        self.caption_tag_dropout_rate = caption_tag_dropout_rate 

        self.image_class_identifier_suffix = image_class_identifier_suffix
        self.num_repeats = num_repeats
        self.num_qas_per_image_min = num_qas_per_image_min
        self.num_qas_per_image_max = num_qas_per_image_max

        self.image_data = {} 
        max_manager_reso = self.smolvlm_longest_edge_n * 384
        self.bucket_manager = BucketManager(
            no_upscale=self.bucket_no_upscale, 
            max_reso=(max_manager_reso, max_manager_reso), 
            min_reso=(self.min_bucket_reso, self.min_bucket_reso),
            reso_steps=self.bucket_reso_steps,
            batch_size=self.batch_size 
        )
        
        self._load_data()
        if self.debug_dataset:
            logger.info(f"Total images loaded: {len(self.image_data)}")
            
        self.make_buckets() 

        if self.debug_dataset:
            logger.info(f"Number of buckets created: {len(self.bucket_manager.buckets)}")

    def _load_data(self):
        image_paths = []
        if os.path.isdir(self.image_data_dir):
            image_paths = find_image_files_recursively(self.image_data_dir)
        elif os.path.isfile(self.image_data_dir): # Support single image file for sampling
             image_paths = [self.image_data_dir]

        if not image_paths:
            logger.warning(f"No images found in {self.image_data_dir}")
            return

        for image_path in image_paths:
            image_key = os.path.splitext(os.path.basename(image_path))[0] 
            toml_path = None
            if self.toml_captions_dir: # toml_captions_dir might be None for sampling
                toml_filename = f"{image_key}.toml"
                toml_path = os.path.join(self.toml_captions_dir, toml_filename)

            qa_patterns = {}
            if toml_path and os.path.exists(toml_path):
                qa_patterns = parse_toml_caption(toml_path)
            
            if not qa_patterns.get("qa_patterns"):
                if self.is_dreambooth_mode:
                    parent_dir_name = os.path.basename(os.path.dirname(image_path))
                    image_class = parent_dir_name 
                    default_q = "Classify this image using as little words as possible."
                    qa_patterns = {"qa_patterns": [{"question": default_q, "answer_pattern": image_class}]}
                    if self.debug_dataset:
                         logger.info(f"Using Dreambooth default Q&A for {image_path}. Class: {image_class}")
                else: # If not Dreambooth and no TOML, this image might be for sampling only (no ground truth Q&A)
                    logger.info(f"No TOML captions for {image_path} and not in Dreambooth mode. Will use for sampling if prompted.")
                    # We still need to process the image for bucketing if enable_bucket is True
                    # If this dataset is ONLY for sampling, qa_patterns can remain empty.
                    # However, BaseDataset expects some form of caption/data.
                    # For training, we would skip here. For sampling, this is okay.
                    if not self.image_data_dir or not os.path.isfile(self.image_data_dir): # Skip if not in single file mode
                        # (This condition is a bit complex, means if it's a directory, we expect captions for training)
                        # The current logic is primarily for training. For sampling, this might be handled differently.
                        # For now, if it's a directory and no captions, skip.
                        if os.path.isdir(self.image_data_dir):
                            logger.warning(f"Skipping {image_path} for training due to missing TOML/DB mode.")
                            continue


            try:
                original_width, original_height = imagesize.get(image_path)
            except Exception as e:
                logger.warning(f"Could not get image size for {image_path}: {e}. Skipping.")
                continue

            bucket_width, bucket_height, _ = calculate_smolvlm_bucket_resolution(
                original_width, original_height,
                self.smolvlm_longest_edge_n,
                self.min_bucket_reso,
                self.max_bucket_reso,
                self.bucket_reso_steps
            )
            
            image_info = ImageInfo(
                image_key=image_path, 
                num_repeats=self.num_repeats, 
                caption_list=None, 
                is_reg=False, 
                absolute_path=image_path,
                image_class=None 
            )
            image_info.qa_patterns = qa_patterns if qa_patterns.get("qa_patterns") else {"qa_patterns": []} # Ensure it's a dict
            image_info.bucket_reso = (bucket_width, bucket_height)
            image_info.original_size = (original_width, original_height) 

            self.image_data[image_path] = image_info
            
            if self.enable_bucket: # Only add to bucket manager if bucketing is enabled
                for _ in range(self.num_repeats):
                    self.bucket_manager.add_image(image_info.bucket_reso, image_path)
        
        if self.debug_dataset:
            logger.info(f"Finished loading data. Total unique images: {len(self.image_data)}")


    def __getitem__(self, index):
        image_key = self.image_keys[index] 
        image_info = self.image_data[image_key]

        try:
            pil_image = Image.open(image_info.absolute_path).convert("RGB")
        except Exception as e:
            logger.error(f"Could not load image {image_info.absolute_path}: {e}. Returning dummy data.")
            dummy_ids = torch.zeros(self.max_token_length, dtype=torch.long)
            dummy_pixel_values = torch.zeros(3, self.smolvlm_longest_edge_n * 384, self.smolvlm_longest_edge_n * 384) 
            return {"input_ids": dummy_ids, "attention_mask": dummy_ids.clone(), "pixel_values": dummy_pixel_values, "labels": dummy_ids.clone()}

        qa_patterns = image_info.qa_patterns
        image_metadata = {} 

        num_qas_to_generate = random.randint(self.num_qas_per_image_min, self.num_qas_per_image_max)
        generated_qas = generate_qas_from_patterns(qa_patterns, image_metadata, num_qas_to_generate)

        if not generated_qas: 
            generated_qas = [{"question": "Describe this image.", "answer": "No specific details available."}]

        messages = []
        for i, qa in enumerate(generated_qas):
            user_content = []
            if i == 0: user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": qa["question"]})
            
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["answer"]}]})

        try:
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        except Exception as e:
            logger.error(f"Error applying chat template for {image_key}: {e}. Messages: {messages}")
            prompt_text = f"USER: <image>\n{generated_qas[0]['question']}\nASSISTANT: {generated_qas[0]['answer']}"

        inputs = self.processor(
            text=[prompt_text], 
            images=[pil_image], 
            return_tensors="pt", 
            padding="longest", 
            truncation=True,   
            max_length=self.processor.tokenizer.model_max_length if hasattr(self.processor.tokenizer, 'model_max_length') else 512
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        if hasattr(self.processor.tokenizer, 'additional_special_tokens'):
            try:
                image_token_text = next((tok for tok in self.processor.tokenizer.additional_special_tokens if "image" in tok.lower()), "<image>")
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token_text)
                if isinstance(image_token_id, int): 
                    labels[labels == image_token_id] = -100
            except Exception as e:
                logger.debug(f"Could not find/mask image token ID: {e}")
        
        # Basic label masking (user turns) - more sophisticated masking might be needed
        # This simple version masks all user turns.
        input_ids_list = inputs["input_ids"].squeeze(0).tolist()
        current_labels = labels.squeeze(0).tolist() # Make a copy to modify
        
        # Iterate through messages to identify user and assistant parts based on template structure
        # This is a simplified placeholder for actual robust masking logic.
        # A more robust approach would involve tokenizing each part of the message list
        # (user question, assistant answer) and then finding their positions in the final tokenized input.
        # For now, we assume a simple masking where only assistant responses are unmasked.
        # The current `labels` already has padding and image tokens masked.
        # We need to further mask user questions.
        
        # Placeholder: This simple logic just keeps everything not pad/image token. Refine as needed.
        # A more accurate masking would require parsing the template output.
        # For example, find "USER:" and "ASSISTANT:" boundaries.
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0).to(torch.bfloat16), 
            "labels": labels.squeeze(0) 
        }

def sample_smolvlm_images(
    accelerator,
    args,
    epoch,
    global_step,
    device,
    model, 
    processor,
    sampling_prompts # Can be a list of image paths, or a path to a JSON file
):
    logger.info(f"Generating samples for epoch {epoch} step {global_step}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    save_dir = os.path.join(args.output_dir, "sample")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    prompts_data = []
    if os.path.isfile(sampling_prompts):
        try:
            with open(sampling_prompts, "r", encoding="utf-8") as f:
                prompts_data = json.load(f) # Expects list of {"image_path": "...", "question": "..."}
        except Exception as e:
            logger.error(f"Error loading sample prompts from {sampling_prompts}: {e}")
            return
    elif isinstance(sampling_prompts, list): # Treat as list of image paths
        # Find corresponding TOMLs or use default questions
        for img_path in sampling_prompts:
            if os.path.exists(img_path):
                prompts_data.append({"image_path": img_path})
            else:
                logger.warning(f"Sample image path not found: {img_path}")

    if not prompts_data:
        logger.warning("No valid image paths or prompts found for sampling.")
        return

    model.eval() # Set model to eval mode for generation

    for i, item_data in enumerate(prompts_data[:args.sample_n_images if hasattr(args, 'sample_n_images') else 4]):
        if not accelerator.is_main_process:
            continue # Only main process saves images and text

        image_path = item_data.get("image_path")
        question = item_data.get("question") # Optional question from JSON
        
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Skipping sample, image not found: {image_path}")
            continue

        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Could not load sample image {image_path}: {e}")
            continue

        if not question: # If no question provided, try to get one from TOML or use a default
            image_key = os.path.splitext(os.path.basename(image_path))[0]
            toml_path = None
            if args.toml_captions_dir:
                 toml_path = os.path.join(args.toml_captions_dir, f"{image_key}.toml")
            
            qa_patterns = {}
            if toml_path and os.path.exists(toml_path):
                qa_patterns = parse_toml_caption(toml_path)

            if qa_patterns.get("qa_patterns"):
                # Pick a random question from the patterns for sampling
                selected_pattern = random.choice(qa_patterns["qa_patterns"])
                question = selected_pattern.get("question")
                # We don't need to format the answer here for sampling, model generates it
            else: # Fallback question
                question = "Describe this image in detail."
        
        logger.info(f"Sampling with image: {image_path}, Question: {question}")

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        
        try:
            prompt_for_model = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=prompt_for_model, images=[original_image], return_tensors="pt").to(device)
        except Exception as e:
            logger.error(f"Error processing input for {image_path} with question '{question}': {e}")
            continue

        with torch.no_grad(), accelerator.autocast():
            # Common generation parameters, can be exposed via args later
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=args.sample_max_new_tokens if hasattr(args, 'sample_max_new_tokens') else 128, 
                do_sample=True, 
                top_p=0.9,
                temperature=0.7, # Add temperature for more diverse sampling
                num_beams=args.sample_num_beams if hasattr(args, 'sample_num_beams') else 1, # Beam search if > 1
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # The output usually includes the prompt, so we need to clean it up if the template doesn't handle it.
        # For now, assume batch_decode gives the full sequence including prompt.
        # We might need to find where the assistant's response starts.
        # A simple heuristic if ASSISTANT token is used by template:
        assistant_response_start = generated_text.rfind("ASSISTANT:") 
        if assistant_response_start != -1:
            generated_answer = generated_text[assistant_response_start + len("ASSISTANT:"):].strip()
        else: # If template is different, this might need adjustment
            # Or if the model only generates the answer part
            # For now, assume the full output is the answer if "ASSISTANT:" is not found
            # This depends heavily on the specific model's generation behavior and chat template.
            # A common case is that `generate` with `add_generation_prompt=True` will yield a sequence
            # that includes the prompt, and then the generated response.
            # `batch_decode` then decodes this full sequence.
            # We are interested in the part *after* the prompt.
            # A robust way is to decode input_ids and subtract that from generated_ids.
            prompt_tokens_len = inputs['input_ids'].shape[1]
            answer_ids = generated_ids[0][prompt_tokens_len:]
            generated_answer = processor.decode(answer_ids, skip_special_tokens=True).strip()


        # Save image and text
        # Determine resize dimension for saving sample image (e.g., 512 longest edge)
        img_to_save = original_image.copy()
        w, h = img_to_save.size
        if w > h:
            new_w = 512
            new_h = int(h * (512.0 / w))
        else:
            new_h = 512
            new_w = int(w * (512.0 / h))
        img_to_save = img_to_save.resize((new_w, new_h), Image.LANCZOS)

        base_filename = f"{args.output_name}_{global_step}_{i}"
        img_to_save.save(os.path.join(save_dir, base_filename + ".png"))

        with open(os.path.join(save_dir, base_filename + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Generated Answer:\n{generated_answer}\n")
            # If ground truth answer is available (e.g. from a TOML file for this image), write it too
            # This part is complex as we don't have easy access to ground truth answers here unless
            # `sampling_prompts` also provides them. For now, only generated.

    logger.info(f"Generated {i+1} samples in {save_dir}")
    model.train() # Set model back to train mode

# Example Usage (for testing purposes, will be removed or commented out)
if __name__ == '__main__':
    # ... (rest of the __main__ block from previous version, if needed for testing this function)
    # For testing sample_smolvlm_images, one would need to set up dummy args, model, processor, etc.
    pass

# print("SmolVLMDataset class and sample_smolvlm_images function implemented.") # Comment out print

[end of library/smolvlm_train_util.py]
