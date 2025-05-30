"""
Utility functions for training SmolVLM (and potentially other vision-language) models.
This includes functions for image preprocessing (bucketing, resizing) and
caption handling (loading captions, creating question-answer pairs).
"""
from PIL import Image
import os
import math # Make sure this is at the top of the file

def get_bucket_reso(image_width, image_height, min_reso, max_reso, long_side_step, output_multiple=64):
    """
    Calculates the target resolution for an image based on bucketing parameters.

    Args:
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.
        min_reso (int): Minimum allowed resolution for the initial target longest side.
        max_reso (int): Maximum allowed resolution for the initial target longest side,
                        and also used to define max_pixel_area (max_reso * max_reso).
        long_side_step (int): Step size for initially rounding the longest side.
        output_multiple (int): Ensures final width and height are multiples of this value. Defaults to 64.

    Returns:
        tuple: Target width and height (target_width, target_height) for resizing.
               Returns (0,0) if input dimensions are zero or result in zero.
    """
    if image_width == 0 or image_height == 0:
        return 0, 0

    original_aspect_ratio = image_width / image_height

    # 1. Determine initial target for the longest side
    if image_width > image_height:
        original_longest_side = image_width
    else:
        original_longest_side = image_height
    
    # Round to nearest multiple of long_side_step, then clamp
    target_longest_side = round(original_longest_side / long_side_step) * long_side_step
    target_longest_side = min(max_reso, max(min_reso, target_longest_side))

    # 2. Calculate initial dimensions based on target_longest_side and aspect ratio
    if image_width > image_height:
        calc_width = target_longest_side
        calc_height = int(round(calc_width / original_aspect_ratio))
    else:
        calc_height = target_longest_side
        calc_width = int(round(calc_height * original_aspect_ratio))

    # 3. Area Constraint
    # Use max_reso to define the side of a maximum square area allowed
    max_allowed_area = max_reso * max_reso  
    current_area = calc_width * calc_height

    if current_area > max_allowed_area:
        scale_factor = math.sqrt(max_allowed_area / current_area)
        calc_width = int(round(calc_width * scale_factor))
        calc_height = int(round(calc_height * scale_factor))

    # 4. Round both dimensions to be multiples of output_multiple (e.g., 64)
    # Ensure dimensions are at least output_multiple
    final_width = max(output_multiple, int(round(calc_width / output_multiple)) * output_multiple)
    final_height = max(output_multiple, int(round(calc_height / output_multiple)) * output_multiple)
    
    # 5. Final check for zero dimensions (should be rare with max(output_multiple, ...) but good practice)
    if final_width == 0 or final_height == 0:
        # This case should ideally not be reached if output_multiple > 0 and inputs are valid
        # Fallback to a minimal size if something went extremely wrong
        return output_multiple, output_multiple 
        
    return final_width, final_height


def load_and_preprocess_image(image_path, min_bucket_reso, max_bucket_reso, printf=print):
    """
    Loads an image from the given path, determines its bucket resolution,
    and resizes it to that resolution.

    Args:
        image_path (str): Path to the image file.
        min_bucket_reso (int): Minimum resolution for bucketing (passed to get_bucket_reso).
        max_bucket_reso (int): Maximum resolution for bucketing (passed to get_bucket_reso).
        printf (function): Function to use for printing messages.

    Returns:
        PIL.Image.Image: The processed (resized) PIL Image object in RGB format,
                         or None if loading or processing fails.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB") # Ensure image is in RGB format
        original_width, original_height = img.size

        if original_width == 0 or original_height == 0:
            printf(f"Warning: Image at {image_path} has zero dimension.")
            return None # Invalid image dimensions

        # Determine target bucket dimensions
        bucket_w, bucket_h = get_bucket_reso(
            original_width,
            original_height,
            min_bucket_reso,
            max_bucket_reso,
            long_side_step=384, # Standard step for SmolVLM-like models
            output_multiple=64  # Assuming get_bucket_reso uses this; explicitly passed for clarity
        )

        if bucket_w == 0 or bucket_h == 0:
            printf(f"Warning: Calculated bucket dimensions for {image_path} are zero. Skipping image.")
            return None # Indicates an issue with bucketing result

        # Calculate intermediate resize dimensions to preserve aspect ratio
        img_aspect_ratio = original_width / original_height
        
        # Resize scale ensures the image is large enough to cover the bucket after resizing
        resize_scale = max(bucket_w / original_width, bucket_h / original_height)
        
        resize_w = int(round(original_width * resize_scale))
        resize_h = int(round(original_height * resize_scale))

        # Ensure resize dimensions are not zero
        if resize_w == 0 or resize_h == 0:
            printf(f"Warning: Intermediate resize dimensions for {image_path} are zero. Skipping image.")
            return None

        # Choose resampling filter based on whether intermediate resize is downscaling or upscaling
        # Comparing areas is a robust way to determine this
        if resize_w * resize_h < original_width * original_height:
            resample_filter = Image.LANCZOS # Generally better for downscaling
        else:
            resample_filter = Image.BICUBIC # Generally better for upscaling
            
        resized_img = img.resize((resize_w, resize_h), resample_filter)

        # Center Crop
        crop_x = (resize_w - bucket_w) // 2
        crop_y = (resize_h - bucket_h) // 2
        
        # Ensure crop coordinates are valid (e.g., not negative if resize_w/h < bucket_w/h, though logic above should prevent this)
        # And that the crop box does not exceed the resized image dimensions.
        # This should ideally not happen if resize_scale logic is correct.
        if crop_x < 0 or crop_y < 0 or (crop_x + bucket_w) > resize_w or (crop_y + bucket_h) > resize_h:
             printf(f"Warning: Invalid crop dimensions for {image_path}. Crop box ({crop_x}, {crop_y}, {crop_x + bucket_w}, {crop_y + bucket_h}) vs resized ({resize_w}, {resize_h}). Skipping.")
             return None

        final_img = resized_img.crop((crop_x, crop_y, crop_x + bucket_w, crop_y + bucket_h))
        
        return final_img

    except FileNotFoundError:
        printf(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        # Catch other PIL errors or unexpected issues
        printf(f"Error processing image {image_path}: {e}")
        return None

# --- Caption Handling Functions ---

def get_caption_path(image_path):
    """
    Returns the corresponding caption file path by replacing the image extension with .txt.
    Example: "image.jpg" -> "image.txt"

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Path to the potential caption file.
    """
    base, _ = os.path.splitext(image_path)
    return base + ".txt"

def load_caption(caption_path, printf=print):
    """
    Reads the content of the caption file.

    Args:
        caption_path (str): Path to the caption file.
        printf (function): Function to use for printing messages.

    Returns:
        str: The caption string (stripped of leading/trailing whitespace), 
             or None if the file is not found or another I/O error occurs.
    """
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return caption
    except FileNotFoundError:
        # This is a common case (image without a caption), so often handled silently by process_image_caption_item
        return None
    except Exception as e:
        printf(f"Error loading caption file {caption_path}: {e}")
        return None

def create_qa_pair(caption_text):
    """
    Creates a question-answer pair from the caption text.
    The question is fixed, designed for prompting a captioning model for training data.
    
    The fixed question guides the model on the desired caption format, including
    the use of "@@@@," for static front parts and {keyword1|keyword2} for substitutions.

    Args:
        caption_text (str): The caption text, which will serve as the "answer".

    Returns:
        dict: A dictionary with "question" and "answer" keys.
    
    Future Work:
        This function is a placeholder for more advanced Q&A generation.
        Future enhancements could involve parsing TOML files associated with images
        that specify multiple, varied questions or structured Q&A pairs,
        allowing for more diverse training interactions beyond simple captioning.
    """
    fixed_question = (
        "Caption this image for stable diffusion training as accurately as possible. "
        "Use \"@@@@,\" to separate the front of the caption which will be kept static "
        "except for keyword substitution. Also for items/actions that may have multiple "
        "keywords, use the format {keyword1|keyword2|keyword3} to allow for substitution "
        "during training. Commas are not allowed inside of curly braces."
    )
    return {"question": fixed_question, "answer": caption_text}

def process_image_caption_item(image_path, log_missing_captions=True, printf=print):
    """
    Orchestrates finding, loading, and forming a Q&A pair for a given image.

    Args:
        image_path (str): Path to the image file.
        log_missing_captions (bool): If True, logs a warning when a caption file
                                     is not found for an image. Defaults to True.
        printf (function): Function to use for printing messages (e.g., print or accelerator.print).

    Returns:
        dict: A Q&A pair dictionary (from `create_qa_pair`) if a caption is found
              and successfully processed. Returns None if the caption is missing or
              if any step in the process fails.
    """
    caption_path = get_caption_path(image_path)
    caption_text = load_caption(caption_path, printf=printf) # Pass printf down

    if caption_text is None:
        if log_missing_captions:
            # This helps identify images that might be missing their corresponding text data.
            printf(f"Warning: Caption file not found for {image_path}, skipping image.")
        return None # Indicates no caption available or an error loading it
    
    # If caption text is loaded, create the Q&A pair
    return create_qa_pair(caption_text)
