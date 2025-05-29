"""
Utility functions for training SmolVLM (and potentially other vision-language) models.
This includes functions for image preprocessing (bucketing, resizing) and
caption handling (loading captions, creating question-answer pairs).
"""
from PIL import Image
import os

def get_bucket_reso(image_width, image_height, min_reso, max_reso, long_side_step):
    """
    Calculates the target resolution for an image based on bucketing parameters,
    aiming to fit one side to a multiple of `long_side_step` while maintaining aspect ratio.

    Args:
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.
        min_reso (int): Minimum allowed resolution for the longest side.
        max_reso (int): Maximum allowed resolution for the longest side.
        long_side_step (int): Step size for rounding the longest side. The longest side
                              of the image will be rounded to the nearest multiple of this value.

    Returns:
        tuple: Target width and height (target_width, target_height) for resizing.
               Returns (0,0) if input dimensions are zero.
    """
    if image_width == 0 or image_height == 0:
        # Avoid division by zero for invalid image dimensions
        return 0, 0

    # Determine longest and shortest sides
    if image_width > image_height:
        longest_side = image_width
        shortest_side = image_height
    else:
        longest_side = image_height
        shortest_side = image_width

    # Calculate target resolution for the longest side, clamped within min/max_reso
    target_longest_side = round(longest_side / long_side_step) * long_side_step
    target_longest_side = min(max_reso, max(min_reso, target_longest_side))

    # Calculate aspect ratio and corresponding shortest side
    if longest_side == 0: 
        aspect_ratio = 1 # Avoid division by zero if original longest side was 0 (though caught above)
    else:
        aspect_ratio = shortest_side / longest_side
    
    target_shortest_side = int(round(target_longest_side * aspect_ratio))

    # Ensure the shorter side is a multiple of a reasonable value (e.g., 64)
    # This can be important for some model architectures to avoid unexpected behavior.
    if target_shortest_side > 0:
         target_shortest_side = max(64, round(target_shortest_side / 64) * 64)
    else: # If target_shortest_side became 0 due to extreme aspect ratio and rounding
        target_shortest_side = 64 # Default to a minimum viable dimension like 64


    # Assign calculated target dimensions back based on original orientation
    if image_width > image_height:
        target_width = target_longest_side
        target_height = target_shortest_side
    else:
        target_width = target_shortest_side
        target_height = target_longest_side
        
    # Final check to ensure dimensions are not zero if inputs were not zero.
    # This handles edge cases where rounding or extreme aspect ratios might result in zero.
    if image_width > 0 and image_height > 0:
        if target_width == 0:
            target_width = long_side_step if image_width > image_height else 64
        if target_height == 0:
            target_height = long_side_step if image_height >= image_width else 64

    return int(target_width), int(target_height)


def load_and_preprocess_image(image_path, min_bucket_reso, max_bucket_reso):
    """
    Loads an image from the given path, determines its bucket resolution,
    and resizes it to that resolution.

    Args:
        image_path (str): Path to the image file.
        min_bucket_reso (int): Minimum resolution for bucketing (passed to get_bucket_reso).
        max_bucket_reso (int): Maximum resolution for bucketing (passed to get_bucket_reso).

    Returns:
        PIL.Image.Image: The processed (resized) PIL Image object in RGB format,
                         or None if loading or processing fails.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB") # Ensure image is in RGB format
        original_width, original_height = img.size

        if original_width == 0 or original_height == 0:
            print(f"Warning: Image at {image_path} has zero dimension.")
            return None # Invalid image dimensions

        # Determine target dimensions using bucketing logic
        target_width, target_height = get_bucket_reso(
            original_width,
            original_height,
            min_bucket_reso,
            max_bucket_reso,
            long_side_step=384 # Standard step for SmolVLM-like models
        )

        if target_width == 0 or target_height == 0:
            print(f"Warning: Calculated target dimensions for {image_path} are zero. Skipping resize.")
            return None # Indicates an issue with bucketing result

        # Choose resampling filter based on whether downscaling or upscaling
        if target_width < original_width or target_height < original_height:
            resample_filter = Image.LANCZOS # Generally better for downscaling
        else:
            resample_filter = Image.BICUBIC # Generally better for upscaling
            
        processed_image = img.resize((target_width, target_height), resample_filter)
        return processed_image

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        # Catch other PIL errors or unexpected issues
        print(f"Error processing image {image_path}: {e}")
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

def load_caption(caption_path):
    """
    Reads the content of the caption file.

    Args:
        caption_path (str): Path to the caption file.

    Returns:
        str: The caption string (stripped of leading/trailing whitespace), 
             or None if the file is not found or another I/O error occurs.
    """
    try:
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return caption
    except FileNotFoundError:
        # This is a common case (image without a caption), so often handled silently
        return None
    except Exception as e:
        print(f"Error loading caption file {caption_path}: {e}")
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

def process_image_caption_item(image_path, log_missing_captions=True):
    """
    Orchestrates finding, loading, and forming a Q&A pair for a given image.

    Args:
        image_path (str): Path to the image file.
        log_missing_captions (bool): If True, logs a warning when a caption file
                                     is not found for an image. Defaults to True.

    Returns:
        dict: A Q&A pair dictionary (from `create_qa_pair`) if a caption is found
              and successfully processed. Returns None if the caption is missing or
              if any step in the process fails.
    """
    caption_path = get_caption_path(image_path)
    caption_text = load_caption(caption_path)

    if caption_text is None:
        if log_missing_captions:
            # This helps identify images that might be missing their corresponding text data.
            print(f"Warning: Caption file not found for {image_path}, skipping image.")
        return None # Indicates no caption available or an error loading it
    
    # If caption text is loaded, create the Q&A pair
    return create_qa_pair(caption_text)
