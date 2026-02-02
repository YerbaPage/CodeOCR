import os
import json
import shutil
import sys

from .config import (
    OUTPUT_DIR, DATASET_DIR, IMAGES_DIR, DATASET_FILENAME, 
    DEFAULT_DATASET_FILENAME, TARGET_RATIOS, USE_EXISTING_IMAGES
)
from .utils import _remove_file_if_exists

def apply_visual_corruption(image_path, ratio):
    """
    Manual visual corruptor: load the source image, downsample by ratio, then upsample back to original size.
    Convention: for any ratio in 1/1.5/2/4/6/8, create a new file with suffix _ratio{ratio}.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # Build new filename: page_001.png -> page_001_ratio1.png
            dir_name = os.path.dirname(image_path)
            base_name = os.path.basename(image_path)
            name_part, ext = os.path.splitext(base_name)
            new_filename = f"{name_part}_ratio{ratio}{ext}"
            new_path = os.path.join(dir_name, new_filename)
            
            if ratio == 1:
                # ratio=1: save original without compression, but rename to _ratio1
                img.save(new_path)
                return new_path
            
            # ratio>1: apply downsample/upsample
            original_w, original_h = img.size
            new_w = max(1, int(original_w / ratio))
            new_h = max(1, int(original_h / ratio))
            
            # Downsample then upsample back to original size
            small_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_img = small_img.resize((original_w, original_h), Image.Resampling.BILINEAR)
            resized_img.save(new_path)
            return new_path
            
    except Exception as e:
        print(f"   ⚠️ Compression failed for ratio {ratio}: {e}")
        return None


def run_module_1_and_2():
    if USE_EXISTING_IMAGES:
        print("🧩 Using existing images (skip Module 1 & 2)")
        print(f"   - images_dir: {os.path.abspath(IMAGES_DIR)}")
        print(f"   - dataset: {os.path.abspath(os.path.join(OUTPUT_DIR, DATASET_FILENAME))}")
        return

    # 0. Cleanup (only delete the current model's images directory)
    if os.path.exists(IMAGES_DIR):
        try:
            shutil.rmtree(IMAGES_DIR)
            print(f"🧹 Cleaned up: {IMAGES_DIR}")
        except Exception as e:
            print(f"⚠️ Failed to clean {IMAGES_DIR}: {e}")
            
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # -------------------------------------------------
    # 🟢 Module 1: Data Miner
    # -------------------------------------------------
    print("\n" + "="*40)
    print("🚀 Running Module 1: Data Miner")
    print("="*40)

    try:
        # Assuming data_miner is in the python path (reconstruction folder)
        # We might need to adjust sys.path if this is run as a package
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from data_miner import fetch_fresh_code
    except Exception as e:
        print("❌ Failed to import data_miner.fetch_fresh_code.")
        print(f"   - error: {e}")
        print("   If you only want to evaluate an existing image set, set USE_EXISTING_IMAGES=1.")
        print("   If you want to run the full pipeline (fetch GitHub code), install dependencies first: pip install PyGithub")
        return
    
    dataset = fetch_fresh_code()
    
    if not dataset:
        print("❌ No data found.")
        return

    dataset_path = os.path.join(DATASET_DIR, DATASET_FILENAME)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------
    # 🔵 Module 2: Visual Corruptor
    # -------------------------------------------------
    print("\n" + "="*40)
    print("🚀 Running Module 2: Visual Corruptor")
    print(f"🎯 Target Ratios: {TARGET_RATIOS}")
    print("="*40)

    total_images_generated = 0
    
    for idx, item in enumerate(dataset):
        code_id = item['id']
        source_code = item['code']
        
        print(f"[{idx+1}/{len(dataset)}] Processing: {code_id} ...")
        
        item_output_dir = os.path.join(IMAGES_DIR, code_id)
        os.makedirs(item_output_dir, exist_ok=True)
        
        temp_file_path = os.path.join(item_output_dir, "temp_source.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(source_code)
            
        try:
            # 1. Generate baseline high-quality images (1x)
            import text_to_image
            generated_paths = text_to_image.generate_images_for_file(
                filename=temp_file_path,
                source_code=source_code,
                base_output_dir=item_output_dir,
                width=1024,
                height=1024,  # Square
                font_size=18,  # Slightly smaller to fit the square canvas
                line_height=1.2,
                dpi=100,
                preserve_newlines=True,
                enable_syntax_highlight=True,
                unique_id="base"
            )
            
            if not generated_paths:
                print("   ❌ No base image generated.")
                continue

            # 2. Run visual compression loop (1x, 2x, 4x, 8x)
            for original_path in generated_paths:
                for ratio in TARGET_RATIOS:
                    new_path = apply_visual_corruption(original_path, ratio)
                    if new_path:
                        total_images_generated += 1
                
                # 🗑️ Delete the original image (all ratio variants are generated)
                try:
                    if os.path.exists(original_path):
                        os.remove(original_path)
                except Exception as e:
                    print(f"      ⚠️ Failed to remove original: {e}")

        except Exception as e:
            print(f"   ❌ Error processing {code_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    print("\n" + "="*40)
    print("🎉 Pipeline Stage 1 & 2 Completed!")
    print(f"📊 Summary:")
    print(f"   - Data Mined: {len(dataset)}")
    print(f"   - Total Variants Generated: {total_images_generated}")
    print(f"   - Output Location: {os.path.abspath(OUTPUT_DIR)}")
    print("="*40)
