import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline
from base_probe import BaseProbe


class InpaintProbe(BaseProbe):
    def run(self, num_images=None, debug=False):
        """
        Generate inpainted images using base SD1.4 images and a fixed-size central mask.
        If debug=True, draw a red outline around the mask region and save to a debug folder.
        """
        prompts = self._load_prompts(num_images)

        # ---- Load inpainting pipeline ----
        print("🔹 Loading StableDiffusionInpaintPipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.pipeline_path, torch_dtype=torch.float16
        ).to(self.device)
        self.pipe.safety_checker = None

        # ---- Load base images ----
        base_images_root = getattr(self.config, "base_images_path", None)
        if base_images_root is None:
            base_images_root = "/share/u/kevin/DiffusionConceptErasure/new_probe_code/results/base_model"

        base_images_path = os.path.join(base_images_root, self.concept, "standardpromptprobe")
        if not os.path.exists(base_images_path):
            raise FileNotFoundError(f"Base images folder not found: {base_images_path}")

        base_images = self._load_images_from_folder(base_images_path)

        # ---- Make central mask ----
        mask_size = getattr(self.config, "mask_size", (256, 256))  # (width, height)
        mask, mask_bounds = self._make_center_mask(base_images[0].size, mask_size)

        # ---- Prepare debug directory ----
        debug_dir = os.path.join(self.output_dir, "debug")
        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        # ---- Generate inpainted images ----
        for i, (prompt, seed) in tqdm(
            enumerate(prompts), total=len(prompts), desc=f"Inpaint {self.concept}"
        ):
            generator = torch.manual_seed(int(seed))
            base_image = base_images[i % len(base_images)]

            image = self.pipe(
                prompt=prompt,
                image=base_image,
                mask_image=mask,
                generator=generator,
            ).images[0]

            # 🔹 Save normal version
            filename = f"{self.concept}_{i:03d}_seed{seed}_inpaint.png"
            self.save_image(image, filename)

            # 🔹 Save debug version (with red outline)
            if debug:
                debug_image = image.copy()
                draw = ImageDraw.Draw(debug_image)
                left, top, right, bottom = mask_bounds
                outline_offset = 3
                draw.rectangle(
                    [
                        left - outline_offset,
                        top - outline_offset,
                        right + outline_offset,
                        bottom + outline_offset,
                    ],
                    outline="red",
                    width=4,
                )
                debug_path = os.path.join(debug_dir, filename)
                debug_image.save(debug_path)

        print(f"✅ Saved {len(prompts)} inpainted images to {self.output_dir}")
        if debug:
            print(f"🪶 Debug images (with red outline) saved to: {debug_dir}")

    # ----------------------------------------------------------------------
    def _load_images_from_folder(self, folder_path):
        exts = (".png", ".jpg", ".jpeg")
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])
        if not files:
            raise FileNotFoundError(f"No images found in {folder_path}")
        return [Image.open(os.path.join(folder_path, f)).convert("RGB") for f in files]

    def _make_center_mask(self, image_size, mask_size):
        """
        Create a mask with a centered WHITE rectangle (to inpaint) surrounded by black background.
        Also returns the bounding box coordinates of the mask for optional debug drawing.
        """
        w, h = image_size
        mask_w, mask_h = mask_size

        left = (w - mask_w) // 2
        top = (h - mask_h) // 2
        right = left + mask_w
        bottom = top + mask_h

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        return mask, (left, top, right, bottom)
