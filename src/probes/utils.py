import torchvision.models as models
import torch

# Optional global cache so the model loads only once
_RESNET_CACHE = None

def rank_with_resnet_in_memory(images, concept=None):
    """
    Given a list of PIL images, ranks them using a pretrained ResNet-50
    and returns the image most likely to represent the given concept.

    Args:
        images (list[PIL.Image]): List of candidate images.
        concept (str): Concept name (e.g. 'golf_ball', 'airliner').

    Returns:
        (best_image, best_prob)
    """
    global _RESNET_CACHE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load or reuse cached model ----
    if _RESNET_CACHE is None:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights).to(device).eval()
        labels = weights.meta["categories"]
        preprocess = weights.transforms()
        _RESNET_CACHE = (resnet, labels, preprocess)

    resnet, labels, preprocess = _RESNET_CACHE

    # ---- Preprocess images ----
    batch = torch.stack([preprocess(img.convert("RGB")) for img in images]).to(device)

    # ---- Forward pass ----
    with torch.no_grad():
        logits = resnet(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)  # [N, 1000]

    # ---- Concept matching ----
    if concept is None:
        raise ValueError("Concept must be provided when calling rank_with_resnet_in_memory().")

    concept_lower = concept.replace("_", " ").lower()
    match_indices = [
        i for i, lbl in enumerate(labels)
        if any(word in lbl.lower() for word in concept_lower.split())
    ]
    if not match_indices:
        print(f"⚠️ No ImageNet classes matched concept '{concept_lower}' — check spelling.")
        return images[0], 0.0  # fallback

    # ---- Aggregate probabilities and pick best ----
    concept_probs = probs[:, match_indices].sum(dim=1)
    best_idx = torch.argmax(concept_probs).item()
    best_prob = concept_probs[best_idx].item()
    best_image = images[best_idx]

    print(f"\n🏆 Best in-memory image for '{concept}': idx={best_idx} (p={best_prob:.4f})")
    return best_image, best_prob
