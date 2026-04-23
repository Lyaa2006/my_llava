import os
from .clip_encoder import CLIPVisionTower


LOCAL_VISION_TOWER_MAP = {
    "openai/clip-vit-large-patch14-336": "/mnt/lyaa/MCITlib/clip-vit-large-patch14-336",
}


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower = LOCAL_VISION_TOWER_MAP.get(vision_tower, vision_tower)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
