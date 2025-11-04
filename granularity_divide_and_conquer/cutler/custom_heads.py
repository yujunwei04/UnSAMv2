from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CascadeROIHeads):
    """Minimal implementation to satisfy ROI_HEADS.NAME in config.
    Inherit CascadeROIHeads without extra changes.
    """
    pass 