import torch
from vit import vit_base
from sfp import SimpleFeaturePyramid, LastLevelMaxPool
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class FeatureExtractor(Backbone):
    def __init__(self, cfg, input_shape):
        """
            input_shape: channels of the input image.
        """

        super().__init__()
        self.vit_model = vit_base(img_size=[448])
        self.load_vit_model(path=cfg.MODEL.VIT.WEIGHTS, device=cfg.MODEL.DEVICE)
        self.model_backbone = SimpleFeaturePyramid(
            net=self.vit_model,
            in_feature=cfg.MODEL.VIT.OUT_FEATURES,
            out_channels=cfg.MODEL.SFP.OUT_CHANNELS,
            scale_factors=cfg.MODEL.SFP.OUT_SCALES,
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=448,
        )

    def forward(self, image):
        return self.model_backbone(image)

    def load_vit_model(self, path, device):
       ckpt = torch.load(path, map_location=device)
       state_dict = {k.partition('module.')[2]: v for k,v in ckpt.items()}
       self.model.load_state_dict(state_dict) 

