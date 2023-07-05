import torch
from .vit import vit_base
from .sfp import SimpleFeaturePyramid, LastLevelMaxPool
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class FeatureExtractor(Backbone):
    def __init__(self, cfg, input_shape):
        """
            input_shape: channels of the input image.
        """

        super().__init__()
        self.vit_model = vit_base(img_size=[448])
        self.load_vit_model(path=cfg.MODEL.WEIGHTS, device=cfg.MODEL.DEVICE)
        self.model_backbone = SimpleFeaturePyramid(
            net=self.vit_model,
            in_feature="last_feat",
            out_channels=256,
            scale_factors=[4.0,2.0,1.0,0.5],
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=448,
        )

    def forward(self, image):
        print(image.shape)
        return self.model_backbone(image)

    def load_vit_model(self, path, device):
       ckpt = torch.load(path, map_location=device)
       state_dict = {k.partition('module.')[2]: v for k,v in ckpt.items()}
       self.vit_model.load_state_dict(state_dict) 

    def output_shape(self):
        return self.model_backbone.output_shape()