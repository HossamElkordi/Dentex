import torch
from .vit import vit_base
from .sfp import SimpleFeaturePyramid, LastLevelMaxPool
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

def load_vit_model(vit_model, path, device):
       ckpt = torch.load(path, map_location=device)
       state_dict = {k.partition('module.')[2]: v for k,v in ckpt.items()}
       vit_model.load_state_dict(state_dict) 

@BACKBONE_REGISTRY.register()
class FeatureExtractor(Backbone):
    def __init__(self, cfg, input_shape):
        """
            input_shape: channels of the input image.
        """

        super().__init__()
        vit_model = vit_base(img_size=[448])
        load_vit_model(vit_model, path='/content/drive/MyDrive/target_encoder.pth.tar', device=cfg.MODEL.DEVICE)
        self.model_backbone = SimpleFeaturePyramid(
            net=vit_model,
            in_feature="last_feat",
            out_channels=256,
            scale_factors=[4.0,2.0,1.0,0.5],
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=448,
        )

    def forward(self, image):
        return self.model_backbone(image)

    def output_shape(self):
        return self.model_backbone.output_shape()