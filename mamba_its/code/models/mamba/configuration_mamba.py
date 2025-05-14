"""mambaVision model configuration"""

from transformers import PretrainedConfig

class MambaConfig(PretrainedConfig):
    model_type = "mamba"

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=12,
        num_channels=3,
        image_size=(256, 320),
        patch_size=16,
        ssm_type="mamba",
        window_size=7,
        patch_merge=True,
        use_local_ssm=True,
        use_global_ssm=True,
        window_interaction=True,
        num_classes=8,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.ssm_type = ssm_type
        self.window_size = window_size
        self.patch_merge = patch_merge
        self.use_local_ssm = use_local_ssm
        self.use_global_ssm = use_global_ssm
        self.window_interaction = window_interaction
        self.num_classes = num_classes
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

logger = logging.get_logger(__name__)