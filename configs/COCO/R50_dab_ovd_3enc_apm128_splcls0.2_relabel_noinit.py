lr = 1e-4
lr_linear_proj_names = []
lr_linear_proj_mult = 0.1
lr_backbone = 0.0
batch_size = 4
weight_decay = 1e-4
epochs = 35
lr_drop = 35
clip_max_norm = 0.1  # gradient clipping max norm
text_len = 15
no_auto_resume = False
save_best = True
model_ema = True
model_ema_decay = 0.99996
no_model_ema = False

# Model parameters
frozen_weights = (
    None  # Path to the pretrained model. If set, only the mask head will be trained
)
multiscale = False
stage1_box = 300
use_deformable_attention = False
only_deform_enc = False
backbone = "clip_RN50"
model_path = ""
dilation = False  # If true, we replace stride with dilation in the last convolutional block (DC5)
position_embedding = (
    "sine"  # Type of positional embedding to use on top of the image features
)
ovd = True
no_clip_init_image = False
pe_temperature = 20
box_conditioned_pe = False
box_proj_dim = 128
pe_proj_dim = -1
region_prompt_path = "<path_to_region_prompt>"
target_class_factor = 8.0
resample_factor = 1.0
filter_classes = False
eval_embedding = ""
rpn = False
matching_threshold = -1.0
pseudo_box = ""
backbone_feature = "layer3"  # choices=['layer3', 'layer4']
end2end = False
disable_init = True
disable_spatial_attn_mask = False

# ovd control flags
use_nms = True
no_nms = False
iou_rescore = False
eval_gt = False
no_target_eval = False

anchor_pre_matching = True
aggresive_eval = False
global_topk = False
softmax_along = "class"  # choices=['class', 'box', 'none']
no_efficient_pooling = False
use_efficient_pe_proj = False
text_dim = 1024
add_gn = False
bg_threshold = 1.0
class_oracle = False
score_threshold = 2.0
classifier_cache = ""

# debug
visualize = False
objectness_alpha = 1.0
split_class_p = 0.2
eval_tau = 100
iou_relabel_eval = False
test_attnpool_path = ""

# * Transformer
enc_layers = 3  # Number of encoding layers in the transformer
dec_layers = 6  # Number of decoding layers in the transformer
dim_feedforward = 1024  # dimension of the FFN in the transformer
hidden_dim = 256  # dimension of the transformer
dropout = 0.1  # Dropout applied in the transformer
nheads = 8  # Number of attention heads in the transformer attention
num_queries = 1000  # Number of query slots
fix_reference_points = False
enable_mem_efficient_sdp = False
enable_flash_sdp = True
enable_math_sdp = True

# * Segmentation
masks = False  # Train segmentation head if True

# Loss
aux_loss = True  # Evable auxiliary decoding losses (loss at each layer)

# * Matcher
set_cost_class = 2.0  # Class coefficient in the matching cost
set_cost_class_rpn = 2.0  # Class coefficient in the matching cost
set_cost_bbox = 5.0  # L1 box coefficient in the matching cost
set_cost_giou = 2.0  # giou box coefficient in the matching cost

# * Loss coefficients
mask_loss_coef = 1.0
dice_loss_coef = 1.0
cls_loss_coef = 2.0
cls_loss_coef_rpn = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
focal_alpha = 0.25
contrastive_loss_coef = 1.0
class_group = ""
semantic_cost = -1.0
topk_matching = -1

# dataset parameters
dataset_file = "coco"
coco_path = "<path_to_coco_dataset>"
lvis_path = ""
label_map = False
coco_panoptic_path = ""
remove_misclassified = True
repeat_factor_sampling = False
repeat_threshold = 0.001
condition_on_text = True
condition_bottleneck = 128

output_dir = ""  # path where to save, empty for no saving
device = "cuda:0"  # device to use for training / testing. We must use cuda.
seed = 42
resume = (
    "<path_to_checkpoint>"  # resume from checkpoint, empty for training from scratch
)
start_epoch = 0
eval = False
eval_target = False
eval_every_epoch = 1
save_every_epoch = 50
num_workers = 2
debug = (
    False  # For debug only. It will perform only a few steps during trainig and val.
)
label_version = "RN50x4base"  # choices=['', 'RN50x4base', 'RN50x4base_coconames', 'RN50x4base_prev', 'RN50base', 'ori', 'custom']
num_label_sampled = -1
clip_aug = False

# distributed training parameters
world_size = 1  # number of distributed processes
dist_url = "env://"  # url used to set up distributed training
