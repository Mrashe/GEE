python launch.py --config configs/edit-n2n.yaml --train --gpu 0 trainer.max_steps=2000 system.prompt_processor.prompt="Turn him into Albert Einstein"  system.max_densify_percent=0.03 system.anchor_weight_init_g0=0.0 system.anchor_weight_init=0.02 system.anchor_weight_multiplier=1.3 system.seg_prompt="man" system.loss.lambda_anchor_color=5 system.loss.lambda_anchor_geo=50 system.loss.lambda_anchor_scale=50 system.loss.lambda_anchor_opacity=50 system.densify_from_iter=100 system.densify_until_iter=5000 system.densification_interval=300 data.source=/home/xshe/4DGaussians/cook0 system.gs_source=/home/xshe/4DGaussians/cook0/  system.loggers.wandb.enable=true system.loggers.wandb.name="edit_n2n_face_Ein"
python launch.py --config configs/edit-ctn.yaml --train --gpu 0 trainer.max_steps=1600 system.cache_dir="ground_fire" system.seg_prompt="grass" system.prompt_processor.prompt="make the whole scene on fire"  system.max_densify_percent=0.0025 system.anchor_weight_init_g0=1 system.anchor_weight_init=0.5 system.anchor_weight_multiplier=1.5  system.loss.lambda_anchor_color=0 system.loss.lambda_anchor_geo=5 system.loss.lambda_anchor_scale=5 system.loss.lambda_anchor_opacity=0 system.densify_from_iter=100 system.densify_until_iter=5000 system.densification_interval=300 data.source=/home/linguosheng/gsedit/bicycle system.gs_source=/home/linguosheng/gsedit/point_cloud.ply  system.loggers.wandb.enable=true system.loggers.wandb.name="edit_ctn_bike_fire"
