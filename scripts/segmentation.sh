python3 train.py \
--obstruction_flow_grid_size=small \
--obstruction_image_grid_size=large \
--obstruction_alpha_grid_size=medium \
--obstruction_control_points_flow=15 \
--obstruction_initial_depth=0.5 \
--obstruction_initial_alpha=0.5 \
--transmission_flow_grid_size=small \
--transmission_image_grid_size=large \
--transmission_initial_depth=1.0 \
--transmission_control_points_flow=15 \
--camera_control_points=15 \
--alpha_weight=2e-3 \
--alpha_temperature=0.15 \
--translation_weight=1e0 \
"$@"