python3 train.py \
--obstruction_flow_grid_size=tiny \
--obstruction_image_grid_size=medium \
--obstruction_alpha_grid_size=medium \
--obstruction_initial_depth=0.5 \
--transmission_flow_grid_size=tiny \
--transmission_image_grid_size=large \
--transmission_initial_depth=1.0 \
--alpha_weight=1e-2 \
--alpha_temperature=0.1 \
--lr=3e-5 \
"$@"
