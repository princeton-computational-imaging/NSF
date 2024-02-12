python3 train.py \
--obstruction_flow_grid_size=tiny \
--obstruction_image_grid_size=large \
--obstruction_alpha_grid_size=tiny \
--obstruction_initial_depth=2.0 \
--transmission_flow_grid_size=tiny \
--transmission_image_grid_size=large \
--transmission_initial_depth=1.0 \
--alpha_weight=5e-3 \
--alpha_temperature=0.1 \
--lr=2e-4 \
--translation_weight=1e-1 \
"$@"
