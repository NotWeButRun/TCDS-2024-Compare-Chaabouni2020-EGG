/root/.local/bin/poetry run python -m egg.zoo.compo_vs_generalization.train \
    --n_values=4 --n_attributes=4 --vocab_size=200 \
    --max_len=16 --batch_size=5120 \
    --sender_cell=lstm --receiver_cell=lstm --random_seed=1