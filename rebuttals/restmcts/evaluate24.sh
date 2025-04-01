  python evaluate24.py \
  --propose_method "gpt" \
  --value_method "gpt" \
  --mode "mcts" \
  --evaluate "scibench" \
  --iteration_limit 10 \
  --use_reflection "simple" \
  --branch 5\
  --low 0.0\
  --high 60.0