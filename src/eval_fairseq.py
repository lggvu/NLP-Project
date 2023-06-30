fairseq-generate data-bin/pho-mt.en-vi \
    --path checkpoints/checkpoint71.pt \
    --batch-size 128 --beam 5 --remove-bpe