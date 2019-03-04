# TODO

1. Check if post-tokenization inputs are longer than max_seq_len
2. ~~Convert lists to numpy arrays (texts and labels)~~
3. Save paths are weird... gotta be a better way to pass a dir and model name? Look at what other libraries do.
    - return save paths from funcs?
    - Consistently use `pathlib`.
4. Tune with ray-tune? Not a lot of vars, maybe epochs? 
5. Evaluation in training loop.
6. ~~Save other model data (num labels, n epochs, max seq len?, multiclass/multilabel column index dict )~~
7. ~~Rename loading to IO? Refactor where functions are in modules.~~
8. Add in predictions dataset name option