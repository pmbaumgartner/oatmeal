# TODO

1. Check if post-tokenization inputs are longer than max_seq_len24. Tune with ray-tune? batch size, epochs, max seq len
    - need to be careful of memory limits here.
2. Evaluation in training loop.
3. Add in predictions dataset name option
4. Reincorporate APEX
5. Add logging/verbose mode


## TODONE

- ~~Convert lists to numpy arrays (texts and labels)~~
- ~~Save other model data (num labels, n epochs, max seq len?, multiclass/multilabel column index dict )~~
- ~~Save paths are weird... gotta be a better way to pass a dir and model name? Look at what other libraries do.~~
- ~~Rename loading to IO? Refactor where functions are in modules.~~
- ~~Return label mapping and allow for custom labels~~
- ~~Refactor to have only multiclass or multilabel, and take column names?~~
- ~~Why does data loading take so long (even for small datasets)?~~ / ~~Error in multilabel train with up to line 1000? tokenizer not instantiated?~~