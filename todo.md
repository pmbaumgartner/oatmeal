# TODO

1. Check if post-tokenization inputs are longer than max_seq_len
2. ~~Convert lists to numpy arrays (texts and labels)~~
3. ~~Save paths are weird... gotta be a better way to pass a dir and model name? Look at what other libraries do.~~
    - ~~return save paths from funcs?~~
    - ~~Consistently use `pathlib`.~~
    - I think click's inmput types handle this well.
4. Tune with ray-tune? batch size, epochs, max seq len
    - need to be careful of memory limits here.
5. Evaluation in training loop.
6. ~~Save other model data (num labels, n epochs, max seq len?, multiclass/multilabel column index dict )~~
7. ~~Rename loading to IO? Refactor where functions are in modules.~~
8. Add in predictions dataset name option
9. Reincorporate APEX
10. ~~Return label mapping and allow for custom labels~~
11. ~~Refactor to have only multiclass or multilabel, and take column names?~~
    - ~~Multiclass: They can pass a DF with text labels, we'll enumerate and return them. OR they can pass a DF with pre-enumerated (integers) and then pass an argument with the labels they want.~~
12. Why does data loading take so long (even for small datasets)?
13. ~~Error in multilabel train with up to line 1000? tokenizer not instantiated?~~
    a. calling this done -- was instantiating tokenizer every loop by accident. 
14. Add logging/verbose mode