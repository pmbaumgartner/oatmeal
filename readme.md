# Oatmeal

![BERT eating Oatmeal](https://i.postimg.cc/0NgG7BZ9/image.png)

I needed something to create single sentence and document classifiers using BERT.

Given a binary prediction problem with training dataset like:

```bash
$ head train.csv
texts, labels
"the sandwiches are good", 1
"I found the food disappointing", 0
...
```

Train a model with:
```
python oatmeal.py train binary --input-data="train.csv"
```

some other helpful commands:
```
oatmeal.py --input-data INPUT_DATA [--model-path MODEL_PATH] [--model-name MODEL_NAME]
```

by default, it'll save your model to a timestamped directory with the model type.

## Predictions

*TODO*


## Model Types

You can currently train binary, multiclass, and multilabel. 

**Binary**

```
$ head binary-output.csv
id, label
0, 0.29349324
1, 0.8934850
```

**Multiclass & Multilabel**

`if multiclass: sum(labels) == 1`

```
$ head multilabel-output.csv
id, label0, label1, ..., labelk
0, 0.3943, 0.8984, ..., 0.3481
1, 0.8293, 0.1883, ..., 0.934
```
