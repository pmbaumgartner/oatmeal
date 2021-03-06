# Oatmeal

![BERT eating Oatmeal](https://i.postimg.cc/0NgG7BZ9/image.png)

I needed something to create single sentence and document classifiers using BERT.

Given a binary prediction problem with training dataset like:

```bash
$ head train.csv
texts,labels
"the sandwiches are good",1
"I found the food disappointing",0
...
```

Train a model with:
```
python oatmeal.py train binary --input-data="train.csv"
```

some other helpful arguments:
```
[--model-path MODEL_PATH] [--model-name MODEL_NAME]
```

by default, it'll save your model to a timestamped directory with the classification type as the model name.

## Predictions

Once you've trained your model, to predict:

```
python oatmeal.py predict --input-data="test.csv" --model-path="your_model_path" --model-name="your_model_name"
```

## Model Types

You can train binary, multiclass, and multilabel. 

**Binary**

```
$ head binary-output.csv
id,label
0,0.2934
1,0.8934
```

**Multiclass & Multilabel**

`if multiclass: sum(labels) == 1`

```
$ head multilabel-output.csv
id,label0,label1,...,labelk
0,0.3943,0.8984,...,0.3481
1,0.8293,0.1883,...,0.9342
```
