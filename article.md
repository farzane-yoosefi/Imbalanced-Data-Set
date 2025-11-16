


title: Imbalanced Dataset
---
date: 2024-11-15
---

# How to handle imbalanced datasets

# The Imbalanced Data Problem: When Your Model Only Sees the Majority
### Let's start with an example:

Imagine a model that needs to predict the rare '0' values among 90 frequent '1' values.

![Imbalanced Dataset Example](.data_sample.png)

In this dataset:
- **Minority class**: 0 (only 3 instances!)
- **Majority class**: 1 (90 instances)

## What would our model learn?
Since '1' appears 97% of the time, the model quickly learns to **always predict '1'**. It achieves 97% accuracy by completely ignoring the minority class!

## But here's the problem:
What if those rare '0' values represent:
- **Fraudulent transactions** in banking?
- **Diseased patients** in medical testing?  
- **Defective products** in manufacturing?

Suddenly, that "97% accurate" model becomes **100% useless** for our actual goal!


## solution:

Thanks to data scientists there are several solutions to this problem like :

- **Oversampling the Minority Class**
- **Undersampling the Majority Class**
- **Using Appropriate Evaluation Metrics** : Precision, Recall, and F1-Score
- **Confusion Matrix**
- **Cost-Sensitive Learning**
- **Ensemble Methods**
- **Anomaly Detection**
The above solutions are taken from the GeeksforGeeks website
Now we are going to try oversampling and undersampling 
The key is to use **resample()** from sklearn 
- What resample actually does : **It randomly picks samples and does not know which samples are important.**
The **choice between** oversampling and undersampling **is determined by** the parameters.
**Let's learn about its parameters:**
## Understanding Resample Parameters

Let's examine the key parameters of the `resample` function:

- **`n_samples`**: How many samples you want after resampling
- **`replace`**: Whether to sample with replacement (True/False)
- **`random_state`**: Makes random sampling reproducible

n_sample > original_size --> oversampling

n_sample < originsl_size --> undersampling

## Oversampling :
In this method, the minority class size increases to match the majority class, preventing model bias.
### Biased model: The model which has been trained based on one specific category, which leads to a poor performance on minority class
Please note that we apply oversampling on the minority class:

**Before Oversampling:**
- Minority class (0): 3 samples
- Majority class (1): 87 samples

**After Oversampling:**

- Minority class (0): 90 samples  (increased)
- Majority class (1): 87 samples  (unchanged)

for this perpose let's adjust the parameters :

**replace = True**

 Meaning we allow sampling the same data points multiple times

**n_samples = 90**

This is crucial because if we set it larger than the original size of the data passed to resample(), it means we are oversampling

**random_state = 33**

Controls the randomness to make results reproducible

and this is s complete code :

```python 
from sklearn.utils import resample
import pandas as pd
import numpy as np

target = [1] * 87 + [0] * 3
np.random.shuffle(target)
data = pd.DataFrame({

    'feature' : range(90),
    'Target' : target

})

majority = data[data.Target == 1]
minority = data[data.Target == 0]


#Oversample the majority class
Over_sampled_minority = resample(minority,replace=True,n_samples=90, random_state=33)
res = pd.concat([majority,Over_sampled_minority])
res = res.sample(frac=1, random_state=33).reset_index(drop=True)


res.head(10)
```
# Downsampling

## Downsample the majority class

Downsampling means bringing the majority class size to match the minority class size.

**replace = False**

It means we don't need to create any new samples of data

**n_samples = len(minority)**

Why len(minoroty)? Because we want the Downsaampled majority class to be the same size as our minority class
Setting it to less than the original data size passed to resample() means we are downsampling.

**random_state = 33**

Controls the randomness to make results reproducible

Finaly this is the sample code for Downsampling :

```python
from sklearn.utils import resample
import pandas as pd
import numpy as np

target = [1] * 87 + [0] * 3
np.random.shuffle(target)
data = pd.DataFrame({

    'feature' : range(90),
    'Target' : target

})

majority = data[data.Target == 1]
minority = data[data.Target == 0]


Down_sampled_majority = resample(majority,replace=False,n_samples=len(minority), random_state=33)
res = pd.concat([minority,Down_sampled_majority])
res = res.sample(frac=1, random_state=33).reset_index(drop=True)

```

## Conclusion
Both oversampling and downsampling are powerful techniques for handling imbalanced data. The choice depends on your dataset size and specific use case. Remember to always evaluate your model on the original imbalanced test data!





