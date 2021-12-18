# Multi-Source Model Transfer with Active Sampling

This repository is the official implementation of [Multi-Source Model Transfer with Active Sampling]

## Requirements

To install requirements 

```setup
pip install -r requirements.txt
```

Datasets in the paper can be downloaded from

```url
https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md
```

## Training

To train the model(s) in the paper, run this command:

```python
import TMR_s as stf
stf.fit(data,label,sourcemodels,LAMBDA=1,MU=1,k=10)
import TMR as tf
tf.fit(data,label,sourcemodels,LAMBDA=1,MU=1,ETA=1,k=10)
```

Names of Hyperparameters are corresponding to the paper.

## Evaluation

To evaluate my model test set after training, run:

```python
tf.score(test[data],test[label]) #return balanced accuracy
```

or we can run as follows:

```python
p = tf.predict(test[data])
sklearn.metrics.balanced_accuracy_score(test[label],p)
```

## Results

Our model achieves the following performance with 1 labeled data for each class on:

### [Office+Caltech]

<table border=0 cellpadding=0 cellspacing=0 width=360 style='border-collapse:
 collapse;table-layout:fixed;width:270pt'>
 <col width=72 span=5 style='width:54pt'>
 <tr height=22 style='height:16.5pt'>
  <td height=22 class=xl65 width=72 style='height:16.5pt;width:54pt'>Model name</td>
  <td colspan=4 class=xl66 width=288 style='width:216pt'>Average&nbsp;Balanced&nbsp;Accuracy (%)</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'></td>
  <td class=xl65>amazon</td>
  <td class=xl65>caltech</td>
  <td class=xl65>webcam</td>
  <td class=xl65>dslr</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>TMR-s</td>
  <td class=xl65>68.43</td>
  <td class=xl65>54.88</td>
  <td class=xl65>72.02</td>
  <td class=xl65>74.23</td>
 </tr>
 <tr height=19 style='height:14.25pt'>
  <td height=19 class=xl65 style='height:14.25pt'>TMR</td>
  <td class=xl65>64.8</td>
  <td class=xl65>51.64</td>
  <td class=xl65>68.63</td>
  <td class=xl65>71.14</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=72 style='width:54pt'></td>
  <td width=72 style='width:54pt'></td>
  <td width=72 style='width:54pt'></td>
  <td width=72 style='width:54pt'></td>
  <td width=72 style='width:54pt'></td>
 </tr>
 <![endif]>
</table>