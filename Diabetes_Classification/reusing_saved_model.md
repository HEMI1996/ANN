

```python
from keras.models import load_model
```

    C:\Users\Hemanth\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
new_classifier = load_model('diabetes_saves_model.h5')
```


```python
import pandas as pd
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
new_classifier.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 6)                 54        
    _________________________________________________________________
    dense_2 (Dense)              (None, 4)                 28        
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 5         
    =================================================================
    Total params: 87
    Trainable params: 87
    Non-trainable params: 0
    _________________________________________________________________
    


```python
new_classifier.get_weights()
```




    [array([[ 0.08577075, -0.00317554, -0.02890483, -0.07273636,  0.01543962,
              0.02276824],
            [-0.41570225, -0.01818268,  0.00212783, -0.36741984, -0.04939605,
              0.96194065],
            [ 0.2736079 , -0.03077346, -0.029422  ,  0.31415927, -0.03245565,
             -0.07987532],
            [ 0.39389527,  0.00478519, -0.03294812,  0.07461452,  0.0132237 ,
              0.24431093],
            [ 0.17169859,  0.01586422,  0.03136007,  0.5087282 , -0.02086371,
              0.11385043],
            [-0.22409384,  0.02264729,  0.01962773, -0.2644744 , -0.0445096 ,
              0.7006111 ],
            [-0.6624032 , -0.03493542, -0.00272056, -0.09548613, -0.03822032,
              0.12958354],
            [-0.981454  ,  0.01391704, -0.04824011, -1.534628  , -0.01486595,
             -0.5878432 ]], dtype=float32),
     array([ 0.61818564, -0.00880962, -0.0107592 ,  0.6393485 ,  0.        ,
            -0.41071126], dtype=float32),
     array([[-0.0182103 ,  0.8377697 , -0.3451043 ,  0.80485106],
            [-0.00784023, -0.01449702,  0.0432823 ,  0.01877617],
            [-0.02944266, -0.03676752,  0.03198533,  0.03810853],
            [-0.0159029 ,  1.243156  , -0.7414553 ,  1.192244  ],
            [-0.00803232, -0.00589422,  0.03284996,  0.03806109],
            [-0.02716299, -1.0236872 ,  1.6672102 , -1.0076737 ]],
           dtype=float32),
     array([ 0.        ,  0.5056678 , -0.05846291,  0.5032398 ], dtype=float32),
     array([[ 0.04354352],
            [-1.3436239 ],
            [ 1.9649764 ],
            [-1.389506  ]], dtype=float32),
     array([-0.10544992], dtype=float32)]




```python
new_pred = new_classifier.predict(X_test)
```


```python
new_pred = (new_pred > 0.5)
```


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, new_pred)
```


```python
cm
```




    array([[92,  9],
           [20, 33]], dtype=int64)




```python
(92+33)/(92+9+20+33)
```




    0.8116883116883117


