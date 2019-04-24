
# DSAI HW2  
## Chien, Hsin Yen
### RE6071088, Institute of Data Science  
### https://nbviewer.jupyter.org/github/moneylys99/DSAI-HW2-Adder-practice/blob/master/Addition_re6071088.ipynb

```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
from six.moves import range
```

# Parameters Config


```python
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
```


```python
TRAINING_SIZE = 80000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
```


```python
class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)
```


```python
ctable = CharacterTable(chars)
```


```python
ctable.indices_char
```




    {0: ' ',
     1: '+',
     2: '0',
     3: '1',
     4: '2',
     5: '3',
     6: '4',
     7: '5',
     8: '6',
     9: '7',
     10: '8',
     11: '9'}



# Data Generation


```python
questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))
```

    Generating data...
    Total addition questions: 80000
    


```python
print(questions[:5], expected[:5])
```

    ['70+7   ', '34+89  ', '7+14   ', '250+31 ', '798+612'] ['77  ', '123 ', '21  ', '281 ', '1410']
    

# Processing


```python
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)
```

    Vectorization...
    


```python
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:20000]
train_y = y[:20000]
test_x = x[20000:]
test_y = y[20000:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)
```

    Training Data:
    (18000, 7, 12)
    (18000, 4, 12)
    Validation Data:
    (2000, 7, 12)
    (2000, 4, 12)
    Testing Data:
    (60000, 7, 12)
    (60000, 4, 12)
    


```python
print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])
```

    input:  [[[False False False False False False False False False False  True
       False]
      [False False False False False  True False False False False False
       False]
      [False False False False False False False False False False  True
       False]
      [False  True False False False False False False False False False
       False]
      [False False False False False False  True False False False False
       False]
      [False False False False False False False False False False False
        True]
      [ True False False False False False False False False False False
       False]]
    
     [[False False False False False False False False False False  True
       False]
      [False False False False  True False False False False False False
       False]
      [False  True False False False False False False False False False
       False]
      [False False False False False False False False False False  True
       False]
      [False False False False False False False  True False False False
       False]
      [ True False False False False False False False False False False
       False]
      [ True False False False False False False False False False False
       False]]
    
     [[False False False False  True False False False False False False
       False]
      [False False False False False False False False  True False False
       False]
      [False False False False False False False False False  True False
       False]
      [False  True False False False False False False False False False
       False]
      [False False  True False False False False False False False False
       False]
      [ True False False False False False False False False False False
       False]
      [ True False False False False False False False False False False
       False]]] 
    
     label:  [[[False False False False False False False False False False  True
       False]
      [False False False False False False False False False False  True
       False]
      [False False False False False False False False False  True False
       False]
      [ True False False False False False False False False False False
       False]]
    
     [[False False False  True False False False False False False False
       False]
      [False False False False False False False False  True False False
       False]
      [False False False False False False False False False  True False
       False]
      [ True False False False False False False False False False False
       False]]
    
     [[False False False False  True False False False False False False
       False]
      [False False False False False False False False  True False False
       False]
      [False False False False False False False False False  True False
       False]
      [ True False False False False False False False False False False
       False]]]
    

# Build Model


```python
print('Build model...')

# Initialising the RNN
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for i in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model.summary()
```

    Build model...
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_31 (LSTM)               (None, 128)               72192     
    _________________________________________________________________
    repeat_vector_1 (RepeatVecto (None, 4, 128)            0         
    _________________________________________________________________
    lstm_32 (LSTM)               (None, 4, 128)            131584    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 4, 12)             1548      
    =================================================================
    Total params: 205,324
    Trainable params: 205,324
    Non-trainable params: 0
    _________________________________________________________________
    

# Training


```python
for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
```

    
    --------------------------------------------------
    Iteration 0
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 6s 343us/step - loss: 2.0227 - acc: 0.3002 - val_loss: 1.9424 - val_acc: 0.3135
    Q 940+226 T 1166 [91m☒[0m 111 
    Q 699+71  T 770  [91m☒[0m 114 
    Q 867+249 T 1116 [91m☒[0m 1110
    Q 460+844 T 1304 [91m☒[0m 111 
    Q 110+754 T 864  [91m☒[0m 115 
    Q 0+94    T 94   [91m☒[0m 114 
    Q 87+170  T 257  [91m☒[0m 114 
    Q 654+8   T 662  [91m☒[0m 114 
    Q 980+62  T 1042 [91m☒[0m 114 
    Q 556+16  T 572  [91m☒[0m 114 
    
    --------------------------------------------------
    Iteration 1
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 232us/step - loss: 1.9033 - acc: 0.3209 - val_loss: 1.8613 - val_acc: 0.3251
    Q 130+2   T 132  [91m☒[0m 13  
    Q 502+306 T 808  [91m☒[0m 130 
    Q 850+0   T 850  [91m☒[0m 147 
    Q 64+201  T 265  [91m☒[0m 137 
    Q 2+297   T 299  [91m☒[0m 147 
    Q 2+512   T 514  [91m☒[0m 13  
    Q 66+990  T 1056 [91m☒[0m 110 
    Q 851+84  T 935  [91m☒[0m 107 
    Q 187+66  T 253  [91m☒[0m 107 
    Q 3+441   T 444  [91m☒[0m 13  
    
    --------------------------------------------------
    Iteration 2
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 246us/step - loss: 1.8503 - acc: 0.3277 - val_loss: 1.8201 - val_acc: 0.3316
    Q 95+581  T 676  [91m☒[0m 109 
    Q 41+270  T 311  [91m☒[0m 224 
    Q 90+597  T 687  [91m☒[0m 104 
    Q 713+621 T 1334 [91m☒[0m 124 
    Q 194+42  T 236  [91m☒[0m 129 
    Q 447+558 T 1005 [91m☒[0m 144 
    Q 419+7   T 426  [91m☒[0m 42  
    Q 54+27   T 81   [91m☒[0m 42  
    Q 358+181 T 539  [91m☒[0m 124 
    Q 270+9   T 279  [91m☒[0m 42  
    
    --------------------------------------------------
    Iteration 3
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 228us/step - loss: 1.8093 - acc: 0.3349 - val_loss: 1.7879 - val_acc: 0.3374
    Q 994+883 T 1877 [91m☒[0m 1511
    Q 7+805   T 812  [91m☒[0m 901 
    Q 371+18  T 389  [91m☒[0m 511 
    Q 444+693 T 1137 [91m☒[0m 111 
    Q 387+483 T 870  [91m☒[0m 1111
    Q 58+844  T 902  [91m☒[0m 101 
    Q 41+353  T 394  [91m☒[0m 555 
    Q 94+614  T 708  [91m☒[0m 101 
    Q 740+79  T 819  [91m☒[0m 101 
    Q 4+740   T 744  [91m☒[0m 511 
    
    --------------------------------------------------
    Iteration 4
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 227us/step - loss: 1.7572 - acc: 0.3538 - val_loss: 1.7603 - val_acc: 0.3526
    Q 390+71  T 461  [91m☒[0m 801 
    Q 7+805   T 812  [91m☒[0m 881 
    Q 136+94  T 230  [91m☒[0m 107 
    Q 202+45  T 247  [91m☒[0m 213 
    Q 8+724   T 732  [91m☒[0m 811 
    Q 554+863 T 1417 [91m☒[0m 1510
    Q 860+487 T 1347 [91m☒[0m 1578
    Q 86+517  T 603  [91m☒[0m 801 
    Q 203+71  T 274  [91m☒[0m 207 
    Q 6+412   T 418  [91m☒[0m 113 
    
    --------------------------------------------------
    Iteration 5
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 242us/step - loss: 1.6898 - acc: 0.3735 - val_loss: 1.6447 - val_acc: 0.3872
    Q 66+347  T 413  [91m☒[0m 667 
    Q 43+911  T 954  [91m☒[0m 599 
    Q 262+787 T 1049 [91m☒[0m 112 
    Q 85+350  T 435  [91m☒[0m 549 
    Q 960+450 T 1410 [91m☒[0m 1544
    Q 98+374  T 472  [91m☒[0m 901 
    Q 26+34   T 60   [91m☒[0m 27  
    Q 238+511 T 749  [91m☒[0m 999 
    Q 107+98  T 205  [91m☒[0m 899 
    Q 78+81   T 159  [91m☒[0m 889 
    
    --------------------------------------------------
    Iteration 6
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 241us/step - loss: 1.6280 - acc: 0.3924 - val_loss: 1.5912 - val_acc: 0.4016
    Q 478+24  T 502  [91m☒[0m 437 
    Q 46+768  T 814  [91m☒[0m 733 
    Q 955+575 T 1530 [91m☒[0m 1511
    Q 862+853 T 1715 [91m☒[0m 1566
    Q 10+653  T 663  [91m☒[0m 667 
    Q 381+44  T 425  [91m☒[0m 440 
    Q 91+191  T 282  [91m☒[0m 999 
    Q 44+268  T 312  [91m☒[0m 490 
    Q 1+71    T 72   [91m☒[0m 11  
    Q 692+71  T 763  [91m☒[0m 837 
    
    --------------------------------------------------
    Iteration 7
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 250us/step - loss: 1.5758 - acc: 0.4116 - val_loss: 1.5495 - val_acc: 0.4207
    Q 876+230 T 1106 [91m☒[0m 1119
    Q 6+215   T 221  [91m☒[0m 22  
    Q 359+563 T 922  [91m☒[0m 107 
    Q 209+36  T 245  [91m☒[0m 333 
    Q 270+9   T 279  [91m☒[0m 22  
    Q 19+671  T 690  [91m☒[0m 887 
    Q 358+0   T 358  [91m☒[0m 11  
    Q 174+138 T 312  [91m☒[0m 289 
    Q 52+545  T 597  [91m☒[0m 519 
    Q 811+74  T 885  [91m☒[0m 981 
    
    --------------------------------------------------
    Iteration 8
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 284us/step - loss: 1.5278 - acc: 0.4278 - val_loss: 1.5036 - val_acc: 0.4378
    Q 78+347  T 425  [91m☒[0m 434 
    Q 343+3   T 346  [91m☒[0m 34  
    Q 504+237 T 741  [91m☒[0m 566 
    Q 9+49    T 58   [91m☒[0m 54  
    Q 275+2   T 277  [91m☒[0m 22  
    Q 4+628   T 632  [91m☒[0m 54  
    Q 843+90  T 933  [91m☒[0m 951 
    Q 46+949  T 995  [91m☒[0m 903 
    Q 139+76  T 215  [91m☒[0m 344 
    Q 560+328 T 888  [91m☒[0m 706 
    
    --------------------------------------------------
    Iteration 9
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 262us/step - loss: 1.4835 - acc: 0.4439 - val_loss: 1.4451 - val_acc: 0.4579
    Q 496+58  T 554  [91m☒[0m 524 
    Q 227+14  T 241  [91m☒[0m 251 
    Q 77+286  T 363  [91m☒[0m 345 
    Q 36+64   T 100  [91m☒[0m 600 
    Q 344+53  T 397  [91m☒[0m 470 
    Q 2+730   T 732  [91m☒[0m 201 
    Q 237+795 T 1032 [91m☒[0m 101 
    Q 6+209   T 215  [91m☒[0m 225 
    Q 648+9   T 657  [91m☒[0m 155 
    Q 44+27   T 71   [91m☒[0m 440 
    
    --------------------------------------------------
    Iteration 10
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 276us/step - loss: 1.4370 - acc: 0.4598 - val_loss: 1.4079 - val_acc: 0.4713
    Q 28+785  T 813  [91m☒[0m 867 
    Q 659+698 T 1357 [91m☒[0m 1354
    Q 669+896 T 1565 [91m☒[0m 1554
    Q 413+129 T 542  [91m☒[0m 544 
    Q 79+43   T 122  [91m☒[0m 144 
    Q 92+507  T 599  [91m☒[0m 667 
    Q 39+978  T 1017 [91m☒[0m 1022
    Q 219+540 T 759  [91m☒[0m 744 
    Q 94+315  T 409  [91m☒[0m 424 
    Q 27+52   T 79   [91m☒[0m 17  
    
    --------------------------------------------------
    Iteration 11
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 262us/step - loss: 1.3935 - acc: 0.4767 - val_loss: 1.3733 - val_acc: 0.4799
    Q 61+801  T 862  [91m☒[0m 988 
    Q 23+399  T 422  [91m☒[0m 316 
    Q 465+978 T 1443 [91m☒[0m 1542
    Q 28+658  T 686  [92m☑[0m 686 
    Q 886+99  T 985  [91m☒[0m 976 
    Q 752+912 T 1664 [91m☒[0m 1516
    Q 139+76  T 215  [91m☒[0m 228 
    Q 46+745  T 791  [91m☒[0m 719 
    Q 788+16  T 804  [91m☒[0m 889 
    Q 39+978  T 1017 [91m☒[0m 1026
    
    --------------------------------------------------
    Iteration 12
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 270us/step - loss: 1.3525 - acc: 0.4936 - val_loss: 1.3302 - val_acc: 0.5024
    Q 55+357  T 412  [91m☒[0m 502 
    Q 7+329   T 336  [91m☒[0m 365 
    Q 848+891 T 1739 [91m☒[0m 1775
    Q 160+894 T 1054 [91m☒[0m 106 
    Q 942+83  T 1025 [91m☒[0m 101 
    Q 28+404  T 432  [91m☒[0m 405 
    Q 313+761 T 1074 [91m☒[0m 106 
    Q 40+459  T 499  [91m☒[0m 599 
    Q 835+304 T 1139 [91m☒[0m 1166
    Q 419+7   T 426  [91m☒[0m 414 
    
    --------------------------------------------------
    Iteration 13
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 276us/step - loss: 1.3109 - acc: 0.5099 - val_loss: 1.3380 - val_acc: 0.4884
    Q 389+395 T 784  [91m☒[0m 876 
    Q 126+136 T 262  [91m☒[0m 270 
    Q 496+979 T 1475 [91m☒[0m 1387
    Q 359+563 T 922  [91m☒[0m 822 
    Q 325+60  T 385  [91m☒[0m 390 
    Q 209+36  T 245  [91m☒[0m 200 
    Q 997+37  T 1034 [91m☒[0m 1021
    Q 489+66  T 555  [91m☒[0m 544 
    Q 94+811  T 905  [91m☒[0m 862 
    Q 632+786 T 1418 [91m☒[0m 1350
    
    --------------------------------------------------
    Iteration 14
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 258us/step - loss: 1.2730 - acc: 0.5241 - val_loss: 1.3000 - val_acc: 0.5031
    Q 66+573  T 639  [91m☒[0m 633 
    Q 2+64    T 66   [92m☑[0m 66  
    Q 806+76  T 882  [91m☒[0m 773 
    Q 653+310 T 963  [91m☒[0m 966 
    Q 163+161 T 324  [91m☒[0m 278 
    Q 657+66  T 723  [91m☒[0m 722 
    Q 95+188  T 283  [91m☒[0m 267 
    Q 721+71  T 792  [91m☒[0m 783 
    Q 389+395 T 784  [91m☒[0m 877 
    Q 3+700   T 703  [91m☒[0m 70  
    
    --------------------------------------------------
    Iteration 15
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 273us/step - loss: 1.2386 - acc: 0.5382 - val_loss: 1.2404 - val_acc: 0.5304
    Q 97+676  T 773  [91m☒[0m 754 
    Q 51+215  T 266  [91m☒[0m 157 
    Q 7+329   T 336  [91m☒[0m 346 
    Q 737+21  T 758  [91m☒[0m 745 
    Q 1+688   T 689  [91m☒[0m 157 
    Q 494+21  T 515  [91m☒[0m 449 
    Q 50+276  T 326  [91m☒[0m 235 
    Q 622+32  T 654  [91m☒[0m 644 
    Q 81+170  T 251  [91m☒[0m 287 
    Q 995+6   T 1001 [91m☒[0m 194 
    
    --------------------------------------------------
    Iteration 16
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 264us/step - loss: 1.2004 - acc: 0.5513 - val_loss: 1.1943 - val_acc: 0.5546
    Q 11+936  T 947  [91m☒[0m 944 
    Q 117+67  T 184  [91m☒[0m 187 
    Q 87+170  T 257  [91m☒[0m 287 
    Q 680+785 T 1465 [91m☒[0m 1467
    Q 11+289  T 300  [91m☒[0m 209 
    Q 958+334 T 1292 [91m☒[0m 1299
    Q 27+826  T 853  [91m☒[0m 859 
    Q 34+916  T 950  [91m☒[0m 955 
    Q 36+96   T 132  [91m☒[0m 101 
    Q 275+2   T 277  [92m☑[0m 277 
    
    --------------------------------------------------
    Iteration 17
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 259us/step - loss: 1.1647 - acc: 0.5646 - val_loss: 1.1673 - val_acc: 0.5606
    Q 7+266   T 273  [91m☒[0m 363 
    Q 564+300 T 864  [91m☒[0m 863 
    Q 580+71  T 651  [91m☒[0m 661 
    Q 13+904  T 917  [91m☒[0m 934 
    Q 44+321  T 365  [91m☒[0m 458 
    Q 55+532  T 587  [91m☒[0m 578 
    Q 825+63  T 888  [91m☒[0m 908 
    Q 882+58  T 940  [91m☒[0m 930 
    Q 617+779 T 1396 [91m☒[0m 1400
    Q 807+889 T 1696 [91m☒[0m 1766
    
    --------------------------------------------------
    Iteration 18
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 1.1270 - acc: 0.5784 - val_loss: 1.1265 - val_acc: 0.5761
    Q 329+972 T 1301 [91m☒[0m 1226
    Q 97+676  T 773  [91m☒[0m 764 
    Q 868+841 T 1709 [91m☒[0m 1628
    Q 29+93   T 122  [91m☒[0m 123 
    Q 767+190 T 957  [92m☑[0m 957 
    Q 82+611  T 693  [91m☒[0m 793 
    Q 7+802   T 809  [91m☒[0m 829 
    Q 908+23  T 931  [91m☒[0m 921 
    Q 43+281  T 324  [91m☒[0m 325 
    Q 12+823  T 835  [91m☒[0m 844 
    
    --------------------------------------------------
    Iteration 19
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 259us/step - loss: 1.0883 - acc: 0.5955 - val_loss: 1.0951 - val_acc: 0.5843
    Q 432+7   T 439  [91m☒[0m 331 
    Q 653+310 T 963  [91m☒[0m 975 
    Q 20+513  T 533  [91m☒[0m 532 
    Q 694+71  T 765  [91m☒[0m 757 
    Q 83+855  T 938  [91m☒[0m 936 
    Q 737+47  T 784  [91m☒[0m 801 
    Q 896+94  T 990  [91m☒[0m 104 
    Q 122+7   T 129  [91m☒[0m 228 
    Q 95+440  T 535  [91m☒[0m 533 
    Q 11+733  T 744  [92m☑[0m 744 
    
    --------------------------------------------------
    Iteration 20
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 257us/step - loss: 1.0496 - acc: 0.6098 - val_loss: 1.0532 - val_acc: 0.6040
    Q 67+435  T 502  [91m☒[0m 492 
    Q 141+20  T 161  [91m☒[0m 142 
    Q 358+0   T 358  [91m☒[0m 459 
    Q 664+73  T 737  [92m☑[0m 737 
    Q 650+0   T 650  [91m☒[0m 611 
    Q 601+444 T 1045 [91m☒[0m 1035
    Q 42+223  T 265  [91m☒[0m 256 
    Q 3+700   T 703  [91m☒[0m 701 
    Q 642+480 T 1122 [91m☒[0m 1125
    Q 334+199 T 533  [91m☒[0m 516 
    
    --------------------------------------------------
    Iteration 21
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 277us/step - loss: 1.0086 - acc: 0.6229 - val_loss: 1.0243 - val_acc: 0.6117
    Q 7+266   T 273  [92m☑[0m 273 
    Q 7+802   T 809  [92m☑[0m 809 
    Q 28+602  T 630  [91m☒[0m 631 
    Q 194+12  T 206  [91m☒[0m 203 
    Q 839+780 T 1619 [91m☒[0m 1798
    Q 214+911 T 1125 [91m☒[0m 1133
    Q 63+76   T 139  [91m☒[0m 149 
    Q 785+621 T 1406 [91m☒[0m 1418
    Q 83+92   T 175  [92m☑[0m 175 
    Q 957+21  T 978  [91m☒[0m 987 
    
    --------------------------------------------------
    Iteration 22
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 263us/step - loss: 0.9639 - acc: 0.6392 - val_loss: 0.9748 - val_acc: 0.6274
    Q 237+397 T 634  [91m☒[0m 516 
    Q 30+465  T 495  [91m☒[0m 496 
    Q 68+75   T 143  [91m☒[0m 152 
    Q 72+231  T 303  [91m☒[0m 394 
    Q 202+45  T 247  [91m☒[0m 277 
    Q 5+923   T 928  [91m☒[0m 937 
    Q 343+63  T 406  [91m☒[0m 497 
    Q 359+79  T 438  [91m☒[0m 446 
    Q 667+432 T 1099 [91m☒[0m 1199
    Q 546+61  T 607  [91m☒[0m 616 
    
    --------------------------------------------------
    Iteration 23
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 229us/step - loss: 0.9158 - acc: 0.6567 - val_loss: 0.9128 - val_acc: 0.6538
    Q 896+94  T 990  [91m☒[0m 980 
    Q 2+679   T 681  [91m☒[0m 680 
    Q 556+16  T 572  [91m☒[0m 571 
    Q 86+930  T 1016 [91m☒[0m 1012
    Q 62+459  T 521  [91m☒[0m 510 
    Q 15+84   T 99   [91m☒[0m 11  
    Q 5+151   T 156  [91m☒[0m 155 
    Q 955+49  T 1004 [91m☒[0m 9004
    Q 797+929 T 1726 [91m☒[0m 1708
    Q 860+226 T 1086 [91m☒[0m 1073
    
    --------------------------------------------------
    Iteration 24
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 237us/step - loss: 0.8668 - acc: 0.6731 - val_loss: 0.8855 - val_acc: 0.6518
    Q 33+342  T 375  [91m☒[0m 366 
    Q 650+552 T 1202 [91m☒[0m 1166
    Q 68+246  T 314  [92m☑[0m 314 
    Q 90+439  T 529  [91m☒[0m 539 
    Q 822+77  T 899  [92m☑[0m 899 
    Q 388+50  T 438  [91m☒[0m 439 
    Q 755+46  T 801  [91m☒[0m 702 
    Q 28+404  T 432  [91m☒[0m 430 
    Q 91+396  T 487  [91m☒[0m 488 
    Q 160+72  T 232  [91m☒[0m 233 
    
    --------------------------------------------------
    Iteration 25
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 251us/step - loss: 0.8128 - acc: 0.6920 - val_loss: 0.8648 - val_acc: 0.6520
    Q 9+645   T 654  [91m☒[0m 555 
    Q 501+83  T 584  [91m☒[0m 591 
    Q 81+302  T 383  [91m☒[0m 393 
    Q 70+87   T 157  [92m☑[0m 157 
    Q 650+707 T 1357 [91m☒[0m 1361
    Q 717+644 T 1361 [91m☒[0m 1393
    Q 44+757  T 801  [91m☒[0m 802 
    Q 392+45  T 437  [91m☒[0m 438 
    Q 995+6   T 1001 [91m☒[0m 190 
    Q 388+361 T 749  [91m☒[0m 741 
    
    --------------------------------------------------
    Iteration 26
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 247us/step - loss: 0.7580 - acc: 0.7136 - val_loss: 0.8130 - val_acc: 0.6809
    Q 886+99  T 985  [91m☒[0m 984 
    Q 272+96  T 368  [92m☑[0m 368 
    Q 80+644  T 724  [92m☑[0m 724 
    Q 21+204  T 225  [91m☒[0m 222 
    Q 474+835 T 1309 [91m☒[0m 1317
    Q 542+90  T 632  [91m☒[0m 634 
    Q 769+863 T 1632 [92m☑[0m 1632
    Q 95+532  T 627  [91m☒[0m 637 
    Q 348+709 T 1057 [91m☒[0m 1077
    Q 680+50  T 730  [91m☒[0m 721 
    
    --------------------------------------------------
    Iteration 27
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 250us/step - loss: 0.7054 - acc: 0.7358 - val_loss: 0.7301 - val_acc: 0.7081 - loss: 0.
    Q 75+520  T 595  [91m☒[0m 586 
    Q 38+25   T 63   [91m☒[0m 73  
    Q 183+89  T 272  [91m☒[0m 271 
    Q 81+887  T 968  [91m☒[0m 967 
    Q 34+88   T 122  [91m☒[0m 131 
    Q 77+286  T 363  [91m☒[0m 353 
    Q 329+972 T 1301 [91m☒[0m 1210
    Q 622+41  T 663  [91m☒[0m 672 
    Q 562+825 T 1387 [91m☒[0m 1377
    Q 333+91  T 424  [91m☒[0m 414 
    
    --------------------------------------------------
    Iteration 28
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 271us/step - loss: 0.6559 - acc: 0.7526 - val_loss: 0.6633 - val_acc: 0.7469
    Q 643+119 T 762  [91m☒[0m 761 
    Q 628+616 T 1244 [91m☒[0m 1243
    Q 213+628 T 841  [91m☒[0m 832 
    Q 13+568  T 581  [92m☑[0m 581 
    Q 91+406  T 497  [91m☒[0m 408 
    Q 130+50  T 180  [91m☒[0m 181 
    Q 426+55  T 481  [92m☑[0m 481 
    Q 307+1   T 308  [91m☒[0m 217 
    Q 126+136 T 262  [91m☒[0m 362 
    Q 62+965  T 1027 [92m☑[0m 1027
    
    --------------------------------------------------
    Iteration 29
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 268us/step - loss: 0.6085 - acc: 0.7742 - val_loss: 0.6973 - val_acc: 0.7137
    Q 890+39  T 929  [91m☒[0m 939 
    Q 472+575 T 1047 [91m☒[0m 1067
    Q 675+59  T 734  [91m☒[0m 735 
    Q 19+98   T 117  [91m☒[0m 116 
    Q 772+1   T 773  [91m☒[0m 875 
    Q 394+51  T 445  [91m☒[0m 436 
    Q 334+199 T 533  [91m☒[0m 504 
    Q 956+103 T 1059 [91m☒[0m 1159
    Q 431+951 T 1382 [91m☒[0m 1374
    Q 3+369   T 372  [91m☒[0m 362 
    
    --------------------------------------------------
    Iteration 30
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 258us/step - loss: 0.5655 - acc: 0.7924 - val_loss: 0.6497 - val_acc: 0.7325
    Q 160+894 T 1054 [91m☒[0m 1055
    Q 2+297   T 299  [92m☑[0m 299 
    Q 615+8   T 623  [91m☒[0m 622 
    Q 325+60  T 385  [91m☒[0m 395 
    Q 24+69   T 93   [91m☒[0m 90  
    Q 200+829 T 1029 [91m☒[0m 1008
    Q 28+80   T 108  [91m☒[0m 809 
    Q 959+65  T 1024 [91m☒[0m 1034
    Q 905+78  T 983  [91m☒[0m 991 
    Q 28+593  T 621  [91m☒[0m 620 
    
    --------------------------------------------------
    Iteration 31
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 259us/step - loss: 0.5301 - acc: 0.8066 - val_loss: 0.5404 - val_acc: 0.8051
    Q 487+918 T 1405 [91m☒[0m 1426
    Q 2+512   T 514  [91m☒[0m 524 
    Q 31+804  T 835  [91m☒[0m 845 
    Q 546+831 T 1377 [92m☑[0m 1377
    Q 576+480 T 1056 [91m☒[0m 1057
    Q 238+732 T 970  [91m☒[0m 969 
    Q 89+43   T 132  [92m☑[0m 132 
    Q 327+77  T 404  [92m☑[0m 404 
    Q 488+38  T 526  [92m☑[0m 526 
    Q 122+7   T 129  [92m☑[0m 129 
    
    --------------------------------------------------
    Iteration 32
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 260us/step - loss: 0.4984 - acc: 0.8208 - val_loss: 0.5557 - val_acc: 0.7816
    Q 7+118   T 125  [91m☒[0m 124 
    Q 896+823 T 1719 [91m☒[0m 1718
    Q 94+234  T 328  [92m☑[0m 328 
    Q 737+21  T 758  [92m☑[0m 758 
    Q 729+328 T 1057 [91m☒[0m 1066
    Q 2+261   T 263  [92m☑[0m 263 
    Q 967+28  T 995  [91m☒[0m 900 
    Q 95+457  T 552  [92m☑[0m 552 
    Q 72+857  T 929  [91m☒[0m 928 
    Q 2+980   T 982  [91m☒[0m 993 
    
    --------------------------------------------------
    Iteration 33
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 268us/step - loss: 0.4627 - acc: 0.8398 - val_loss: 0.4881 - val_acc: 0.8193
    Q 778+863 T 1641 [91m☒[0m 1652
    Q 602+46  T 648  [91m☒[0m 658 
    Q 11+206  T 217  [91m☒[0m 227 
    Q 45+373  T 418  [91m☒[0m 419 
    Q 795+792 T 1587 [91m☒[0m 1677
    Q 858+929 T 1787 [92m☑[0m 1787
    Q 931+608 T 1539 [91m☒[0m 1549
    Q 44+27   T 71   [91m☒[0m 70  
    Q 91+396  T 487  [92m☑[0m 487 
    Q 862+853 T 1715 [91m☒[0m 1725
    
    --------------------------------------------------
    Iteration 34
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.4348 - acc: 0.8493 - val_loss: 0.4522 - val_acc: 0.8417
    Q 437+370 T 807  [91m☒[0m 817 
    Q 279+375 T 654  [91m☒[0m 664 
    Q 493+30  T 523  [92m☑[0m 523 
    Q 374+850 T 1224 [92m☑[0m 1224
    Q 585+577 T 1162 [92m☑[0m 1162
    Q 82+930  T 1012 [92m☑[0m 1012
    Q 74+69   T 143  [92m☑[0m 143 
    Q 329+972 T 1301 [91m☒[0m 1211
    Q 19+324  T 343  [92m☑[0m 343 
    Q 61+304  T 365  [92m☑[0m 365 
    
    --------------------------------------------------
    Iteration 35
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 270us/step - loss: 0.4097 - acc: 0.8597 - val_loss: 0.4377 - val_acc: 0.8373
    Q 515+48  T 563  [92m☑[0m 563 
    Q 64+201  T 265  [91m☒[0m 266 
    Q 797+38  T 835  [91m☒[0m 836 
    Q 348+709 T 1057 [91m☒[0m 1077
    Q 344+18  T 362  [91m☒[0m 363 
    Q 10+76   T 86   [91m☒[0m 87  
    Q 0+700   T 700  [92m☑[0m 700 
    Q 530+34  T 564  [92m☑[0m 564 
    Q 686+47  T 733  [92m☑[0m 733 
    Q 179+32  T 211  [92m☑[0m 211 
    
    --------------------------------------------------
    Iteration 36
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 256us/step - loss: 0.3857 - acc: 0.8703 - val_loss: 0.4231 - val_acc: 0.8482
    Q 141+49  T 190  [91m☒[0m 199 
    Q 371+18  T 389  [91m☒[0m 399 
    Q 123+93  T 216  [92m☑[0m 216 
    Q 737+21  T 758  [92m☑[0m 758 
    Q 934+528 T 1462 [92m☑[0m 1462
    Q 447+680 T 1127 [92m☑[0m 1127
    Q 38+25   T 63   [92m☑[0m 63  
    Q 906+684 T 1590 [92m☑[0m 1590
    Q 895+80  T 975  [91m☒[0m 974 
    Q 40+459  T 499  [91m☒[0m 599 
    
    --------------------------------------------------
    Iteration 37
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 260us/step - loss: 0.3647 - acc: 0.8790 - val_loss: 0.4355 - val_acc: 0.8248
    Q 787+92  T 879  [92m☑[0m 879 
    Q 8+314   T 322  [92m☑[0m 322 
    Q 887+69  T 956  [92m☑[0m 956 
    Q 289+394 T 683  [91m☒[0m 783 
    Q 22+101  T 123  [91m☒[0m 122 
    Q 44+27   T 71   [91m☒[0m 70  
    Q 858+80  T 938  [91m☒[0m 948 
    Q 98+846  T 944  [91m☒[0m 954 
    Q 0+721   T 721  [91m☒[0m 702 
    Q 783+37  T 820  [92m☑[0m 820 
    
    --------------------------------------------------
    Iteration 38
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 263us/step - loss: 0.3445 - acc: 0.8883 - val_loss: 0.4027 - val_acc: 0.8471
    Q 485+89  T 574  [91m☒[0m 584 
    Q 36+983  T 1019 [91m☒[0m 1029
    Q 46+94   T 140  [92m☑[0m 140 
    Q 908+64  T 972  [92m☑[0m 972 
    Q 859+902 T 1761 [91m☒[0m 1771
    Q 641+199 T 840  [92m☑[0m 840 
    Q 117+67  T 184  [91m☒[0m 183 
    Q 419+942 T 1361 [92m☑[0m 1361
    Q 741+1   T 742  [91m☒[0m 732 
    Q 998+1   T 999  [91m☒[0m 199 
    
    --------------------------------------------------
    Iteration 39
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 244us/step - loss: 0.3238 - acc: 0.8975 - val_loss: 0.3540 - val_acc: 0.8774
    Q 797+38  T 835  [92m☑[0m 835 
    Q 419+942 T 1361 [92m☑[0m 1361
    Q 13+638  T 651  [91m☒[0m 650 
    Q 650+707 T 1357 [92m☑[0m 1357
    Q 29+379  T 408  [91m☒[0m 307 
    Q 258+67  T 325  [91m☒[0m 324 
    Q 46+770  T 816  [92m☑[0m 816 
    Q 97+532  T 629  [92m☑[0m 629 
    Q 253+3   T 256  [92m☑[0m 256 
    Q 622+32  T 654  [92m☑[0m 654 
    
    --------------------------------------------------
    Iteration 40
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 251us/step - loss: 0.3031 - acc: 0.9047 - val_loss: 0.3480 - val_acc: 0.8715
    Q 749+24  T 773  [92m☑[0m 773 
    Q 551+569 T 1120 [92m☑[0m 1120
    Q 932+865 T 1797 [91m☒[0m 1897
    Q 645+554 T 1199 [92m☑[0m 1199
    Q 5+570   T 575  [92m☑[0m 575 
    Q 73+152  T 225  [91m☒[0m 226 
    Q 839+780 T 1619 [91m☒[0m 1610
    Q 957+99  T 1056 [91m☒[0m 1055
    Q 296+58  T 354  [92m☑[0m 354 
    Q 327+20  T 347  [92m☑[0m 347 
    
    --------------------------------------------------
    Iteration 41
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 264us/step - loss: 0.2879 - acc: 0.9122 - val_loss: 0.3630 - val_acc: 0.8524 0.2884 - acc: 0.91
    Q 704+550 T 1254 [92m☑[0m 1254
    Q 455+7   T 462  [92m☑[0m 462 
    Q 95+532  T 627  [92m☑[0m 627 
    Q 85+150  T 235  [92m☑[0m 235 
    Q 580+71  T 651  [92m☑[0m 651 
    Q 777+657 T 1434 [92m☑[0m 1434
    Q 901+823 T 1724 [92m☑[0m 1724
    Q 496+58  T 554  [91m☒[0m 553 
    Q 100+680 T 780  [92m☑[0m 780 
    Q 921+635 T 1556 [92m☑[0m 1556
    
    --------------------------------------------------
    Iteration 42
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 238us/step - loss: 0.2730 - acc: 0.9168 - val_loss: 0.3280 - val_acc: 0.8871
    Q 4+853   T 857  [92m☑[0m 857 
    Q 31+112  T 143  [91m☒[0m 133 
    Q 351+655 T 1006 [92m☑[0m 1006
    Q 1+573   T 574  [92m☑[0m 574 
    Q 972+748 T 1720 [91m☒[0m 1710
    Q 997+815 T 1812 [91m☒[0m 1710
    Q 8+694   T 702  [91m☒[0m 701 
    Q 959+65  T 1024 [92m☑[0m 1024
    Q 789+889 T 1678 [92m☑[0m 1678
    Q 61+304  T 365  [92m☑[0m 365 
    
    --------------------------------------------------
    Iteration 43
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 239us/step - loss: 0.2578 - acc: 0.9224 - val_loss: 0.2846 - val_acc: 0.9042
    Q 359+563 T 922  [92m☑[0m 922 
    Q 559+461 T 1020 [92m☑[0m 1020
    Q 50+133  T 183  [92m☑[0m 183 
    Q 569+728 T 1297 [92m☑[0m 1297
    Q 14+826  T 840  [92m☑[0m 840 
    Q 98+760  T 858  [92m☑[0m 858 
    Q 843+90  T 933  [92m☑[0m 933 
    Q 7+805   T 812  [92m☑[0m 812 
    Q 957+21  T 978  [92m☑[0m 978 
    Q 73+22   T 95   [91m☒[0m 96  
    
    --------------------------------------------------
    Iteration 44
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 233us/step - loss: 0.2432 - acc: 0.9293 - val_loss: 0.2672 - val_acc: 0.9130
    Q 90+76   T 166  [92m☑[0m 166 
    Q 75+784  T 859  [91m☒[0m 869 
    Q 406+902 T 1308 [91m☒[0m 1318
    Q 81+889  T 970  [91m☒[0m 960 
    Q 392+873 T 1265 [92m☑[0m 1265
    Q 80+828  T 908  [92m☑[0m 908 
    Q 831+27  T 858  [92m☑[0m 858 
    Q 203+231 T 434  [92m☑[0m 434 
    Q 68+75   T 143  [92m☑[0m 143 
    Q 32+367  T 399  [91m☒[0m 409 
    
    --------------------------------------------------
    Iteration 45
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 252us/step - loss: 0.2296 - acc: 0.9335 - val_loss: 0.2495 - val_acc: 0.9256
    Q 9+291   T 300  [91m☒[0m 200 
    Q 4+191   T 195  [91m☒[0m 194 
    Q 22+55   T 77   [92m☑[0m 77  
    Q 136+82  T 218  [91m☒[0m 228 
    Q 905+78  T 983  [92m☑[0m 983 
    Q 203+231 T 434  [92m☑[0m 434 
    Q 753+59  T 812  [92m☑[0m 812 
    Q 276+70  T 346  [92m☑[0m 346 
    Q 162+278 T 440  [92m☑[0m 440 
    Q 902+89  T 991  [92m☑[0m 991 
    
    --------------------------------------------------
    Iteration 46
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.2169 - acc: 0.9392 - val_loss: 0.3175 - val_acc: 0.8820
    Q 67+436  T 503  [92m☑[0m 503 
    Q 612+22  T 634  [92m☑[0m 634 
    Q 576+480 T 1056 [92m☑[0m 1056
    Q 82+151  T 233  [92m☑[0m 233 
    Q 461+68  T 529  [91m☒[0m 530 
    Q 4+160   T 164  [92m☑[0m 164 
    Q 83+983  T 1066 [92m☑[0m 1066
    Q 50+724  T 774  [91m☒[0m 775 
    Q 886+679 T 1565 [92m☑[0m 1565
    Q 1+102   T 103  [91m☒[0m 102 
    
    --------------------------------------------------
    Iteration 47
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 272us/step - loss: 0.2071 - acc: 0.9420 - val_loss: 0.2459 - val_acc: 0.9179
    Q 704+89  T 793  [91m☒[0m 893 
    Q 568+48  T 616  [92m☑[0m 616 
    Q 767+190 T 957  [91m☒[0m 958 
    Q 198+43  T 241  [91m☒[0m 231 
    Q 75+684  T 759  [92m☑[0m 759 
    Q 345+212 T 557  [92m☑[0m 557 
    Q 76+301  T 377  [92m☑[0m 377 
    Q 697+947 T 1644 [92m☑[0m 1644
    Q 97+115  T 212  [92m☑[0m 212 
    Q 755+498 T 1253 [92m☑[0m 1253
    
    --------------------------------------------------
    Iteration 48
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 272us/step - loss: 0.1969 - acc: 0.9441 - val_loss: 0.2536 - val_acc: 0.9103
    Q 682+7   T 689  [92m☑[0m 689 
    Q 21+481  T 502  [92m☑[0m 502 
    Q 742+839 T 1581 [91m☒[0m 1571
    Q 147+104 T 251  [91m☒[0m 250 
    Q 50+394  T 444  [92m☑[0m 444 
    Q 2+512   T 514  [92m☑[0m 514 
    Q 0+721   T 721  [91m☒[0m 722 
    Q 37+790  T 827  [92m☑[0m 827 
    Q 425+55  T 480  [92m☑[0m 480 
    Q 27+52   T 79   [92m☑[0m 79  
    
    --------------------------------------------------
    Iteration 49
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 255us/step - loss: 0.1892 - acc: 0.9472 - val_loss: 0.2230 - val_acc: 0.9328
    Q 198+43  T 241  [92m☑[0m 241 
    Q 713+461 T 1174 [91m☒[0m 1164
    Q 256+20  T 276  [92m☑[0m 276 
    Q 970+54  T 1024 [91m☒[0m 1034
    Q 305+282 T 587  [91m☒[0m 687 
    Q 91+406  T 497  [92m☑[0m 497 
    Q 805+38  T 843  [92m☑[0m 843 
    Q 74+69   T 143  [91m☒[0m 153 
    Q 175+79  T 254  [92m☑[0m 254 
    Q 642+115 T 757  [92m☑[0m 757 
    
    --------------------------------------------------
    Iteration 50
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 272us/step - loss: 0.1763 - acc: 0.9530 - val_loss: 0.2380 - val_acc: 0.9151 0.173
    Q 798+790 T 1588 [91m☒[0m 1589
    Q 762+64  T 826  [92m☑[0m 826 
    Q 425+81  T 506  [92m☑[0m 506 
    Q 384+806 T 1190 [91m☒[0m 1180
    Q 238+258 T 496  [92m☑[0m 496 
    Q 243+28  T 271  [92m☑[0m 271 
    Q 13+744  T 757  [92m☑[0m 757 
    Q 11+624  T 635  [92m☑[0m 635 
    Q 345+77  T 422  [92m☑[0m 422 
    Q 863+2   T 865  [92m☑[0m 865 
    
    --------------------------------------------------
    Iteration 51
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 265us/step - loss: 0.1681 - acc: 0.9558 - val_loss: 0.2310 - val_acc: 0.9176
    Q 824+73  T 897  [92m☑[0m 897 
    Q 22+234  T 256  [92m☑[0m 256 
    Q 1+84    T 85   [91m☒[0m 86  
    Q 5+62    T 67   [91m☒[0m 66  
    Q 33+448  T 481  [92m☑[0m 481 
    Q 115+275 T 390  [92m☑[0m 390 
    Q 285+52  T 337  [92m☑[0m 337 
    Q 506+14  T 520  [91m☒[0m 519 
    Q 802+518 T 1320 [92m☑[0m 1320
    Q 74+827  T 901  [92m☑[0m 901 
    
    --------------------------------------------------
    Iteration 52
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.1585 - acc: 0.9574 - val_loss: 0.2650 - val_acc: 0.8961
    Q 763+0   T 763  [91m☒[0m 754 
    Q 678+80  T 758  [92m☑[0m 758 
    Q 67+436  T 503  [91m☒[0m 403 
    Q 495+376 T 871  [91m☒[0m 861 
    Q 268+4   T 272  [91m☒[0m 262 
    Q 92+329  T 421  [92m☑[0m 421 
    Q 916+170 T 1086 [91m☒[0m 1087
    Q 552+61  T 613  [92m☑[0m 613 
    Q 685+416 T 1101 [91m☒[0m 1001
    Q 828+68  T 896  [91m☒[0m 996 
    
    --------------------------------------------------
    Iteration 53
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.1531 - acc: 0.9595 - val_loss: 0.2253 - val_acc: 0.9199
    Q 501+5   T 506  [91m☒[0m 505 
    Q 23+475  T 498  [92m☑[0m 498 
    Q 331+66  T 397  [91m☒[0m 387 
    Q 312+9   T 321  [92m☑[0m 321 
    Q 385+10  T 395  [91m☒[0m 394 
    Q 734+694 T 1428 [92m☑[0m 1428
    Q 94+45   T 139  [92m☑[0m 139 
    Q 894+832 T 1726 [92m☑[0m 1726
    Q 92+329  T 421  [92m☑[0m 421 
    Q 68+862  T 930  [91m☒[0m 920 
    
    --------------------------------------------------
    Iteration 54
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 261us/step - loss: 0.1447 - acc: 0.9628 - val_loss: 0.1998 - val_acc: 0.9338
    Q 731+16  T 747  [92m☑[0m 747 
    Q 525+37  T 562  [92m☑[0m 562 
    Q 59+537  T 596  [92m☑[0m 596 
    Q 99+503  T 602  [92m☑[0m 602 
    Q 772+88  T 860  [91m☒[0m 850 
    Q 889+2   T 891  [92m☑[0m 891 
    Q 78+490  T 568  [92m☑[0m 568 
    Q 843+473 T 1316 [92m☑[0m 1316
    Q 725+26  T 751  [92m☑[0m 751 
    Q 791+81  T 872  [92m☑[0m 872 
    
    --------------------------------------------------
    Iteration 55
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.1372 - acc: 0.9666 - val_loss: 0.1795 - val_acc: 0.9433
    Q 474+835 T 1309 [92m☑[0m 1309
    Q 289+394 T 683  [91m☒[0m 783 
    Q 759+185 T 944  [92m☑[0m 944 
    Q 33+333  T 366  [92m☑[0m 366 
    Q 402+578 T 980  [92m☑[0m 980 
    Q 789+889 T 1678 [92m☑[0m 1678
    Q 612+22  T 634  [92m☑[0m 634 
    Q 13+638  T 651  [92m☑[0m 651 
    Q 93+797  T 890  [91m☒[0m 880 
    Q 285+31  T 316  [92m☑[0m 316 
    
    --------------------------------------------------
    Iteration 56
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 6s 314us/step - loss: 0.1316 - acc: 0.9670 - val_loss: 0.1573 - val_acc: 0.9519
    Q 94+315  T 409  [92m☑[0m 409 
    Q 56+949  T 1005 [92m☑[0m 1005
    Q 935+50  T 985  [92m☑[0m 985 
    Q 896+823 T 1719 [91m☒[0m 1710
    Q 43+790  T 833  [92m☑[0m 833 
    Q 60+78   T 138  [92m☑[0m 138 
    Q 894+482 T 1376 [92m☑[0m 1376
    Q 551+73  T 624  [92m☑[0m 624 
    Q 57+207  T 264  [92m☑[0m 264 
    Q 2+97    T 99   [91m☒[0m 108 
    
    --------------------------------------------------
    Iteration 57
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 273us/step - loss: 0.1265 - acc: 0.9676 - val_loss: 0.1802 - val_acc: 0.9410
    Q 282+25  T 307  [92m☑[0m 307 
    Q 453+355 T 808  [92m☑[0m 808 
    Q 15+982  T 997  [92m☑[0m 997 
    Q 14+51   T 65   [92m☑[0m 65  
    Q 224+6   T 230  [92m☑[0m 230 
    Q 30+3    T 33   [92m☑[0m 33  
    Q 758+277 T 1035 [92m☑[0m 1035
    Q 37+973  T 1010 [92m☑[0m 1010
    Q 893+604 T 1497 [92m☑[0m 1497
    Q 41+92   T 133  [91m☒[0m 123 
    
    --------------------------------------------------
    Iteration 58
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 242us/step - loss: 0.1184 - acc: 0.9710 - val_loss: 0.1792 - val_acc: 0.9408
    Q 858+0   T 858  [92m☑[0m 858 
    Q 516+464 T 980  [91m☒[0m 970 
    Q 345+77  T 422  [92m☑[0m 422 
    Q 392+3   T 395  [92m☑[0m 395 
    Q 276+70  T 346  [92m☑[0m 346 
    Q 48+764  T 812  [92m☑[0m 812 
    Q 61+338  T 399  [92m☑[0m 399 
    Q 354+5   T 359  [92m☑[0m 359 
    Q 465+978 T 1443 [92m☑[0m 1443
    Q 28+319  T 347  [92m☑[0m 347 
    
    --------------------------------------------------
    Iteration 59
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.1141 - acc: 0.9711 - val_loss: 0.1718 - val_acc: 0.9410
    Q 57+652  T 709  [92m☑[0m 709 
    Q 85+294  T 379  [92m☑[0m 379 
    Q 851+723 T 1574 [92m☑[0m 1574
    Q 893+604 T 1497 [92m☑[0m 1497
    Q 390+71  T 461  [92m☑[0m 461 
    Q 523+78  T 601  [92m☑[0m 601 
    Q 939+177 T 1116 [92m☑[0m 1116
    Q 98+91   T 189  [91m☒[0m 180 
    Q 889+2   T 891  [92m☑[0m 891 
    Q 131+83  T 214  [92m☑[0m 214 
    
    --------------------------------------------------
    Iteration 60
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 272us/step - loss: 0.1101 - acc: 0.9729 - val_loss: 0.1829 - val_acc: 0.9334
    Q 63+7    T 70   [92m☑[0m 70  
    Q 48+764  T 812  [92m☑[0m 812 
    Q 997+834 T 1831 [91m☒[0m 1830
    Q 247+255 T 502  [91m☒[0m 511 
    Q 50+394  T 444  [92m☑[0m 444 
    Q 0+871   T 871  [91m☒[0m 861 
    Q 407+773 T 1180 [92m☑[0m 1180
    Q 183+89  T 272  [91m☒[0m 271 
    Q 928+15  T 943  [92m☑[0m 943 
    Q 907+79  T 986  [91m☒[0m 985 
    
    --------------------------------------------------
    Iteration 61
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 223us/step - loss: 0.1058 - acc: 0.9735 - val_loss: 0.2251 - val_acc: 0.9177
    Q 95+845  T 940  [91m☒[0m 941 
    Q 889+35  T 924  [92m☑[0m 924 
    Q 850+418 T 1268 [91m☒[0m 1269
    Q 659+698 T 1357 [91m☒[0m 1347
    Q 56+751  T 807  [92m☑[0m 807 
    Q 86+234  T 320  [92m☑[0m 320 
    Q 55+357  T 412  [92m☑[0m 412 
    Q 549+955 T 1504 [92m☑[0m 1504
    Q 828+68  T 896  [91m☒[0m 996 
    Q 931+437 T 1368 [91m☒[0m 1369
    
    --------------------------------------------------
    Iteration 62
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 229us/step - loss: 0.0999 - acc: 0.9756 - val_loss: 0.1477 - val_acc: 0.9501
    Q 10+76   T 86   [91m☒[0m 87  
    Q 32+219  T 251  [92m☑[0m 251 
    Q 243+89  T 332  [92m☑[0m 332 
    Q 315+329 T 644  [92m☑[0m 644 
    Q 118+613 T 731  [91m☒[0m 721 
    Q 1+578   T 579  [92m☑[0m 579 
    Q 873+731 T 1604 [92m☑[0m 1604
    Q 893+953 T 1846 [92m☑[0m 1846
    Q 96+496  T 592  [92m☑[0m 592 
    Q 46+264  T 310  [92m☑[0m 310 
    
    --------------------------------------------------
    Iteration 63
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 229us/step - loss: 0.0984 - acc: 0.9757 - val_loss: 0.1335 - val_acc: 0.9566
    Q 169+72  T 241  [92m☑[0m 241 
    Q 70+41   T 111  [92m☑[0m 111 
    Q 33+924  T 957  [92m☑[0m 957 
    Q 52+823  T 875  [92m☑[0m 875 
    Q 62+527  T 589  [92m☑[0m 589 
    Q 3+369   T 372  [92m☑[0m 372 
    Q 680+785 T 1465 [92m☑[0m 1465
    Q 601+79  T 680  [92m☑[0m 680 
    Q 287+14  T 301  [91m☒[0m 201 
    Q 97+618  T 715  [92m☑[0m 715 
    
    --------------------------------------------------
    Iteration 64
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 227us/step - loss: 0.0926 - acc: 0.9771 - val_loss: 0.1212 - val_acc: 0.9627
    Q 46+264  T 310  [92m☑[0m 310 
    Q 15+195  T 210  [92m☑[0m 210 
    Q 100+680 T 780  [91m☒[0m 770 
    Q 666+361 T 1027 [92m☑[0m 1027
    Q 86+804  T 890  [91m☒[0m 880 
    Q 601+814 T 1415 [92m☑[0m 1415
    Q 12+823  T 835  [92m☑[0m 835 
    Q 765+520 T 1285 [92m☑[0m 1285
    Q 324+26  T 350  [92m☑[0m 350 
    Q 837+65  T 902  [92m☑[0m 902 
    
    --------------------------------------------------
    Iteration 65
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 271us/step - loss: 0.0870 - acc: 0.9791 - val_loss: 0.2506 - val_acc: 0.9014
    Q 196+28  T 224  [92m☑[0m 224 
    Q 928+234 T 1162 [92m☑[0m 1162
    Q 208+7   T 215  [92m☑[0m 215 
    Q 883+85  T 968  [92m☑[0m 968 
    Q 410+755 T 1165 [92m☑[0m 1165
    Q 37+565  T 602  [92m☑[0m 602 
    Q 50+133  T 183  [91m☒[0m 173 
    Q 92+986  T 1078 [92m☑[0m 1078
    Q 30+310  T 340  [92m☑[0m 340 
    Q 619+8   T 627  [92m☑[0m 627 
    
    --------------------------------------------------
    Iteration 66
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 263us/step - loss: 0.0890 - acc: 0.9770 - val_loss: 0.1130 - val_acc: 0.9666
    Q 525+65  T 590  [92m☑[0m 590 
    Q 83+409  T 492  [92m☑[0m 492 
    Q 11+14   T 25   [92m☑[0m 25  
    Q 732+922 T 1654 [91m☒[0m 1644
    Q 551+569 T 1120 [92m☑[0m 1120
    Q 3+213   T 216  [92m☑[0m 216 
    Q 749+24  T 773  [92m☑[0m 773 
    Q 475+14  T 489  [92m☑[0m 489 
    Q 357+43  T 400  [92m☑[0m 400 
    Q 86+234  T 320  [92m☑[0m 320 
    
    --------------------------------------------------
    Iteration 67
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 280us/step - loss: 0.0842 - acc: 0.9797 - val_loss: 0.1214 - val_acc: 0.9609
    Q 91+0    T 91   [92m☑[0m 91  
    Q 34+909  T 943  [92m☑[0m 943 
    Q 16+81   T 97   [91m☒[0m 17  
    Q 82+151  T 233  [92m☑[0m 233 
    Q 159+544 T 703  [92m☑[0m 703 
    Q 41+270  T 311  [92m☑[0m 311 
    Q 800+624 T 1424 [92m☑[0m 1424
    Q 79+490  T 569  [91m☒[0m 579 
    Q 807+889 T 1696 [91m☒[0m 1796
    Q 120+639 T 759  [92m☑[0m 759 
    
    --------------------------------------------------
    Iteration 68
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 231us/step - loss: 0.0798 - acc: 0.9815 - val_loss: 0.1453 - val_acc: 0.9510
    Q 75+434  T 509  [92m☑[0m 509 
    Q 239+930 T 1169 [91m☒[0m 1159
    Q 851+84  T 935  [91m☒[0m 936 
    Q 850+28  T 878  [91m☒[0m 868 
    Q 74+69   T 143  [92m☑[0m 143 
    Q 560+46  T 606  [92m☑[0m 606 
    Q 388+361 T 749  [92m☑[0m 749 
    Q 28+785  T 813  [92m☑[0m 813 
    Q 561+43  T 604  [92m☑[0m 604 
    Q 187+56  T 243  [92m☑[0m 243 
    
    --------------------------------------------------
    Iteration 69
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 234us/step - loss: 0.0733 - acc: 0.9836 - val_loss: 0.1649 - val_acc: 0.9405
    Q 46+745  T 791  [92m☑[0m 791 
    Q 42+639  T 681  [92m☑[0m 681 
    Q 474+835 T 1309 [91m☒[0m 1319
    Q 32+367  T 399  [91m☒[0m 499 
    Q 39+398  T 437  [92m☑[0m 437 
    Q 439+18  T 457  [92m☑[0m 457 
    Q 14+328  T 342  [92m☑[0m 342 
    Q 680+50  T 730  [92m☑[0m 730 
    Q 33+390  T 423  [92m☑[0m 423 
    Q 584+59  T 643  [92m☑[0m 643 
    
    --------------------------------------------------
    Iteration 70
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 237us/step - loss: 0.0757 - acc: 0.9821 - val_loss: 0.3080 - val_acc: 0.8938
    Q 906+684 T 1590 [91m☒[0m 1591
    Q 86+389  T 475  [91m☒[0m 476 
    Q 90+76   T 166  [91m☒[0m 167 
    Q 237+684 T 921  [92m☑[0m 921 
    Q 885+62  T 947  [91m☒[0m 948 
    Q 721+74  T 795  [92m☑[0m 795 
    Q 721+81  T 802  [91m☒[0m 702 
    Q 15+839  T 854  [92m☑[0m 854 
    Q 39+978  T 1017 [92m☑[0m 1017
    Q 893+58  T 951  [91m☒[0m 952 
    
    --------------------------------------------------
    Iteration 71
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 250us/step - loss: 0.0709 - acc: 0.9833 - val_loss: 0.1056 - val_acc: 0.9691
    Q 7+802   T 809  [92m☑[0m 809 
    Q 31+958  T 989  [92m☑[0m 989 
    Q 45+193  T 238  [92m☑[0m 238 
    Q 66+75   T 141  [92m☑[0m 141 
    Q 384+806 T 1190 [92m☑[0m 1190
    Q 46+85   T 131  [92m☑[0m 131 
    Q 0+716   T 716  [92m☑[0m 716 
    Q 882+58  T 940  [92m☑[0m 940 
    Q 647+48  T 695  [92m☑[0m 695 
    Q 92+813  T 905  [92m☑[0m 905 
    
    --------------------------------------------------
    Iteration 72
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.0678 - acc: 0.9847 - val_loss: 0.1289 - val_acc: 0.9553
    Q 194+12  T 206  [92m☑[0m 206 
    Q 402+122 T 524  [92m☑[0m 524 
    Q 75+520  T 595  [92m☑[0m 595 
    Q 1+563   T 564  [92m☑[0m 564 
    Q 84+637  T 721  [92m☑[0m 721 
    Q 850+823 T 1673 [91m☒[0m 1672
    Q 652+254 T 906  [92m☑[0m 906 
    Q 401+1   T 402  [91m☒[0m 401 
    Q 640+52  T 692  [92m☑[0m 692 
    Q 4+42    T 46   [91m☒[0m 47  
    
    --------------------------------------------------
    Iteration 73
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 232us/step - loss: 0.0691 - acc: 0.9842 - val_loss: 0.0995 - val_acc: 0.9690
    Q 41+102  T 143  [92m☑[0m 143 
    Q 425+31  T 456  [92m☑[0m 456 
    Q 345+77  T 422  [92m☑[0m 422 
    Q 941+944 T 1885 [92m☑[0m 1885
    Q 286+775 T 1061 [91m☒[0m 1051
    Q 514+571 T 1085 [92m☑[0m 1085
    Q 89+43   T 132  [92m☑[0m 132 
    Q 3+376   T 379  [92m☑[0m 379 
    Q 873+114 T 987  [92m☑[0m 987 
    Q 514+14  T 528  [92m☑[0m 528 
    
    --------------------------------------------------
    Iteration 74
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 237us/step - loss: 0.0638 - acc: 0.9857 - val_loss: 0.1906 - val_acc: 0.9286
    Q 161+381 T 542  [92m☑[0m 542 
    Q 58+86   T 144  [92m☑[0m 144 
    Q 237+795 T 1032 [92m☑[0m 1032
    Q 282+445 T 727  [92m☑[0m 727 
    Q 92+813  T 905  [92m☑[0m 905 
    Q 599+944 T 1543 [92m☑[0m 1543
    Q 20+890  T 910  [92m☑[0m 910 
    Q 60+565  T 625  [92m☑[0m 625 
    Q 956+935 T 1891 [91m☒[0m 1892
    Q 759+185 T 944  [92m☑[0m 944 
    
    --------------------------------------------------
    Iteration 75
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 228us/step - loss: 0.0647 - acc: 0.9842 - val_loss: 0.1031 - val_acc: 0.9688
    Q 73+767  T 840  [92m☑[0m 840 
    Q 237+397 T 634  [92m☑[0m 634 
    Q 432+7   T 439  [92m☑[0m 439 
    Q 772+981 T 1753 [92m☑[0m 1753
    Q 957+710 T 1667 [92m☑[0m 1667
    Q 273+93  T 366  [92m☑[0m 366 
    Q 81+889  T 970  [91m☒[0m 960 
    Q 44+27   T 71   [92m☑[0m 71  
    Q 90+76   T 166  [92m☑[0m 166 
    Q 290+437 T 727  [92m☑[0m 727 
    
    --------------------------------------------------
    Iteration 76
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 233us/step - loss: 0.0630 - acc: 0.9852 - val_loss: 0.1000 - val_acc: 0.9689
    Q 33+658  T 691  [92m☑[0m 691 
    Q 925+82  T 1007 [92m☑[0m 1007
    Q 441+382 T 823  [92m☑[0m 823 
    Q 11+14   T 25   [92m☑[0m 25  
    Q 26+329  T 355  [92m☑[0m 355 
    Q 20+23   T 43   [92m☑[0m 43  
    Q 612+22  T 634  [92m☑[0m 634 
    Q 506+14  T 520  [92m☑[0m 520 
    Q 755+426 T 1181 [92m☑[0m 1181
    Q 941+944 T 1885 [92m☑[0m 1885
    
    --------------------------------------------------
    Iteration 77
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 294us/step - loss: 0.0557 - acc: 0.9883 - val_loss: 0.3344 - val_acc: 0.8906
    Q 37+588  T 625  [92m☑[0m 625 
    Q 395+84  T 479  [92m☑[0m 479 
    Q 358+181 T 539  [91m☒[0m 549 
    Q 69+108  T 177  [92m☑[0m 177 
    Q 275+58  T 333  [91m☒[0m 332 
    Q 652+477 T 1129 [92m☑[0m 1129
    Q 772+496 T 1268 [92m☑[0m 1268
    Q 149+84  T 233  [92m☑[0m 233 
    Q 45+193  T 238  [91m☒[0m 237 
    Q 370+35  T 405  [91m☒[0m 404 
    
    --------------------------------------------------
    Iteration 78
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.0579 - acc: 0.9865 - val_loss: 0.1852 - val_acc: 0.9414
    Q 29+211  T 240  [91m☒[0m 230 
    Q 997+446 T 1443 [92m☑[0m 1443
    Q 494+354 T 848  [92m☑[0m 848 
    Q 225+58  T 283  [92m☑[0m 283 
    Q 953+811 T 1764 [92m☑[0m 1764
    Q 15+629  T 644  [92m☑[0m 644 
    Q 81+52   T 133  [92m☑[0m 133 
    Q 34+100  T 134  [92m☑[0m 134 
    Q 65+638  T 703  [91m☒[0m 603 
    Q 65+678  T 743  [92m☑[0m 743 
    
    --------------------------------------------------
    Iteration 79
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 256us/step - loss: 0.0562 - acc: 0.9874 - val_loss: 0.0989 - val_acc: 0.9686
    Q 284+387 T 671  [92m☑[0m 671 
    Q 27+52   T 79   [92m☑[0m 79  
    Q 99+695  T 794  [92m☑[0m 794 
    Q 772+88  T 860  [91m☒[0m 850 
    Q 449+938 T 1387 [92m☑[0m 1387
    Q 61+142  T 203  [92m☑[0m 203 
    Q 55+2    T 57   [91m☒[0m 56  
    Q 308+88  T 396  [92m☑[0m 396 
    Q 30+530  T 560  [92m☑[0m 560 
    Q 321+15  T 336  [92m☑[0m 336 
    
    --------------------------------------------------
    Iteration 80
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.0559 - acc: 0.9866 - val_loss: 0.0939 - val_acc: 0.9724
    Q 980+62  T 1042 [92m☑[0m 1042
    Q 533+62  T 595  [92m☑[0m 595 
    Q 51+962  T 1013 [92m☑[0m 1013
    Q 58+662  T 720  [92m☑[0m 720 
    Q 275+2   T 277  [92m☑[0m 277 
    Q 628+1   T 629  [92m☑[0m 629 
    Q 52+545  T 597  [92m☑[0m 597 
    Q 70+49   T 119  [92m☑[0m 119 
    Q 83+92   T 175  [92m☑[0m 175 
    Q 9+663   T 672  [92m☑[0m 672 
    
    --------------------------------------------------
    Iteration 81
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 267us/step - loss: 0.0494 - acc: 0.9890 - val_loss: 0.1451 - val_acc: 0.9478
    Q 66+431  T 497  [91m☒[0m 487 
    Q 772+981 T 1753 [92m☑[0m 1753
    Q 12+14   T 26   [92m☑[0m 26  
    Q 84+990  T 1074 [92m☑[0m 1074
    Q 233+706 T 939  [91m☒[0m 949 
    Q 635+46  T 681  [92m☑[0m 681 
    Q 108+251 T 359  [92m☑[0m 359 
    Q 545+581 T 1126 [92m☑[0m 1126
    Q 41+19   T 60   [91m☒[0m 59  
    Q 465+25  T 490  [91m☒[0m 480 
    
    --------------------------------------------------
    Iteration 82
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 246us/step - loss: 0.0559 - acc: 0.9865 - val_loss: 0.1086 - val_acc: 0.9632
    Q 348+23  T 371  [92m☑[0m 371 
    Q 533+6   T 539  [92m☑[0m 539 
    Q 794+9   T 803  [91m☒[0m 804 
    Q 118+613 T 731  [91m☒[0m 721 
    Q 91+698  T 789  [92m☑[0m 789 
    Q 30+530  T 560  [92m☑[0m 560 
    Q 983+100 T 1083 [92m☑[0m 1083
    Q 28+14   T 42   [92m☑[0m 42  
    Q 18+938  T 956  [92m☑[0m 956 
    Q 807+889 T 1696 [91m☒[0m 1796
    
    --------------------------------------------------
    Iteration 83
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 250us/step - loss: 0.0452 - acc: 0.9902 - val_loss: 0.0955 - val_acc: 0.9698
    Q 635+583 T 1218 [92m☑[0m 1218
    Q 908+88  T 996  [92m☑[0m 996 
    Q 634+311 T 945  [92m☑[0m 945 
    Q 43+21   T 64   [92m☑[0m 64  
    Q 80+674  T 754  [92m☑[0m 754 
    Q 72+180  T 252  [92m☑[0m 252 
    Q 27+826  T 853  [92m☑[0m 853 
    Q 110+754 T 864  [92m☑[0m 864 
    Q 902+89  T 991  [91m☒[0m 992 
    Q 37+293  T 330  [92m☑[0m 330 
    
    --------------------------------------------------
    Iteration 84
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.0528 - acc: 0.9871 - val_loss: 0.0799 - val_acc: 0.9754
    Q 696+872 T 1568 [92m☑[0m 1568
    Q 920+24  T 944  [92m☑[0m 944 
    Q 678+95  T 773  [92m☑[0m 773 
    Q 3+445   T 448  [92m☑[0m 448 
    Q 351+655 T 1006 [92m☑[0m 1006
    Q 264+389 T 653  [92m☑[0m 653 
    Q 111+967 T 1078 [92m☑[0m 1078
    Q 850+0   T 850  [92m☑[0m 850 
    Q 694+73  T 767  [92m☑[0m 767 
    Q 721+80  T 801  [92m☑[0m 801 
    
    --------------------------------------------------
    Iteration 85
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 290us/step - loss: 0.0450 - acc: 0.9898 - val_loss: 0.1107 - val_acc: 0.9636
    Q 734+478 T 1212 [92m☑[0m 1212
    Q 634+311 T 945  [92m☑[0m 945 
    Q 69+108  T 177  [91m☒[0m 176 
    Q 27+826  T 853  [92m☑[0m 853 
    Q 4+42    T 46   [91m☒[0m 47  
    Q 141+49  T 190  [91m☒[0m 180 
    Q 75+434  T 509  [92m☑[0m 509 
    Q 52+390  T 442  [92m☑[0m 442 
    Q 272+401 T 673  [92m☑[0m 673 
    Q 930+378 T 1308 [91m☒[0m 1208
    
    --------------------------------------------------
    Iteration 86
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 269us/step - loss: 0.0484 - acc: 0.9884 - val_loss: 0.0977 - val_acc: 0.9660
    Q 58+125  T 183  [92m☑[0m 183 
    Q 294+171 T 465  [92m☑[0m 465 
    Q 406+6   T 412  [92m☑[0m 412 
    Q 6+372   T 378  [92m☑[0m 378 
    Q 312+33  T 345  [92m☑[0m 345 
    Q 9+971   T 980  [92m☑[0m 980 
    Q 587+81  T 668  [92m☑[0m 668 
    Q 66+573  T 639  [92m☑[0m 639 
    Q 27+908  T 935  [92m☑[0m 935 
    Q 41+102  T 143  [92m☑[0m 143 
    
    --------------------------------------------------
    Iteration 87
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 266us/step - loss: 0.0413 - acc: 0.9914 - val_loss: 0.0881 - val_acc: 0.9720
    Q 83+290  T 373  [91m☒[0m 473 
    Q 52+823  T 875  [92m☑[0m 875 
    Q 329+0   T 329  [92m☑[0m 329 
    Q 800+624 T 1424 [92m☑[0m 1424
    Q 33+191  T 224  [92m☑[0m 224 
    Q 26+527  T 553  [92m☑[0m 553 
    Q 26+168  T 194  [92m☑[0m 194 
    Q 731+16  T 747  [92m☑[0m 747 
    Q 63+992  T 1055 [92m☑[0m 1055
    Q 232+717 T 949  [92m☑[0m 949 
    
    --------------------------------------------------
    Iteration 88
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 260us/step - loss: 0.0471 - acc: 0.9894 - val_loss: 0.1108 - val_acc: 0.9655
    Q 58+652  T 710  [92m☑[0m 710 
    Q 835+304 T 1139 [92m☑[0m 1139
    Q 525+65  T 590  [92m☑[0m 590 
    Q 788+308 T 1096 [91m☒[0m 1006
    Q 263+366 T 629  [92m☑[0m 629 
    Q 243+89  T 332  [92m☑[0m 332 
    Q 382+927 T 1309 [91m☒[0m 1319
    Q 52+205  T 257  [92m☑[0m 257 
    Q 28+21   T 49   [92m☑[0m 49  
    Q 163+161 T 324  [92m☑[0m 324 
    
    --------------------------------------------------
    Iteration 89
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 264us/step - loss: 0.0403 - acc: 0.9904 - val_loss: 0.0823 - val_acc: 0.9737
    Q 618+99  T 717  [92m☑[0m 717 
    Q 98+91   T 189  [92m☑[0m 189 
    Q 68+75   T 143  [92m☑[0m 143 
    Q 567+47  T 614  [92m☑[0m 614 
    Q 82+961  T 1043 [92m☑[0m 1043
    Q 67+190  T 257  [92m☑[0m 257 
    Q 131+610 T 741  [91m☒[0m 731 
    Q 935+781 T 1716 [92m☑[0m 1716
    Q 63+598  T 661  [92m☑[0m 661 
    Q 64+203  T 267  [92m☑[0m 267 
    
    --------------------------------------------------
    Iteration 90
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 259us/step - loss: 0.0435 - acc: 0.9909 - val_loss: 0.0748 - val_acc: 0.9762
    Q 7+266   T 273  [92m☑[0m 273 
    Q 528+90  T 618  [92m☑[0m 618 
    Q 28+593  T 621  [92m☑[0m 621 
    Q 63+598  T 661  [92m☑[0m 661 
    Q 810+8   T 818  [91m☒[0m 808 
    Q 255+503 T 758  [92m☑[0m 758 
    Q 481+499 T 980  [92m☑[0m 980 
    Q 83+792  T 875  [92m☑[0m 875 
    Q 529+84  T 613  [92m☑[0m 613 
    Q 795+47  T 842  [92m☑[0m 842 
    
    --------------------------------------------------
    Iteration 91
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 254us/step - loss: 0.0408 - acc: 0.9902 - val_loss: 0.0782 - val_acc: 0.9758
    Q 495+13  T 508  [92m☑[0m 508 
    Q 964+3   T 967  [92m☑[0m 967 
    Q 827+750 T 1577 [92m☑[0m 1577
    Q 94+614  T 708  [92m☑[0m 708 
    Q 874+922 T 1796 [91m☒[0m 1896
    Q 9+841   T 850  [92m☑[0m 850 
    Q 777+951 T 1728 [92m☑[0m 1728
    Q 120+62  T 182  [92m☑[0m 182 
    Q 11+206  T 217  [92m☑[0m 217 
    Q 222+51  T 273  [92m☑[0m 273 
    
    --------------------------------------------------
    Iteration 92
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 257us/step - loss: 0.0418 - acc: 0.9909 - val_loss: 0.0877 - val_acc: 0.9700
    Q 922+343 T 1265 [92m☑[0m 1265
    Q 150+98  T 248  [92m☑[0m 248 
    Q 63+308  T 371  [92m☑[0m 371 
    Q 237+795 T 1032 [92m☑[0m 1032
    Q 546+831 T 1377 [92m☑[0m 1377
    Q 98+846  T 944  [92m☑[0m 944 
    Q 314+81  T 395  [92m☑[0m 395 
    Q 948+34  T 982  [92m☑[0m 982 
    Q 34+935  T 969  [92m☑[0m 969 
    Q 14+364  T 378  [92m☑[0m 378 
    
    --------------------------------------------------
    Iteration 93
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 258us/step - loss: 0.0381 - acc: 0.9914 - val_loss: 0.0953 - val_acc: 0.9690
    Q 228+79  T 307  [92m☑[0m 307 
    Q 444+472 T 916  [92m☑[0m 916 
    Q 33+234  T 267  [92m☑[0m 267 
    Q 707+33  T 740  [92m☑[0m 740 
    Q 10+157  T 167  [92m☑[0m 167 
    Q 36+983  T 1019 [92m☑[0m 1019
    Q 621+17  T 638  [92m☑[0m 638 
    Q 494+874 T 1368 [92m☑[0m 1368
    Q 351+655 T 1006 [92m☑[0m 1006
    Q 0+119   T 119  [91m☒[0m 1100
    
    --------------------------------------------------
    Iteration 94
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 246us/step - loss: 0.0384 - acc: 0.9917 - val_loss: 0.1023 - val_acc: 0.9670
    Q 63+634  T 697  [92m☑[0m 697 
    Q 4+788   T 792  [92m☑[0m 792 
    Q 682+56  T 738  [92m☑[0m 738 
    Q 29+930  T 959  [92m☑[0m 959 
    Q 9+663   T 672  [92m☑[0m 672 
    Q 31+327  T 358  [92m☑[0m 358 
    Q 43+281  T 324  [92m☑[0m 324 
    Q 117+67  T 184  [92m☑[0m 184 
    Q 937+5   T 942  [92m☑[0m 942 
    Q 414+72  T 486  [92m☑[0m 486 
    
    --------------------------------------------------
    Iteration 95
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 261us/step - loss: 0.0367 - acc: 0.9926 - val_loss: 0.2545 - val_acc: 0.9246
    Q 86+389  T 475  [92m☑[0m 475 
    Q 52+560  T 612  [92m☑[0m 612 
    Q 37+293  T 330  [92m☑[0m 330 
    Q 653+310 T 963  [91m☒[0m 973 
    Q 86+804  T 890  [92m☑[0m 890 
    Q 449+938 T 1387 [91m☒[0m 1397
    Q 0+64    T 64   [92m☑[0m 64  
    Q 93+109  T 202  [92m☑[0m 202 
    Q 896+94  T 990  [92m☑[0m 990 
    Q 870+881 T 1751 [92m☑[0m 1751
    
    --------------------------------------------------
    Iteration 96
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 243us/step - loss: 0.0370 - acc: 0.9920 - val_loss: 0.0742 - val_acc: 0.9756
    Q 1+688   T 689  [92m☑[0m 689 
    Q 489+66  T 555  [92m☑[0m 555 
    Q 534+3   T 537  [92m☑[0m 537 
    Q 666+361 T 1027 [92m☑[0m 1027
    Q 714+445 T 1159 [91m☒[0m 1169
    Q 664+33  T 697  [92m☑[0m 697 
    Q 783+239 T 1022 [92m☑[0m 1022
    Q 546+61  T 607  [92m☑[0m 607 
    Q 494+178 T 672  [91m☒[0m 682 
    Q 58+982  T 1040 [92m☑[0m 1040
    
    --------------------------------------------------
    Iteration 97
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 239us/step - loss: 0.0349 - acc: 0.9924 - val_loss: 0.0736 - val_acc: 0.9765
    Q 31+79   T 110  [92m☑[0m 110 
    Q 407+773 T 1180 [92m☑[0m 1180
    Q 392+45  T 437  [92m☑[0m 437 
    Q 404+79  T 483  [92m☑[0m 483 
    Q 99+856  T 955  [92m☑[0m 955 
    Q 80+575  T 655  [92m☑[0m 655 
    Q 94+493  T 587  [92m☑[0m 587 
    Q 64+667  T 731  [92m☑[0m 731 
    Q 555+27  T 582  [92m☑[0m 582 
    Q 799+2   T 801  [92m☑[0m 801 
    
    --------------------------------------------------
    Iteration 98
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 4s 249us/step - loss: 0.0327 - acc: 0.9923 - val_loss: 0.0808 - val_acc: 0.9741
    Q 35+985  T 1020 [92m☑[0m 1020
    Q 986+9   T 995  [92m☑[0m 995 
    Q 44+27   T 71   [92m☑[0m 71  
    Q 190+4   T 194  [92m☑[0m 194 
    Q 91+43   T 134  [92m☑[0m 134 
    Q 815+96  T 911  [92m☑[0m 911 
    Q 227+241 T 468  [92m☑[0m 468 
    Q 656+95  T 751  [92m☑[0m 751 
    Q 86+281  T 367  [92m☑[0m 367 
    Q 307+911 T 1218 [92m☑[0m 1218
    
    --------------------------------------------------
    Iteration 99
    Train on 18000 samples, validate on 2000 samples
    Epoch 1/1
    18000/18000 [==============================] - 5s 262us/step - loss: 0.0389 - acc: 0.9914 - val_loss: 0.0908 - val_acc: 0.9704
    Q 935+829 T 1764 [92m☑[0m 1764
    Q 309+759 T 1068 [92m☑[0m 1068
    Q 835+628 T 1463 [92m☑[0m 1463
    Q 7+802   T 809  [92m☑[0m 809 
    Q 642+192 T 834  [91m☒[0m 824 
    Q 82+611  T 693  [92m☑[0m 693 
    Q 14+22   T 36   [92m☑[0m 36  
    Q 91+56   T 147  [92m☑[0m 147 
    Q 354+91  T 445  [92m☑[0m 445 
    Q 834+214 T 1048 [92m☑[0m 1048
    

# Testing 1 (by test data)


```python
print("MSG : Prediction")
preds = model.predict_classes(test_x)
for i in range(10):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
```

    MSG : Prediction
    Q 415+13  T 428  [92m☑[0m 428 
    Q 35+76   T 111  [92m☑[0m 111 
    Q 157+28  T 185  [92m☑[0m 185 
    Q 74+606  T 680  [92m☑[0m 680 
    Q 122+67  T 189  [92m☑[0m 189 
    Q 707+47  T 754  [92m☑[0m 754 
    Q 60+257  T 317  [92m☑[0m 317 
    Q 102+160 T 262  [91m☒[0m 271 
    Q 417+308 T 725  [92m☑[0m 725 
    Q 404+359 T 763  [92m☑[0m 763 
    

# Testing 2 (by new question)
Q: 555+175, 860+7  , 340+29


```python
newQ = ['555+175', '860+7  ', '340+29 ']
newA = ['730 ', '867 ', '369 ']
print('Vectorization...')
x = np.zeros((len(newQ), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(newA), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(newQ):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(newA):
    y[i] = ctable.encode(sentence, DIGITS + 1)
    
print("MSG : Prediction")
for i in range(len(newQ)):
    preds = model.predict_classes(x)
    q = ctable.decode(x[i])
    correct = ctable.decode(y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
```

    Vectorization...
    MSG : Prediction
    Q 555+175 T 730  [92m☑[0m 730 
    Q 860+7   T 867  [92m☑[0m 867 
    Q 340+29  T 369  [92m☑[0m 369 
    


```python

```
