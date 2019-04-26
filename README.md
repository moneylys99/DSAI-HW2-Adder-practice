
# DSAI HW2: Adder & Subtractor Practice by LSTM     
## Chien, Hsin Yen
### RE6071088, Institute of Data Science  
### https://nbviewer.jupyter.org/github/moneylys99/DSAI-HW2-Adder-practice/blob/master/Adder_Subtractor_re6071088.ipynb

Data Generation: 80000 data for adder, 80000 data for subtractor  
Digits available: <= 3  
LSTM:  
1 hidden layer  
Hidden layer size = 128  
Batch size = 128  
training epoch = 100  

# Import package


```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
from six.moves import range
```

    Using TensorFlow backend.
    

# Parameters Config


```python
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
```


```python
TRAINING_SIZE =160000
DIGITS = 3
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+- '
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
     2: '-',
     3: '0',
     4: '1',
     5: '2',
     6: '3',
     7: '4',
     8: '5',
     9: '6',
     10: '7',
     11: '8',
     12: '9'}



# Data Generation

### Generating data for adder


```python
questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE/2:
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
```

    Generating data...
    

### Generating data for subtractor


```python
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total questions:', len(questions))
```

    Total questions: 160000
    


```python
print(questions[:3], expected[:3])
print(questions[150000:150003], expected[150000:150003])
```

    ['8+0    ', '502+976', '9+570  '] ['8   ', '1478', '579 ']
    ['537-724', '160-752', '33-620 '] ['-187', '-592', '-587']
    

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
train_x = np.concatenate((x[:20000], x[80000:100000]), axis=0)
train_y = np.concatenate((y[:20000], y[80000:100000]),axis=0)
test_x = np.concatenate((x[20000:80000], x[100000:]),axis=0)
test_y = np.concatenate((y[20000:80000], y[100000:]),axis=0)

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
    (36000, 7, 13)
    (36000, 4, 13)
    Validation Data:
    (4000, 7, 13)
    (4000, 4, 13)
    Testing Data:
    (120000, 7, 13)
    (120000, 4, 13)
    


```python
print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])
```

    input:  [[[False False False False False False False False False False False
       False  True]
      [False False False False False False False False False False False
       False  True]
      [False False False False False  True False False False False False
       False False]
      [False  True False False False False False False False False False
       False False]
      [False False False False False  True False False False False False
       False False]
      [False False False False False  True False False False False False
       False False]
      [ True False False False False False False False False False False
       False False]]
    
     [[False False False False False False False False False False False
       False  True]
      [False False False False False False False False  True False False
       False False]
      [False False False False False False False False False  True False
       False False]
      [False  True False False False False False False False False False
       False False]
      [False False False False False False False False False  True False
       False False]
      [False False False False False False False False False False False
       False  True]
      [False False False False False False False False False False False
       False  True]]
    
     [[False False False False False False False False False False False
       False  True]
      [False False False  True False False False False False False False
       False False]
      [False False False False False False False  True False False False
       False False]
      [False  True False False False False False False False False False
       False False]
      [False False False False False False  True False False False False
       False False]
      [False False False  True False False False False False False False
       False False]
      [False False False False False False False  True False False False
       False False]]] 
    
     label:  [[[False False False False  True False False False False False False
       False False]
      [False False False  True False False False False False False False
       False False]
      [False False False False  True False False False False False False
       False False]
      [False False False False False False False  True False False False
       False False]]
    
     [[False False False False  True False False False False False False
       False False]
      [False False False False False False False False False  True False
       False False]
      [False False False False False False False False  True False False
       False False]
      [False False False False False False False False  True False False
       False False]]
    
     [[False False False False  True False False False False False False
       False False]
      [False False False False False  True False False False False False
       False False]
      [False False False  True False False False False False False False
       False False]
      [False False False False False False False False False False False
        True False]]]
    

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
    lstm_1 (LSTM)                (None, 128)               72704     
    _________________________________________________________________
    repeat_vector_1 (RepeatVecto (None, 4, 128)            0         
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 4, 128)            131584    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 4, 13)             1677      
    =================================================================
    Total params: 205,965
    Trainable params: 205,965
    Non-trainable params: 0
    _________________________________________________________________
    

# Training
Combine adder and substractor  
Training epoch = 100  


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
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 212us/step - loss: 2.0607 - acc: 0.2932 - val_loss: 1.8962 - val_acc: 0.3249
    Q 459+764 T 1223 [91m☒[0m 110 
    Q 760-83  T 677  [91m☒[0m 13  
    Q 378-940 T -562 [91m☒[0m -13 
    Q 738-601 T 137  [91m☒[0m 127 
    Q 905-949 T -44  [91m☒[0m 113 
    Q 143+254 T 397  [91m☒[0m 163 
    Q 332-34  T 298  [91m☒[0m -3  
    Q 66+829  T 895  [91m☒[0m 100 
    Q 66+634  T 700  [91m☒[0m 136 
    Q 32+941  T 973  [91m☒[0m 136 
    
    --------------------------------------------------
    Iteration 1
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 7s 203us/step - loss: 1.8651 - acc: 0.3278 - val_loss: 1.8271 - val_acc: 0.3330
    Q 836+477 T 1313 [91m☒[0m 116 
    Q 575-237 T 338  [91m☒[0m 21  
    Q 245+311 T 556  [91m☒[0m 111 
    Q 36+386  T 422  [91m☒[0m 116 
    Q 856-90  T 766  [91m☒[0m 21  
    Q 91-178  T -87  [91m☒[0m -21 
    Q 197+733 T 930  [91m☒[0m 116 
    Q 81+925  T 1006 [91m☒[0m 106 
    Q 785+465 T 1250 [91m☒[0m 1111
    Q 431-844 T -413 [91m☒[0m -11 
    
    --------------------------------------------------
    Iteration 2
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 10s 278us/step - loss: 1.7913 - acc: 0.3450 - val_loss: 1.7563 - val_acc: 0.3622
    Q 33-556  T -523 [91m☒[0m -660
    Q 979+723 T 1702 [91m☒[0m 1660
    Q 972-264 T 708  [91m☒[0m 600 
    Q 209-931 T -722 [91m☒[0m -610
    Q 59+905  T 964  [91m☒[0m 900 
    Q 550-591 T -41  [91m☒[0m -10 
    Q 49+128  T 177  [91m☒[0m 600 
    Q 746-96  T 650  [91m☒[0m 400 
    Q 36+47   T 83   [91m☒[0m 41  
    Q 30-242  T -212 [91m☒[0m -200
    
    --------------------------------------------------
    Iteration 3
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 11s 311us/step - loss: 1.7243 - acc: 0.3671 - val_loss: 1.7216 - val_acc: 0.3734
    Q 773+33  T 806  [91m☒[0m 643 
    Q 31+46   T 77   [91m☒[0m 12  
    Q 805+8   T 813  [91m☒[0m 100 
    Q 694+624 T 1318 [91m☒[0m 103 
    Q 920+283 T 1203 [91m☒[0m 104 
    Q 551-117 T 434  [91m☒[0m 201 
    Q 267-133 T 134  [91m☒[0m 11  
    Q 790-585 T 205  [91m☒[0m 213 
    Q 952+65  T 1017 [91m☒[0m 703 
    Q 377-76  T 301  [91m☒[0m 211 
    
    --------------------------------------------------
    Iteration 4
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 11s 307us/step - loss: 1.6504 - acc: 0.3896 - val_loss: 1.6093 - val_acc: 0.4007
    Q 922-631 T 291  [91m☒[0m 494 
    Q 494-48  T 446  [91m☒[0m 441 
    Q 44+878  T 922  [91m☒[0m 801 
    Q 355-684 T -329 [91m☒[0m -399
    Q 76-753  T -677 [91m☒[0m -572
    Q 214-330 T -116 [91m☒[0m -219
    Q 86-796  T -710 [91m☒[0m -698
    Q 648-99  T 549  [91m☒[0m 491 
    Q 666-45  T 621  [91m☒[0m 591 
    Q 392+64  T 456  [91m☒[0m 591 
    
    --------------------------------------------------
    Iteration 5
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 259us/step - loss: 1.5793 - acc: 0.4135 - val_loss: 1.5651 - val_acc: 0.4174
    Q 758-434 T 324  [91m☒[0m 300 
    Q 325+600 T 925  [91m☒[0m 806 
    Q 34-991  T -957 [91m☒[0m -860
    Q 58-687  T -629 [91m☒[0m -600
    Q 346-388 T -42  [91m☒[0m -108
    Q 800-59  T 741  [91m☒[0m 668 
    Q 124+465 T 589  [91m☒[0m 500 
    Q 66-333  T -267 [91m☒[0m -370
    Q 965+7   T 972  [91m☒[0m 866 
    Q 269+447 T 716  [91m☒[0m 506 
    
    --------------------------------------------------
    Iteration 6
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 1.5209 - acc: 0.4347 - val_loss: 1.5990 - val_acc: 0.4044
    Q 33-583  T -550 [91m☒[0m -722
    Q 411-430 T -19  [91m☒[0m -155
    Q 113+265 T 378  [91m☒[0m 505 
    Q 688+18  T 706  [91m☒[0m 755 
    Q 509+71  T 580  [91m☒[0m 755 
    Q 126+65  T 191  [91m☒[0m 205 
    Q 243-480 T -237 [91m☒[0m -325
    Q 686-545 T 141  [91m☒[0m 11  
    Q 13-557  T -544 [91m☒[0m -752
    Q 4+351   T 355  [91m☒[0m 555 
    
    --------------------------------------------------
    Iteration 7
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 1.4690 - acc: 0.4558 - val_loss: 1.4424 - val_acc: 0.4654
    Q 171-404 T -233 [91m☒[0m -110
    Q 855+127 T 982  [91m☒[0m 900 
    Q 403+8   T 411  [91m☒[0m 355 
    Q 92+733  T 825  [91m☒[0m 833 
    Q 44+621  T 665  [91m☒[0m 688 
    Q 798-16  T 782  [91m☒[0m 830 
    Q 744+35  T 779  [91m☒[0m 638 
    Q 159-417 T -258 [91m☒[0m -210
    Q 296+529 T 825  [91m☒[0m 700 
    Q 929+21  T 950  [91m☒[0m 990 
    
    --------------------------------------------------
    Iteration 8
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 1.4260 - acc: 0.4718 - val_loss: 1.4184 - val_acc: 0.4741
    Q 118+652 T 770  [91m☒[0m 766 
    Q 56-141  T -85  [91m☒[0m -11 
    Q 531-398 T 133  [91m☒[0m 141 
    Q 560-20  T 540  [91m☒[0m 566 
    Q 528-478 T 50   [91m☒[0m 11  
    Q 282+40  T 322  [91m☒[0m 364 
    Q 841+1   T 842  [91m☒[0m 811 
    Q 82+877  T 959  [91m☒[0m 941 
    Q 939-58  T 881  [91m☒[0m 866 
    Q 274+968 T 1242 [91m☒[0m 1266
    
    --------------------------------------------------
    Iteration 9
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 1.3883 - acc: 0.4869 - val_loss: 1.3829 - val_acc: 0.4863
    Q 583-329 T 254  [91m☒[0m 333 
    Q 872-415 T 457  [91m☒[0m 509 
    Q 4+84    T 88   [91m☒[0m 80  
    Q 33-825  T -792 [91m☒[0m -799
    Q 14-461  T -447 [91m☒[0m -419
    Q 546-71  T 475  [91m☒[0m 582 
    Q 4+84    T 88   [91m☒[0m 80  
    Q 71+274  T 345  [91m☒[0m 334 
    Q 20+846  T 866  [91m☒[0m 883 
    Q 876-114 T 762  [91m☒[0m 743 
    
    --------------------------------------------------
    Iteration 10
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 238us/step - loss: 1.3535 - acc: 0.5000 - val_loss: 1.3326 - val_acc: 0.5070
    Q 556+85  T 641  [91m☒[0m 625 
    Q 95+15   T 110  [91m☒[0m 100 
    Q 346+46  T 392  [91m☒[0m 382 
    Q 838+92  T 930  [91m☒[0m 905 
    Q 639-27  T 612  [91m☒[0m 588 
    Q 23-337  T -314 [91m☒[0m -290
    Q 415+41  T 456  [91m☒[0m 444 
    Q 620-199 T 421  [91m☒[0m 449 
    Q 235+25  T 260  [91m☒[0m 231 
    Q 733+44  T 777  [91m☒[0m 795 
    
    --------------------------------------------------
    Iteration 11
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 237us/step - loss: 1.3235 - acc: 0.5118 - val_loss: 1.3520 - val_acc: 0.4946
    Q 55+270  T 325  [91m☒[0m 300 
    Q 771-340 T 431  [91m☒[0m 400 
    Q 421-39  T 382  [91m☒[0m 391 
    Q 670+88  T 758  [91m☒[0m 751 
    Q 18+306  T 324  [91m☒[0m 301 
    Q 613-753 T -140 [91m☒[0m -109
    Q 225+30  T 255  [91m☒[0m 261 
    Q 897-673 T 224  [91m☒[0m 210 
    Q 654-274 T 380  [91m☒[0m 490 
    Q 213+93  T 306  [91m☒[0m 211 
    
    --------------------------------------------------
    Iteration 12
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 1.2961 - acc: 0.5212 - val_loss: 1.2913 - val_acc: 0.5209
    Q 718-38  T 680  [91m☒[0m 689 
    Q 970-446 T 524  [91m☒[0m 423 
    Q 504-405 T 99   [91m☒[0m 15  
    Q 35+50   T 85   [91m☒[0m 87  
    Q 474-88  T 386  [91m☒[0m 314 
    Q 284-119 T 165  [91m☒[0m 151 
    Q 50-873  T -823 [91m☒[0m -830
    Q 426+835 T 1261 [91m☒[0m 1187
    Q 817+46  T 863  [91m☒[0m 891 
    Q 538-609 T -71  [91m☒[0m -10 
    
    --------------------------------------------------
    Iteration 13
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 231us/step - loss: 1.2673 - acc: 0.5328 - val_loss: 1.2619 - val_acc: 0.5284
    Q 526-988 T -462 [91m☒[0m -511
    Q 480+772 T 1252 [91m☒[0m 1238
    Q 183-838 T -655 [91m☒[0m -668
    Q 550+5   T 555  [91m☒[0m 558 
    Q 61+99   T 160  [91m☒[0m 155 
    Q 254+298 T 552  [91m☒[0m 519 
    Q 636+834 T 1470 [91m☒[0m 1488
    Q 24-534  T -510 [91m☒[0m -508
    Q 236-85  T 151  [91m☒[0m 135 
    Q 821-25  T 796  [91m☒[0m 788 
    
    --------------------------------------------------
    Iteration 14
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 222us/step - loss: 1.2420 - acc: 0.5408 - val_loss: 1.2896 - val_acc: 0.5183
    Q 126-371 T -245 [91m☒[0m -219
    Q 78+582  T 660  [91m☒[0m 667 
    Q 77+208  T 285  [91m☒[0m 237 
    Q 73+89   T 162  [91m☒[0m 148 
    Q 2+802   T 804  [91m☒[0m 813 
    Q 672-28  T 644  [91m☒[0m 579 
    Q 970-634 T 336  [91m☒[0m 298 
    Q 267-28  T 239  [91m☒[0m 269 
    Q 828-258 T 570  [91m☒[0m 573 
    Q 607+739 T 1346 [91m☒[0m 1311
    
    --------------------------------------------------
    Iteration 15
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 220us/step - loss: 1.2147 - acc: 0.5498 - val_loss: 1.2380 - val_acc: 0.5383
    Q 226+36  T 262  [91m☒[0m 278 
    Q 217+702 T 919  [91m☒[0m 922 
    Q 474-88  T 386  [91m☒[0m 388 
    Q 82-623  T -541 [91m☒[0m -551
    Q 479+838 T 1317 [91m☒[0m 1366
    Q 549-762 T -213 [91m☒[0m -211
    Q 212-603 T -391 [91m☒[0m -418
    Q 93-731  T -638 [91m☒[0m -652
    Q 408-716 T -308 [91m☒[0m -335
    Q 596+42  T 638  [91m☒[0m 613 
    
    --------------------------------------------------
    Iteration 16
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 1.1890 - acc: 0.5602 - val_loss: 1.2196 - val_acc: 0.5444
    Q 891+50  T 941  [91m☒[0m 900 
    Q 4+464   T 468  [91m☒[0m 477 
    Q 42+752  T 794  [91m☒[0m 786 
    Q 372+77  T 449  [91m☒[0m 554 
    Q 882+843 T 1725 [91m☒[0m 1635
    Q 283-81  T 202  [91m☒[0m 217 
    Q 577+652 T 1229 [91m☒[0m 1200
    Q 429+69  T 498  [91m☒[0m 557 
    Q 32+906  T 938  [91m☒[0m 945 
    Q 9+939   T 948  [91m☒[0m 945 
    
    --------------------------------------------------
    Iteration 17
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 238us/step - loss: 1.1615 - acc: 0.5694 - val_loss: 1.1707 - val_acc: 0.5615
    Q 61+783  T 844  [91m☒[0m 849 
    Q 528-66  T 462  [91m☒[0m 413 
    Q 823+70  T 893  [91m☒[0m 888 
    Q 0+552   T 552  [91m☒[0m 557 
    Q 606-891 T -285 [92m☑[0m -285
    Q 309-507 T -198 [91m☒[0m -275
    Q 164-656 T -492 [91m☒[0m -400
    Q 103-966 T -863 [91m☒[0m -848
    Q 115+51  T 166  [91m☒[0m 178 
    Q 61-577  T -516 [91m☒[0m -532
    
    --------------------------------------------------
    Iteration 18
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 228us/step - loss: 1.1336 - acc: 0.5787 - val_loss: 1.1532 - val_acc: 0.5693
    Q 205-505 T -300 [91m☒[0m -334
    Q 389-819 T -430 [92m☑[0m -430
    Q 208+193 T 401  [91m☒[0m 414 
    Q 390-84  T 306  [91m☒[0m 314 
    Q 49-852  T -803 [91m☒[0m -804
    Q 734+33  T 767  [91m☒[0m 774 
    Q 267-133 T 134  [91m☒[0m 14  
    Q 513-725 T -212 [91m☒[0m -224
    Q 645-830 T -185 [91m☒[0m -284
    Q 906+37  T 943  [91m☒[0m 904 
    
    --------------------------------------------------
    Iteration 19
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 237us/step - loss: 1.1058 - acc: 0.5896 - val_loss: 1.1600 - val_acc: 0.5660
    Q 59-913  T -854 [91m☒[0m -888
    Q 714-722 T -8   [91m☒[0m -19 
    Q 29+439  T 468  [91m☒[0m 471 
    Q 803+147 T 950  [91m☒[0m 940 
    Q 174+16  T 190  [91m☒[0m 180 
    Q 545-383 T 162  [91m☒[0m 111 
    Q 905-800 T 105  [91m☒[0m 11  
    Q 825+342 T 1167 [91m☒[0m 1178
    Q 39-166  T -127 [91m☒[0m -130
    Q 252+472 T 724  [91m☒[0m 709 
    
    --------------------------------------------------
    Iteration 20
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 1.0767 - acc: 0.6020 - val_loss: 1.1139 - val_acc: 0.5848
    Q 825+342 T 1167 [91m☒[0m 1148
    Q 737-15  T 722  [91m☒[0m 788 
    Q 946-18  T 928  [91m☒[0m 988 
    Q 83+283  T 366  [91m☒[0m 361 
    Q 67+365  T 432  [91m☒[0m 431 
    Q 273+93  T 366  [91m☒[0m 361 
    Q 358-622 T -264 [91m☒[0m -288
    Q 86-664  T -578 [91m☒[0m -571
    Q 165+260 T 425  [91m☒[0m 428 
    Q 86+924  T 1010 [91m☒[0m 1018
    
    --------------------------------------------------
    Iteration 21
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 1.0486 - acc: 0.6115 - val_loss: 1.1324 - val_acc: 0.5791
    Q 508-997 T -489 [91m☒[0m -490
    Q 364-501 T -137 [91m☒[0m -155
    Q 775-899 T -124 [91m☒[0m -10 
    Q 768-611 T 157  [91m☒[0m 215 
    Q 223-765 T -542 [91m☒[0m -520
    Q 150-64  T 86   [91m☒[0m 60  
    Q 775-899 T -124 [91m☒[0m -10 
    Q 841-858 T -17  [91m☒[0m 10  
    Q 554-905 T -351 [91m☒[0m -365
    Q 526-394 T 132  [91m☒[0m 155 
    
    --------------------------------------------------
    Iteration 22
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 1.0203 - acc: 0.6223 - val_loss: 1.0405 - val_acc: 0.6101
    Q 379+365 T 744  [91m☒[0m 733 
    Q 35+165  T 200  [91m☒[0m 105 
    Q 75-592  T -517 [91m☒[0m -514
    Q 665-694 T -29  [91m☒[0m -11 
    Q 717+58  T 775  [91m☒[0m 771 
    Q 78-953  T -875 [91m☒[0m -877
    Q 150-273 T -123 [91m☒[0m -15 
    Q 134+162 T 296  [91m☒[0m 294 
    Q 681-318 T 363  [91m☒[0m 353 
    Q 297-486 T -189 [91m☒[0m -171
    
    --------------------------------------------------
    Iteration 23
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 224us/step - loss: 0.9939 - acc: 0.6312 - val_loss: 1.0218 - val_acc: 0.6137
    Q 638-433 T 205  [91m☒[0m 100 
    Q 63-430  T -367 [91m☒[0m -363
    Q 631-563 T 68   [91m☒[0m 10  
    Q 462+617 T 1079 [91m☒[0m 1060
    Q 89+32   T 121  [91m☒[0m 120 
    Q 599-629 T -30  [91m☒[0m -43 
    Q 404-925 T -521 [91m☒[0m -514
    Q 725-757 T -32  [91m☒[0m -35 
    Q 287-667 T -380 [91m☒[0m -393
    Q 548-29  T 519  [91m☒[0m 503 
    
    --------------------------------------------------
    Iteration 24
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 224us/step - loss: 0.9677 - acc: 0.6417 - val_loss: 0.9826 - val_acc: 0.6347
    Q 950+93  T 1043 [92m☑[0m 1043
    Q 724-821 T -97  [91m☒[0m -12 
    Q 57+240  T 297  [91m☒[0m 393 
    Q 95+758  T 853  [92m☑[0m 853 
    Q 919-453 T 466  [91m☒[0m 453 
    Q 875+16  T 891  [91m☒[0m 893 
    Q 759-968 T -209 [91m☒[0m -213
    Q 173-304 T -131 [91m☒[0m -143
    Q 61-795  T -734 [91m☒[0m -726
    Q 93+820  T 913  [91m☒[0m 903 
    
    --------------------------------------------------
    Iteration 25
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.9445 - acc: 0.6512 - val_loss: 0.9625 - val_acc: 0.6384
    Q 536+370 T 906  [91m☒[0m 806 
    Q 365-66  T 299  [91m☒[0m 291 
    Q 633+3   T 636  [91m☒[0m 646 
    Q 768+66  T 834  [91m☒[0m 830 
    Q 404-80  T 324  [91m☒[0m 321 
    Q 219-879 T -660 [91m☒[0m -668
    Q 61+99   T 160  [91m☒[0m 141 
    Q 729+969 T 1698 [91m☒[0m 1611
    Q 77+791  T 868  [91m☒[0m 863 
    Q 127+921 T 1048 [91m☒[0m 1062
    
    --------------------------------------------------
    Iteration 26
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 245us/step - loss: 0.9228 - acc: 0.6602 - val_loss: 1.0029 - val_acc: 0.6161
    Q 84-712  T -628 [91m☒[0m -633
    Q 599+529 T 1128 [91m☒[0m 1113
    Q 33+184  T 217  [91m☒[0m 209 
    Q 26-240  T -214 [91m☒[0m -210
    Q 238-677 T -439 [91m☒[0m -433
    Q 873-54  T 819  [91m☒[0m 835 
    Q 196+27  T 223  [91m☒[0m 214 
    Q 929-15  T 914  [91m☒[0m 999 
    Q 896+963 T 1859 [91m☒[0m 1867
    Q 81+925  T 1006 [91m☒[0m 1003
    
    --------------------------------------------------
    Iteration 27
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 10s 267us/step - loss: 0.9037 - acc: 0.6668 - val_loss: 0.9519 - val_acc: 0.6417
    Q 380+572 T 952  [91m☒[0m 960 
    Q 4+700   T 704  [91m☒[0m 716 
    Q 91+438  T 529  [92m☑[0m 529 
    Q 20+582  T 602  [91m☒[0m 598 
    Q 393+103 T 496  [91m☒[0m 491 
    Q 860-779 T 81   [91m☒[0m 71  
    Q 52-251  T -199 [91m☒[0m -200
    Q 51-143  T -92  [91m☒[0m -10 
    Q 802-468 T 334  [91m☒[0m 247 
    Q 84-712  T -628 [91m☒[0m -637
    
    --------------------------------------------------
    Iteration 28
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 0.8836 - acc: 0.6745 - val_loss: 0.9647 - val_acc: 0.6382
    Q 676-188 T 488  [91m☒[0m 593 
    Q 509+71  T 580  [91m☒[0m 581 
    Q 69-754  T -685 [91m☒[0m -689
    Q 851+213 T 1064 [91m☒[0m 1057
    Q 464-174 T 290  [91m☒[0m 273 
    Q 132+219 T 351  [92m☑[0m 351 
    Q 263+464 T 727  [91m☒[0m 729 
    Q 52+279  T 331  [91m☒[0m 321 
    Q 606-891 T -285 [91m☒[0m -271
    Q 166-125 T 41   [91m☒[0m 44  
    
    --------------------------------------------------
    Iteration 29
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 249us/step - loss: 0.8656 - acc: 0.6820 - val_loss: 1.0001 - val_acc: 0.6142
    Q 918+564 T 1482 [91m☒[0m 1377
    Q 869-21  T 848  [91m☒[0m 746 
    Q 175-893 T -718 [91m☒[0m -710
    Q 986-589 T 397  [91m☒[0m 490 
    Q 618+89  T 707  [91m☒[0m 606 
    Q 385-751 T -366 [91m☒[0m -370
    Q 421+78  T 499  [91m☒[0m 401 
    Q 757+8   T 765  [91m☒[0m 763 
    Q 760-83  T 677  [91m☒[0m 681 
    Q 124+90  T 214  [91m☒[0m 204 
    
    --------------------------------------------------
    Iteration 30
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 254us/step - loss: 0.8520 - acc: 0.6875 - val_loss: 0.9224 - val_acc: 0.6482
    Q 39+9    T 48   [91m☒[0m 59  
    Q 436+47  T 483  [91m☒[0m 581 
    Q 454+370 T 824  [91m☒[0m 827 
    Q 110-79  T 31   [91m☒[0m 32  
    Q 88-773  T -685 [91m☒[0m -687
    Q 21+42   T 63   [91m☒[0m 71  
    Q 105-425 T -320 [91m☒[0m -322
    Q 61-465  T -404 [91m☒[0m -497
    Q 494+89  T 583  [91m☒[0m 587 
    Q 762-849 T -87  [91m☒[0m -10 
    
    --------------------------------------------------
    Iteration 31
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 233us/step - loss: 0.8345 - acc: 0.6936 - val_loss: 0.8852 - val_acc: 0.6723
    Q 16+540  T 556  [91m☒[0m 559 
    Q 694+624 T 1318 [91m☒[0m 1313
    Q 82+56   T 138  [91m☒[0m 149 
    Q 771+213 T 984  [91m☒[0m 980 
    Q 463+11  T 474  [92m☑[0m 474 
    Q 281+50  T 331  [91m☒[0m 333 
    Q 81+925  T 1006 [91m☒[0m 1003
    Q 756-214 T 542  [91m☒[0m 534 
    Q 62-687  T -625 [91m☒[0m -623
    Q 741+58  T 799  [91m☒[0m 892 
    
    --------------------------------------------------
    Iteration 32
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 222us/step - loss: 0.8187 - acc: 0.6991 - val_loss: 0.8750 - val_acc: 0.6696
    Q 613+631 T 1244 [91m☒[0m 1242
    Q 647-917 T -270 [91m☒[0m -261
    Q 48+378  T 426  [91m☒[0m 424 
    Q 968+59  T 1027 [91m☒[0m 1032
    Q 71+274  T 345  [91m☒[0m 347 
    Q 312-923 T -611 [91m☒[0m -612
    Q 517+19  T 536  [91m☒[0m 532 
    Q 743+38  T 781  [91m☒[0m 782 
    Q 696-970 T -274 [91m☒[0m -281
    Q 677+10  T 687  [91m☒[0m 691 
    
    --------------------------------------------------
    Iteration 33
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 224us/step - loss: 0.8046 - acc: 0.7050 - val_loss: 0.8702 - val_acc: 0.6771
    Q 905+970 T 1875 [91m☒[0m 1806
    Q 931-248 T 683  [91m☒[0m 689 
    Q 4+608   T 612  [91m☒[0m 615 
    Q 386+64  T 450  [91m☒[0m 444 
    Q 716+220 T 936  [91m☒[0m 931 
    Q 174+731 T 905  [91m☒[0m 819 
    Q 606-126 T 480  [91m☒[0m 477 
    Q 99+771  T 870  [91m☒[0m 869 
    Q 310+96  T 406  [91m☒[0m 401 
    Q 212-603 T -391 [91m☒[0m -290
    
    --------------------------------------------------
    Iteration 34
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 238us/step - loss: 0.7899 - acc: 0.7119 - val_loss: 0.8514 - val_acc: 0.6839
    Q 589+575 T 1164 [92m☑[0m 1164
    Q 88-149  T -61  [91m☒[0m -78 
    Q 0+244   T 244  [91m☒[0m 243 
    Q 485+73  T 558  [91m☒[0m 557 
    Q 87-414  T -327 [91m☒[0m -324
    Q 318-567 T -249 [91m☒[0m -257
    Q 68-408  T -340 [91m☒[0m -333
    Q 546-76  T 470  [91m☒[0m 467 
    Q 777-564 T 213  [91m☒[0m 212 
    Q 102-798 T -696 [91m☒[0m -688
    
    --------------------------------------------------
    Iteration 35
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 230us/step - loss: 0.7746 - acc: 0.7178 - val_loss: 0.8349 - val_acc: 0.6810
    Q 411-430 T -19  [91m☒[0m -14 
    Q 707-107 T 600  [91m☒[0m 602 
    Q 68+88   T 156  [91m☒[0m 155 
    Q 51+542  T 593  [91m☒[0m 596 
    Q 77+43   T 120  [91m☒[0m 110 
    Q 108-947 T -839 [91m☒[0m -842
    Q 263+89  T 352  [91m☒[0m 351 
    Q 767-374 T 393  [91m☒[0m 406 
    Q 639-27  T 612  [91m☒[0m 616 
    Q 929-61  T 868  [92m☑[0m 868 
    
    --------------------------------------------------
    Iteration 36
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.7593 - acc: 0.7223 - val_loss: 0.8292 - val_acc: 0.6819
    Q 167-774 T -607 [91m☒[0m -606
    Q 522+510 T 1032 [91m☒[0m 1034
    Q 959-61  T 898  [91m☒[0m 895 
    Q 323-262 T 61   [91m☒[0m 56  
    Q 46-792  T -746 [91m☒[0m -744
    Q 946-451 T 495  [91m☒[0m 591 
    Q 261+576 T 837  [91m☒[0m 839 
    Q 2+651   T 653  [91m☒[0m 655 
    Q 29+222  T 251  [92m☑[0m 251 
    Q 773-236 T 537  [91m☒[0m 549 
    
    --------------------------------------------------
    Iteration 37
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 236us/step - loss: 0.7428 - acc: 0.7304 - val_loss: 0.8065 - val_acc: 0.6947
    Q 51-143  T -92  [91m☒[0m -10 
    Q 79+961  T 1040 [91m☒[0m 1041
    Q 74+624  T 698  [91m☒[0m 608 
    Q 518-147 T 371  [91m☒[0m 374 
    Q 938+643 T 1581 [92m☑[0m 1581
    Q 747-992 T -245 [91m☒[0m -243
    Q 709-947 T -238 [91m☒[0m -245
    Q 632+352 T 984  [91m☒[0m 987 
    Q 47+307  T 354  [91m☒[0m 363 
    Q 182+4   T 186  [91m☒[0m 189 
    
    --------------------------------------------------
    Iteration 38
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 244us/step - loss: 0.7265 - acc: 0.7350 - val_loss: 0.7977 - val_acc: 0.7015
    Q 1+358   T 359  [91m☒[0m 351 
    Q 405-414 T -9   [91m☒[0m -1  
    Q 854+18  T 872  [91m☒[0m 878 
    Q 370+58  T 428  [91m☒[0m 427 
    Q 89+899  T 988  [91m☒[0m 987 
    Q 517+40  T 557  [91m☒[0m 550 
    Q 112-168 T -56  [91m☒[0m -67 
    Q 799-416 T 383  [91m☒[0m 371 
    Q 1+226   T 227  [91m☒[0m 222 
    Q 668-624 T 44   [91m☒[0m 46  
    
    --------------------------------------------------
    Iteration 39
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 236us/step - loss: 0.7103 - acc: 0.7410 - val_loss: 0.7744 - val_acc: 0.7073
    Q 79+29   T 108  [91m☒[0m 106 
    Q 121-721 T -600 [91m☒[0m -602
    Q 14+30   T 44   [92m☑[0m 44  
    Q 472-388 T 84   [91m☒[0m 90  
    Q 418+47  T 465  [91m☒[0m 462 
    Q 233-481 T -248 [91m☒[0m -247
    Q 37+79   T 116  [91m☒[0m 115 
    Q 66+956  T 1022 [92m☑[0m 1022
    Q 679-787 T -108 [91m☒[0m -117
    Q 540+116 T 656  [91m☒[0m 650 
    
    --------------------------------------------------
    Iteration 40
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 233us/step - loss: 0.6916 - acc: 0.7467 - val_loss: 0.7785 - val_acc: 0.6991
    Q 513+30  T 543  [92m☑[0m 543 
    Q 459-82  T 377  [91m☒[0m 384 
    Q 557-52  T 505  [91m☒[0m 516 
    Q 706-625 T 81   [91m☒[0m 80  
    Q 389-176 T 213  [91m☒[0m 222 
    Q 379-88  T 291  [91m☒[0m 292 
    Q 699-945 T -246 [91m☒[0m -244
    Q 324-11  T 313  [91m☒[0m 304 
    Q 361+816 T 1177 [91m☒[0m 1174
    Q 486-677 T -191 [92m☑[0m -191
    
    --------------------------------------------------
    Iteration 41
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 236us/step - loss: 0.6726 - acc: 0.7541 - val_loss: 0.7493 - val_acc: 0.7158
    Q 198-318 T -120 [91m☒[0m -121
    Q 681-318 T 363  [91m☒[0m 361 
    Q 384+63  T 447  [91m☒[0m 446 
    Q 50+388  T 438  [92m☑[0m 438 
    Q 46-706  T -660 [91m☒[0m -665
    Q 86-551  T -465 [92m☑[0m -465
    Q 272-841 T -569 [91m☒[0m -576
    Q 644-53  T 591  [91m☒[0m 594 
    Q 121-288 T -167 [92m☑[0m -167
    Q 217+1   T 218  [91m☒[0m 228 
    
    --------------------------------------------------
    Iteration 42
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 241us/step - loss: 0.6519 - acc: 0.7617 - val_loss: 0.7492 - val_acc: 0.7146
    Q 269-885 T -616 [91m☒[0m -618
    Q 875+16  T 891  [91m☒[0m 895 
    Q 94+690  T 784  [91m☒[0m 775 
    Q 841+53  T 894  [91m☒[0m 896 
    Q 919+79  T 998  [91m☒[0m 990 
    Q 26+962  T 988  [91m☒[0m 998 
    Q 870+420 T 1290 [91m☒[0m 1280
    Q 803-386 T 417  [91m☒[0m 416 
    Q 709+259 T 968  [91m☒[0m 970 
    Q 511+37  T 548  [91m☒[0m 557 
    
    --------------------------------------------------
    Iteration 43
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 230us/step - loss: 0.6319 - acc: 0.7694 - val_loss: 0.7914 - val_acc: 0.6969
    Q 29+439  T 468  [91m☒[0m 467 
    Q 339-220 T 119  [91m☒[0m 116 
    Q 681-176 T 505  [91m☒[0m 405 
    Q 939-334 T 605  [92m☑[0m 605 
    Q 462+617 T 1079 [91m☒[0m 1066
    Q 58+336  T 394  [92m☑[0m 394 
    Q 664+3   T 667  [92m☑[0m 667 
    Q 886-964 T -78  [91m☒[0m -95 
    Q 518-147 T 371  [91m☒[0m 375 
    Q 33+184  T 217  [92m☑[0m 217 
    
    --------------------------------------------------
    Iteration 44
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.6091 - acc: 0.7780 - val_loss: 0.7339 - val_acc: 0.7184
    Q 885-129 T 756  [91m☒[0m 750 
    Q 44-664  T -620 [91m☒[0m -619
    Q 121-288 T -167 [91m☒[0m -168
    Q 478+76  T 554  [92m☑[0m 554 
    Q 996+623 T 1619 [91m☒[0m 1620
    Q 887-780 T 107  [91m☒[0m 108 
    Q 963-352 T 611  [91m☒[0m 610 
    Q 29-450  T -421 [91m☒[0m -424
    Q 88-767  T -679 [91m☒[0m -677
    Q 199-807 T -608 [91m☒[0m -612
    
    --------------------------------------------------
    Iteration 45
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 230us/step - loss: 0.5848 - acc: 0.7865 - val_loss: 0.6816 - val_acc: 0.7356
    Q 379-88  T 291  [91m☒[0m 281 
    Q 123+247 T 370  [92m☑[0m 370 
    Q 817-238 T 579  [91m☒[0m 580 
    Q 48-761  T -713 [91m☒[0m -714
    Q 220-73  T 147  [91m☒[0m 140 
    Q 1+223   T 224  [91m☒[0m 234 
    Q 523-301 T 222  [91m☒[0m 211 
    Q 121-104 T 17   [91m☒[0m 40  
    Q 443-580 T -137 [91m☒[0m -146
    Q 333-92  T 241  [91m☒[0m 243 
    
    --------------------------------------------------
    Iteration 46
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.5610 - acc: 0.7958 - val_loss: 0.6656 - val_acc: 0.7396
    Q 771+773 T 1544 [91m☒[0m 1543
    Q 55+884  T 939  [91m☒[0m 940 
    Q 602-856 T -254 [91m☒[0m -255
    Q 75+24   T 99   [91m☒[0m 90  
    Q 11-499  T -488 [92m☑[0m -488
    Q 675+867 T 1542 [91m☒[0m 1541
    Q 553-260 T 293  [92m☑[0m 293 
    Q 848-484 T 364  [91m☒[0m 365 
    Q 289+366 T 655  [92m☑[0m 655 
    Q 621-540 T 81   [91m☒[0m 80  
    
    --------------------------------------------------
    Iteration 47
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.5375 - acc: 0.8047 - val_loss: 0.6236 - val_acc: 0.7567
    Q 210-211 T -1   [91m☒[0m 91  
    Q 498+999 T 1497 [91m☒[0m 1588
    Q 850+649 T 1499 [91m☒[0m 1407
    Q 2+872   T 874  [92m☑[0m 874 
    Q 82+247  T 329  [91m☒[0m 339 
    Q 763-196 T 567  [91m☒[0m 566 
    Q 84+793  T 877  [92m☑[0m 877 
    Q 61-577  T -516 [92m☑[0m -516
    Q 693-273 T 420  [91m☒[0m 426 
    Q 897+128 T 1025 [91m☒[0m 1016
    
    --------------------------------------------------
    Iteration 48
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 260us/step - loss: 0.5091 - acc: 0.8149 - val_loss: 0.6103 - val_acc: 0.7549
    Q 956+247 T 1203 [91m☒[0m 1193
    Q 563-542 T 21   [91m☒[0m 19  
    Q 881-584 T 297  [91m☒[0m 202 
    Q 86+159  T 245  [92m☑[0m 245 
    Q 971-983 T -12  [91m☒[0m -2  
    Q 550-83  T 467  [91m☒[0m 469 
    Q 251-154 T 97   [91m☒[0m 19  
    Q 33-610  T -577 [92m☑[0m -577
    Q 441-267 T 174  [91m☒[0m 176 
    Q 682-25  T 657  [91m☒[0m 658 
    
    --------------------------------------------------
    Iteration 49
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 263us/step - loss: 0.4820 - acc: 0.8249 - val_loss: 0.5456 - val_acc: 0.7842
    Q 874-439 T 435  [91m☒[0m 444 
    Q 439-710 T -271 [92m☑[0m -271
    Q 892-594 T 298  [91m☒[0m 297 
    Q 358-622 T -264 [92m☑[0m -264
    Q 673+62  T 735  [92m☑[0m 735 
    Q 998-21  T 977  [91m☒[0m 979 
    Q 401-53  T 348  [91m☒[0m 358 
    Q 372+66  T 438  [92m☑[0m 438 
    Q 631+84  T 715  [91m☒[0m 716 
    Q 840-236 T 604  [91m☒[0m 606 
    
    --------------------------------------------------
    Iteration 50
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 246us/step - loss: 0.4548 - acc: 0.8362 - val_loss: 0.6578 - val_acc: 0.7664
    Q 71+55   T 126  [92m☑[0m 126 
    Q 545+607 T 1152 [92m☑[0m 1152
    Q 692-92  T 600  [91m☒[0m 591 
    Q 2+849   T 851  [92m☑[0m 851 
    Q 677-270 T 407  [92m☑[0m 407 
    Q 920+283 T 1203 [91m☒[0m 1202
    Q 283-81  T 202  [91m☒[0m 210 
    Q 570-766 T -196 [91m☒[0m -297
    Q 103-658 T -555 [91m☒[0m -546
    Q 618+89  T 707  [92m☑[0m 707 
    
    --------------------------------------------------
    Iteration 51
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 0.4264 - acc: 0.8479 - val_loss: 0.4887 - val_acc: 0.8051
    Q 7+78    T 85   [91m☒[0m 94  
    Q 17+696  T 713  [92m☑[0m 713 
    Q 447+1   T 448  [91m☒[0m 447 
    Q 746+13  T 759  [92m☑[0m 759 
    Q 443+856 T 1299 [91m☒[0m 1309
    Q 899-51  T 848  [91m☒[0m 842 
    Q 884-781 T 103  [92m☑[0m 103 
    Q 48-761  T -713 [92m☑[0m -713
    Q 41+377  T 418  [92m☑[0m 418 
    Q 79+29   T 108  [92m☑[0m 108 
    
    --------------------------------------------------
    Iteration 52
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 237us/step - loss: 0.4028 - acc: 0.8575 - val_loss: 0.4917 - val_acc: 0.8068
    Q 436+630 T 1066 [92m☑[0m 1066
    Q 74-265  T -191 [91m☒[0m -190
    Q 88-219  T -131 [92m☑[0m -131
    Q 60+709  T 769  [91m☒[0m 779 
    Q 411+51  T 462  [92m☑[0m 462 
    Q 146+89  T 235  [92m☑[0m 235 
    Q 500-298 T 202  [91m☒[0m 112 
    Q 224-79  T 145  [92m☑[0m 145 
    Q 350-949 T -599 [92m☑[0m -599
    Q 444-497 T -53  [91m☒[0m -55 
    
    --------------------------------------------------
    Iteration 53
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 233us/step - loss: 0.3786 - acc: 0.8681 - val_loss: 0.4920 - val_acc: 0.8062
    Q 198-80  T 118  [92m☑[0m 118 
    Q 87+740  T 827  [92m☑[0m 827 
    Q 814+10  T 824  [92m☑[0m 824 
    Q 78+882  T 960  [91m☒[0m 950 
    Q 798-503 T 295  [91m☒[0m 285 
    Q 341+185 T 526  [91m☒[0m 517 
    Q 205-505 T -300 [92m☑[0m -300
    Q 926+625 T 1551 [92m☑[0m 1551
    Q 91-552  T -461 [91m☒[0m -450
    Q 5+336   T 341  [92m☑[0m 341 
    
    --------------------------------------------------
    Iteration 54
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 227us/step - loss: 0.3539 - acc: 0.8797 - val_loss: 0.4313 - val_acc: 0.8318
    Q 195+277 T 472  [91m☒[0m 460 
    Q 919+96  T 1015 [91m☒[0m 1016
    Q 993-640 T 353  [91m☒[0m 343 
    Q 359+55  T 414  [92m☑[0m 414 
    Q 345-244 T 101  [91m☒[0m 100 
    Q 945+50  T 995  [91m☒[0m 996 
    Q 367-29  T 338  [91m☒[0m 339 
    Q 617+61  T 678  [91m☒[0m 689 
    Q 580+5   T 585  [91m☒[0m 586 
    Q 445-743 T -298 [92m☑[0m -298
    
    --------------------------------------------------
    Iteration 55
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.3320 - acc: 0.8883 - val_loss: 0.4147 - val_acc: 0.8360
    Q 298+832 T 1130 [91m☒[0m 1121
    Q 23+259  T 282  [92m☑[0m 282 
    Q 523+18  T 541  [91m☒[0m 542 
    Q 54-966  T -912 [92m☑[0m -912
    Q 768-471 T 297  [91m☒[0m 295 
    Q 978+428 T 1406 [91m☒[0m 1495
    Q 692-557 T 135  [92m☑[0m 135 
    Q 858+573 T 1431 [92m☑[0m 1431
    Q 876+603 T 1479 [91m☒[0m 1478
    Q 115+51  T 166  [91m☒[0m 165 
    
    --------------------------------------------------
    Iteration 56
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 0.3123 - acc: 0.8971 - val_loss: 0.4103 - val_acc: 0.8426
    Q 8+642   T 650  [92m☑[0m 650 
    Q 993-207 T 786  [91m☒[0m 686 
    Q 89+679  T 768  [92m☑[0m 768 
    Q 675+90  T 765  [92m☑[0m 765 
    Q 978-782 T 196  [91m☒[0m 185 
    Q 54-337  T -283 [91m☒[0m -284
    Q 347+5   T 352  [92m☑[0m 352 
    Q 821-25  T 796  [92m☑[0m 796 
    Q 25-782  T -757 [91m☒[0m -758
    Q 275-521 T -246 [92m☑[0m -246
    
    --------------------------------------------------
    Iteration 57
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 236us/step - loss: 0.2934 - acc: 0.9046 - val_loss: 0.3882 - val_acc: 0.8474
    Q 258-568 T -310 [91m☒[0m -300
    Q 891+309 T 1200 [91m☒[0m 1210
    Q 147-556 T -409 [92m☑[0m -409
    Q 24+762  T 786  [91m☒[0m 785 
    Q 606-891 T -285 [91m☒[0m -275
    Q 133-504 T -371 [92m☑[0m -371
    Q 75-936  T -861 [91m☒[0m -851
    Q 267+493 T 760  [91m☒[0m 750 
    Q 13+65   T 78   [92m☑[0m 78  
    Q 251-154 T 97   [91m☒[0m 98  
    
    --------------------------------------------------
    Iteration 58
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.2767 - acc: 0.9114 - val_loss: 0.3701 - val_acc: 0.8538
    Q 94-202  T -108 [91m☒[0m -118
    Q 606-891 T -285 [92m☑[0m -285
    Q 45+168  T 213  [92m☑[0m 213 
    Q 308+922 T 1230 [91m☒[0m 1220
    Q 44+475  T 519  [92m☑[0m 519 
    Q 62+863  T 925  [92m☑[0m 925 
    Q 77-143  T -66  [91m☒[0m -65 
    Q 71+55   T 126  [91m☒[0m 127 
    Q 550-83  T 467  [91m☒[0m 457 
    Q 905-800 T 105  [91m☒[0m 14  
    
    --------------------------------------------------
    Iteration 59
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 0.2614 - acc: 0.9180 - val_loss: 0.3881 - val_acc: 0.8608
    Q 648-99  T 549  [91m☒[0m 558 
    Q 764+3   T 767  [92m☑[0m 767 
    Q 897-673 T 224  [92m☑[0m 224 
    Q 156+60  T 216  [92m☑[0m 216 
    Q 174+16  T 190  [92m☑[0m 190 
    Q 373-484 T -111 [91m☒[0m -11 
    Q 226-808 T -582 [92m☑[0m -582
    Q 307+23  T 330  [92m☑[0m 330 
    Q 78+51   T 129  [92m☑[0m 129 
    Q 59-913  T -854 [91m☒[0m -844
    
    --------------------------------------------------
    Iteration 60
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 224us/step - loss: 0.2450 - acc: 0.9260 - val_loss: 0.4888 - val_acc: 0.8337
    Q 760-63  T 697  [92m☑[0m 697 
    Q 568-888 T -320 [91m☒[0m -310
    Q 44+586  T 630  [92m☑[0m 630 
    Q 2+92    T 94   [91m☒[0m 93  
    Q 645-58  T 587  [92m☑[0m 587 
    Q 620+86  T 706  [91m☒[0m 606 
    Q 223-186 T 37   [91m☒[0m 15  
    Q 799-330 T 469  [92m☑[0m 469 
    Q 46+969  T 1015 [92m☑[0m 1015
    Q 115+51  T 166  [92m☑[0m 166 
    
    --------------------------------------------------
    Iteration 61
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 0.2317 - acc: 0.9317 - val_loss: 0.3360 - val_acc: 0.8721
    Q 513+19  T 532  [92m☑[0m 532 
    Q 966+993 T 1959 [91m☒[0m 1960
    Q 411+51  T 462  [92m☑[0m 462 
    Q 472-322 T 150  [91m☒[0m 140 
    Q 61-465  T -404 [91m☒[0m -304
    Q 689+42  T 731  [92m☑[0m 731 
    Q 647-458 T 189  [92m☑[0m 189 
    Q 54-501  T -447 [91m☒[0m -457
    Q 69-798  T -729 [91m☒[0m -720
    Q 899-98  T 801  [91m☒[0m 790 
    
    --------------------------------------------------
    Iteration 62
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 227us/step - loss: 0.2188 - acc: 0.9357 - val_loss: 0.3234 - val_acc: 0.8770
    Q 803+348 T 1151 [91m☒[0m 1141
    Q 49+128  T 177  [92m☑[0m 177 
    Q 41+334  T 375  [92m☑[0m 375 
    Q 778-36  T 742  [92m☑[0m 742 
    Q 127+14  T 141  [91m☒[0m 131 
    Q 114-466 T -352 [91m☒[0m -253
    Q 693+629 T 1322 [91m☒[0m 1321
    Q 374-55  T 319  [92m☑[0m 319 
    Q 406+851 T 1257 [92m☑[0m 1257
    Q 445-939 T -494 [92m☑[0m -494
    
    --------------------------------------------------
    Iteration 63
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 220us/step - loss: 0.2067 - acc: 0.9395 - val_loss: 0.3943 - val_acc: 0.8686
    Q 273+859 T 1132 [91m☒[0m 1131
    Q 258+706 T 964  [92m☑[0m 964 
    Q 50+265  T 315  [92m☑[0m 315 
    Q 899-236 T 663  [92m☑[0m 663 
    Q 23-252  T -229 [92m☑[0m -229
    Q 993-146 T 847  [91m☒[0m 848 
    Q 203-842 T -639 [91m☒[0m -640
    Q 0+535   T 535  [92m☑[0m 535 
    Q 585+363 T 948  [91m☒[0m 959 
    Q 22+769  T 791  [92m☑[0m 791 
    
    --------------------------------------------------
    Iteration 64
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 224us/step - loss: 0.1969 - acc: 0.9435 - val_loss: 0.3389 - val_acc: 0.8726
    Q 31+364  T 395  [91m☒[0m 396 
    Q 235+25  T 260  [92m☑[0m 260 
    Q 237-254 T -17  [91m☒[0m -27 
    Q 81+565  T 646  [92m☑[0m 646 
    Q 588-950 T -362 [92m☑[0m -362
    Q 56-113  T -57  [91m☒[0m -48 
    Q 788-745 T 43   [91m☒[0m 44  
    Q 583+65  T 648  [92m☑[0m 648 
    Q 378-71  T 307  [92m☑[0m 307 
    Q 216-357 T -141 [91m☒[0m -131
    
    --------------------------------------------------
    Iteration 65
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 232us/step - loss: 0.1850 - acc: 0.9473 - val_loss: 0.2731 - val_acc: 0.9012
    Q 4+745   T 749  [92m☑[0m 749 
    Q 515+878 T 1393 [92m☑[0m 1393
    Q 365+8   T 373  [92m☑[0m 373 
    Q 559+784 T 1343 [92m☑[0m 1343
    Q 371-188 T 183  [92m☑[0m 183 
    Q 127+921 T 1048 [92m☑[0m 1048
    Q 2+255   T 257  [92m☑[0m 257 
    Q 981-17  T 964  [92m☑[0m 964 
    Q 153+32  T 185  [92m☑[0m 185 
    Q 191-465 T -274 [92m☑[0m -274
    
    --------------------------------------------------
    Iteration 66
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 232us/step - loss: 0.1755 - acc: 0.9513 - val_loss: 0.2797 - val_acc: 0.9025
    Q 606-539 T 67   [91m☒[0m 77  
    Q 661+363 T 1024 [92m☑[0m 1024
    Q 195+277 T 472  [91m☒[0m 462 
    Q 266-722 T -456 [92m☑[0m -456
    Q 91+438  T 529  [92m☑[0m 529 
    Q 76+970  T 1046 [91m☒[0m 1056
    Q 164-656 T -492 [92m☑[0m -492
    Q 451-27  T 424  [92m☑[0m 424 
    Q 928-741 T 187  [92m☑[0m 187 
    Q 389-87  T 302  [92m☑[0m 302 
    
    --------------------------------------------------
    Iteration 67
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 234us/step - loss: 0.1659 - acc: 0.9545 - val_loss: 0.2657 - val_acc: 0.9052
    Q 971+47  T 1018 [92m☑[0m 1018
    Q 760+58  T 818  [92m☑[0m 818 
    Q 839+115 T 954  [91m☒[0m 955 
    Q 492+51  T 543  [92m☑[0m 543 
    Q 392-699 T -307 [92m☑[0m -307
    Q 633+3   T 636  [92m☑[0m 636 
    Q 32-932  T -900 [92m☑[0m -900
    Q 269-34  T 235  [92m☑[0m 235 
    Q 749-634 T 115  [92m☑[0m 115 
    Q 828-835 T -7   [91m☒[0m -   
    
    --------------------------------------------------
    Iteration 68
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.1593 - acc: 0.9566 - val_loss: 0.2321 - val_acc: 0.9184
    Q 88-753  T -665 [92m☑[0m -665
    Q 54+372  T 426  [92m☑[0m 426 
    Q 358-622 T -264 [92m☑[0m -264
    Q 88-149  T -61  [91m☒[0m -71 
    Q 657+720 T 1377 [92m☑[0m 1377
    Q 24-964  T -940 [92m☑[0m -940
    Q 930+62  T 992  [92m☑[0m 992 
    Q 711-53  T 658  [91m☒[0m 668 
    Q 75+143  T 218  [92m☑[0m 218 
    Q 13+897  T 910  [92m☑[0m 910 
    
    --------------------------------------------------
    Iteration 69
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 221us/step - loss: 0.1502 - acc: 0.9597 - val_loss: 0.2230 - val_acc: 0.9204
    Q 54+372  T 426  [92m☑[0m 426 
    Q 882+878 T 1760 [91m☒[0m 1751
    Q 440-637 T -197 [92m☑[0m -197
    Q 315+37  T 352  [92m☑[0m 352 
    Q 28-627  T -599 [91m☒[0m -508
    Q 259+707 T 966  [91m☒[0m 965 
    Q 52+39   T 91   [91m☒[0m 902 
    Q 852+387 T 1239 [92m☑[0m 1239
    Q 812+195 T 1007 [92m☑[0m 1007
    Q 281+18  T 299  [92m☑[0m 299 
    
    --------------------------------------------------
    Iteration 70
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 216us/step - loss: 0.1427 - acc: 0.9623 - val_loss: 0.2588 - val_acc: 0.9109
    Q 716-28  T 688  [92m☑[0m 688 
    Q 577-250 T 327  [92m☑[0m 327 
    Q 243-720 T -477 [92m☑[0m -477
    Q 124-36  T 88   [92m☑[0m 88  
    Q 777-564 T 213  [92m☑[0m 213 
    Q 388+64  T 452  [92m☑[0m 452 
    Q 858+801 T 1659 [91m☒[0m 1668
    Q 313+767 T 1080 [92m☑[0m 1080
    Q 33-648  T -615 [92m☑[0m -615
    Q 407+713 T 1120 [92m☑[0m 1120
    
    --------------------------------------------------
    Iteration 71
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 215us/step - loss: 0.1356 - acc: 0.9637 - val_loss: 0.2447 - val_acc: 0.9089
    Q 99+916  T 1015 [92m☑[0m 1015
    Q 82+748  T 830  [92m☑[0m 830 
    Q 284-119 T 165  [92m☑[0m 165 
    Q 52-923  T -871 [91m☒[0m -872
    Q 46-792  T -746 [92m☑[0m -746
    Q 911+30  T 941  [92m☑[0m 941 
    Q 56+60   T 116  [92m☑[0m 116 
    Q 7+502   T 509  [91m☒[0m 510 
    Q 509+50  T 559  [91m☒[0m 569 
    Q 263+464 T 727  [92m☑[0m 727 
    
    --------------------------------------------------
    Iteration 72
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 223us/step - loss: 0.1294 - acc: 0.9653 - val_loss: 0.2909 - val_acc: 0.8945
    Q 384-250 T 134  [92m☑[0m 134 
    Q 297-486 T -189 [92m☑[0m -189
    Q 755-312 T 443  [92m☑[0m 443 
    Q 537-472 T 65   [92m☑[0m 65  
    Q 963+50  T 1013 [91m☒[0m 1023
    Q 805+8   T 813  [91m☒[0m 812 
    Q 258-69  T 189  [92m☑[0m 189 
    Q 979-175 T 804  [92m☑[0m 804 
    Q 918+564 T 1482 [91m☒[0m 1382
    Q 476-321 T 155  [92m☑[0m 155 
    
    --------------------------------------------------
    Iteration 73
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 257us/step - loss: 0.1260 - acc: 0.9675 - val_loss: 0.2111 - val_acc: 0.9266
    Q 23+925  T 948  [92m☑[0m 948 
    Q 406+5   T 411  [92m☑[0m 411 
    Q 795-34  T 761  [91m☒[0m 760 
    Q 593+425 T 1018 [92m☑[0m 1018
    Q 869-610 T 259  [92m☑[0m 259 
    Q 120-494 T -374 [92m☑[0m -374
    Q 537-987 T -450 [92m☑[0m -450
    Q 78+882  T 960  [92m☑[0m 960 
    Q 2+255   T 257  [91m☒[0m 258 
    Q 694-46  T 648  [91m☒[0m 649 
    
    --------------------------------------------------
    Iteration 74
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 256us/step - loss: 0.1181 - acc: 0.9691 - val_loss: 0.2201 - val_acc: 0.9212
    Q 83+885  T 968  [92m☑[0m 968 
    Q 371-908 T -537 [92m☑[0m -537
    Q 658-927 T -269 [92m☑[0m -269
    Q 170+897 T 1067 [92m☑[0m 1067
    Q 108-383 T -275 [91m☒[0m -274
    Q 307-658 T -351 [92m☑[0m -351
    Q 22+603  T 625  [92m☑[0m 625 
    Q 384+199 T 583  [92m☑[0m 583 
    Q 196-904 T -708 [92m☑[0m -708
    Q 458-21  T 437  [92m☑[0m 437 
    
    --------------------------------------------------
    Iteration 75
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 248us/step - loss: 0.1128 - acc: 0.9713 - val_loss: 0.1982 - val_acc: 0.9267
    Q 88-149  T -61  [91m☒[0m -71 
    Q 803+147 T 950  [91m☒[0m 949 
    Q 288+739 T 1027 [92m☑[0m 1027
    Q 49+859  T 908  [91m☒[0m 918 
    Q 172+917 T 1089 [92m☑[0m 1089
    Q 171-599 T -428 [92m☑[0m -428
    Q 890-935 T -45  [92m☑[0m -45 
    Q 96+153  T 249  [91m☒[0m 259 
    Q 78+51   T 129  [92m☑[0m 129 
    Q 631+84  T 715  [92m☑[0m 715 
    
    --------------------------------------------------
    Iteration 76
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 228us/step - loss: 0.1093 - acc: 0.9717 - val_loss: 0.2087 - val_acc: 0.9209
    Q 17-230  T -213 [92m☑[0m -213
    Q 217+1   T 218  [92m☑[0m 218 
    Q 108-372 T -264 [92m☑[0m -264
    Q 660-41  T 619  [91m☒[0m 629 
    Q 608-946 T -338 [92m☑[0m -338
    Q 69-754  T -685 [92m☑[0m -685
    Q 243-65  T 178  [92m☑[0m 178 
    Q 609-62  T 547  [92m☑[0m 547 
    Q 842-978 T -136 [92m☑[0m -136
    Q 23+43   T 66   [91m☒[0m 67  
    
    --------------------------------------------------
    Iteration 77
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 229us/step - loss: 0.1027 - acc: 0.9736 - val_loss: 0.2829 - val_acc: 0.9006
    Q 237-77  T 160  [91m☒[0m 150 
    Q 156+60  T 216  [91m☒[0m 215 
    Q 108-758 T -650 [91m☒[0m -640
    Q 699-50  T 649  [92m☑[0m 649 
    Q 653-608 T 45   [92m☑[0m 45  
    Q 63+62   T 125  [92m☑[0m 125 
    Q 304-568 T -264 [92m☑[0m -264
    Q 478+86  T 564  [92m☑[0m 564 
    Q 57+316  T 373  [92m☑[0m 373 
    Q 358+104 T 462  [92m☑[0m 462 
    
    --------------------------------------------------
    Iteration 78
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 227us/step - loss: 0.0997 - acc: 0.9747 - val_loss: 0.2034 - val_acc: 0.9276
    Q 95+15   T 110  [92m☑[0m 110 
    Q 373-101 T 272  [91m☒[0m 271 
    Q 81-101  T -20  [91m☒[0m -19 
    Q 973+895 T 1868 [92m☑[0m 1868
    Q 3+179   T 182  [92m☑[0m 182 
    Q 910+28  T 938  [91m☒[0m 948 
    Q 40+174  T 214  [92m☑[0m 214 
    Q 76-753  T -677 [92m☑[0m -677
    Q 882-50  T 832  [92m☑[0m 832 
    Q 882+843 T 1725 [92m☑[0m 1725
    
    --------------------------------------------------
    Iteration 79
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 233us/step - loss: 0.0952 - acc: 0.9766 - val_loss: 0.1986 - val_acc: 0.9274
    Q 410+9   T 419  [92m☑[0m 419 
    Q 882-50  T 832  [91m☒[0m 822 
    Q 334+940 T 1274 [92m☑[0m 1274
    Q 89+414  T 503  [92m☑[0m 503 
    Q 652+976 T 1628 [92m☑[0m 1628
    Q 0+526   T 526  [91m☒[0m 516 
    Q 185+632 T 817  [92m☑[0m 817 
    Q 315+749 T 1064 [91m☒[0m 1054
    Q 546-71  T 475  [92m☑[0m 475 
    Q 463-476 T -13  [91m☒[0m -23 
    
    --------------------------------------------------
    Iteration 80
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 233us/step - loss: 0.0934 - acc: 0.9762 - val_loss: 0.1758 - val_acc: 0.9405
    Q 84-176  T -92  [92m☑[0m -92 
    Q 442+27  T 469  [92m☑[0m 469 
    Q 9+363   T 372  [92m☑[0m 372 
    Q 652+90  T 742  [92m☑[0m 742 
    Q 557-145 T 412  [92m☑[0m 412 
    Q 224-36  T 188  [92m☑[0m 188 
    Q 405-414 T -9   [91m☒[0m -98 
    Q 0+760   T 760  [91m☒[0m 770 
    Q 943+5   T 948  [92m☑[0m 948 
    Q 714+769 T 1483 [92m☑[0m 1483
    
    --------------------------------------------------
    Iteration 81
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 243us/step - loss: 0.0876 - acc: 0.9780 - val_loss: 0.1799 - val_acc: 0.9363
    Q 929+57  T 986  [92m☑[0m 986 
    Q 883-935 T -52  [92m☑[0m -52 
    Q 950+93  T 1043 [92m☑[0m 1043
    Q 697-499 T 198  [91m☒[0m 298 
    Q 513-768 T -255 [92m☑[0m -255
    Q 83-684  T -601 [91m☒[0m -501
    Q 39+67   T 106  [92m☑[0m 106 
    Q 584-52  T 532  [91m☒[0m 542 
    Q 885+20  T 905  [92m☑[0m 905 
    Q 54+359  T 413  [92m☑[0m 413 
    
    --------------------------------------------------
    Iteration 82
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 231us/step - loss: 0.0848 - acc: 0.9786 - val_loss: 0.2359 - val_acc: 0.9179
    Q 787-474 T 313  [92m☑[0m 313 
    Q 9+805   T 814  [91m☒[0m 813 
    Q 44-432  T -388 [92m☑[0m -388
    Q 88-902  T -814 [91m☒[0m -824
    Q 850+649 T 1499 [91m☒[0m 1409
    Q 30+369  T 399  [92m☑[0m 399 
    Q 74-433  T -359 [92m☑[0m -359
    Q 355-467 T -112 [92m☑[0m -112
    Q 236+10  T 246  [92m☑[0m 246 
    Q 462+617 T 1079 [91m☒[0m 1080
    
    --------------------------------------------------
    Iteration 83
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 228us/step - loss: 0.0821 - acc: 0.9800 - val_loss: 0.2130 - val_acc: 0.9252
    Q 657-674 T -17  [91m☒[0m -27 
    Q 16+477  T 493  [92m☑[0m 493 
    Q 66-437  T -371 [92m☑[0m -371
    Q 21+429  T 450  [91m☒[0m 440 
    Q 607-336 T 271  [92m☑[0m 271 
    Q 75+89   T 164  [92m☑[0m 164 
    Q 571-843 T -272 [92m☑[0m -272
    Q 80-589  T -509 [92m☑[0m -509
    Q 86-796  T -710 [91m☒[0m -700
    Q 918+564 T 1482 [92m☑[0m 1482
    
    --------------------------------------------------
    Iteration 84
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.0791 - acc: 0.9806 - val_loss: 0.1844 - val_acc: 0.9339
    Q 999-553 T 446  [92m☑[0m 446 
    Q 824-168 T 656  [92m☑[0m 656 
    Q 873+62  T 935  [92m☑[0m 935 
    Q 775-73  T 702  [92m☑[0m 702 
    Q 147-859 T -712 [92m☑[0m -712
    Q 70-964  T -894 [92m☑[0m -894
    Q 874+91  T 965  [92m☑[0m 965 
    Q 940+479 T 1419 [92m☑[0m 1419
    Q 756-92  T 664  [92m☑[0m 664 
    Q 20+582  T 602  [92m☑[0m 602 
    
    --------------------------------------------------
    Iteration 85
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.0753 - acc: 0.9817 - val_loss: 0.1632 - val_acc: 0.9411
    Q 88+163  T 251  [92m☑[0m 251 
    Q 366+946 T 1312 [92m☑[0m 1312
    Q 568-888 T -320 [91m☒[0m -310
    Q 851+255 T 1106 [92m☑[0m 1106
    Q 482-784 T -302 [91m☒[0m -303
    Q 24-940  T -916 [92m☑[0m -916
    Q 48-138  T -90  [91m☒[0m -99 
    Q 607+872 T 1479 [91m☒[0m 1478
    Q 186-118 T 68   [92m☑[0m 68  
    Q 90-210  T -120 [91m☒[0m -111
    
    --------------------------------------------------
    Iteration 86
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 226us/step - loss: 0.0737 - acc: 0.9820 - val_loss: 0.2149 - val_acc: 0.9286
    Q 88+605  T 693  [92m☑[0m 693 
    Q 386+748 T 1134 [92m☑[0m 1134
    Q 12+523  T 535  [92m☑[0m 535 
    Q 675+90  T 765  [92m☑[0m 765 
    Q 165-55  T 110  [91m☒[0m 100 
    Q 836+95  T 931  [92m☑[0m 931 
    Q 31+82   T 113  [92m☑[0m 113 
    Q 261+576 T 837  [92m☑[0m 837 
    Q 665-602 T 63   [91m☒[0m 53  
    Q 57+240  T 297  [92m☑[0m 297 
    
    --------------------------------------------------
    Iteration 87
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 225us/step - loss: 0.0709 - acc: 0.9826 - val_loss: 0.1653 - val_acc: 0.9434
    Q 13+841  T 854  [91m☒[0m 864 
    Q 707-843 T -136 [91m☒[0m -135
    Q 725-757 T -32  [92m☑[0m -32 
    Q 268-499 T -231 [92m☑[0m -231
    Q 382-826 T -444 [92m☑[0m -444
    Q 503-375 T 128  [92m☑[0m 128 
    Q 14+928  T 942  [91m☒[0m 952 
    Q 139+25  T 164  [92m☑[0m 164 
    Q 874+91  T 965  [92m☑[0m 965 
    Q 559-363 T 196  [92m☑[0m 196 
    
    --------------------------------------------------
    Iteration 88
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 231us/step - loss: 0.0693 - acc: 0.9837 - val_loss: 0.1779 - val_acc: 0.9392
    Q 798+653 T 1451 [92m☑[0m 1451
    Q 326-20  T 306  [92m☑[0m 306 
    Q 651-86  T 565  [92m☑[0m 565 
    Q 778+519 T 1297 [91m☒[0m 1287
    Q 327-402 T -75  [92m☑[0m -75 
    Q 588+172 T 760  [92m☑[0m 760 
    Q 843-577 T 266  [92m☑[0m 266 
    Q 863-675 T 188  [92m☑[0m 188 
    Q 707-766 T -59  [91m☒[0m -69 
    Q 545-962 T -417 [92m☑[0m -417
    
    --------------------------------------------------
    Iteration 89
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 257us/step - loss: 0.0657 - acc: 0.9842 - val_loss: 0.1507 - val_acc: 0.9456
    Q 929-15  T 914  [92m☑[0m 914 
    Q 205+151 T 356  [92m☑[0m 356 
    Q 73+89   T 162  [91m☒[0m 161 
    Q 54+372  T 426  [92m☑[0m 426 
    Q 563-542 T 21   [91m☒[0m 10  
    Q 763+582 T 1345 [92m☑[0m 1345
    Q 158-769 T -611 [92m☑[0m -611
    Q 521+84  T 605  [92m☑[0m 605 
    Q 116-627 T -511 [92m☑[0m -511
    Q 105+92  T 197  [91m☒[0m 297 
    
    --------------------------------------------------
    Iteration 90
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 231us/step - loss: 0.0643 - acc: 0.9849 - val_loss: 0.1628 - val_acc: 0.9430
    Q 999-553 T 446  [92m☑[0m 446 
    Q 200-514 T -314 [91m☒[0m -414
    Q 466+295 T 761  [91m☒[0m 750 
    Q 653-61  T 592  [92m☑[0m 592 
    Q 7+502   T 509  [91m☒[0m 510 
    Q 826-711 T 115  [92m☑[0m 115 
    Q 83+483  T 566  [92m☑[0m 566 
    Q 790-585 T 205  [92m☑[0m 205 
    Q 69-798  T -729 [91m☒[0m -728
    Q 24+11   T 35   [92m☑[0m 35  
    
    --------------------------------------------------
    Iteration 91
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 234us/step - loss: 0.0624 - acc: 0.9856 - val_loss: 0.1528 - val_acc: 0.9472
    Q 584+43  T 627  [92m☑[0m 627 
    Q 876-114 T 762  [92m☑[0m 762 
    Q 560-20  T 540  [91m☒[0m 530 
    Q 289+366 T 655  [92m☑[0m 655 
    Q 203+116 T 319  [91m☒[0m 329 
    Q 1+201   T 202  [92m☑[0m 202 
    Q 976+126 T 1102 [92m☑[0m 1102
    Q 94-314  T -220 [92m☑[0m -220
    Q 505+89  T 594  [92m☑[0m 594 
    Q 53+142  T 195  [92m☑[0m 195 
    
    --------------------------------------------------
    Iteration 92
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 262us/step - loss: 0.0596 - acc: 0.9861 - val_loss: 0.1365 - val_acc: 0.9519
    Q 638-765 T -127 [91m☒[0m -117
    Q 580-50  T 530  [91m☒[0m 630 
    Q 87-656  T -569 [92m☑[0m -569
    Q 825+342 T 1167 [92m☑[0m 1167
    Q 651-318 T 333  [92m☑[0m 333 
    Q 30+351  T 381  [92m☑[0m 381 
    Q 238+55  T 293  [92m☑[0m 293 
    Q 110-79  T 31   [92m☑[0m 31  
    Q 4+745   T 749  [92m☑[0m 749 
    Q 83+409  T 492  [92m☑[0m 492 
    
    --------------------------------------------------
    Iteration 93
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 259us/step - loss: 0.0583 - acc: 0.9863 - val_loss: 0.1651 - val_acc: 0.9415
    Q 873-54  T 819  [91m☒[0m 829 
    Q 47+345  T 392  [91m☒[0m 391 
    Q 15+78   T 93   [91m☒[0m 92  
    Q 489+79  T 568  [92m☑[0m 568 
    Q 591-349 T 242  [92m☑[0m 242 
    Q 800+92  T 892  [91m☒[0m 893 
    Q 83+283  T 366  [92m☑[0m 366 
    Q 191-391 T -200 [91m☒[0m -100
    Q 165+5   T 170  [91m☒[0m 160 
    Q 904-30  T 874  [91m☒[0m 864 
    
    --------------------------------------------------
    Iteration 94
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 245us/step - loss: 0.0574 - acc: 0.9865 - val_loss: 0.1635 - val_acc: 0.9430
    Q 693-273 T 420  [92m☑[0m 420 
    Q 3+437   T 440  [92m☑[0m 440 
    Q 36+352  T 388  [92m☑[0m 388 
    Q 78+882  T 960  [92m☑[0m 960 
    Q 15-314  T -299 [91m☒[0m -399
    Q 433+205 T 638  [92m☑[0m 638 
    Q 577-16  T 561  [91m☒[0m 560 
    Q 462+617 T 1079 [91m☒[0m 1089
    Q 24+11   T 35   [92m☑[0m 35  
    Q 334+940 T 1274 [92m☑[0m 1274
    
    --------------------------------------------------
    Iteration 95
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 10s 272us/step - loss: 0.0559 - acc: 0.9872 - val_loss: 0.1678 - val_acc: 0.9409
    Q 121-721 T -600 [91m☒[0m -590
    Q 35+886  T 921  [92m☑[0m 921 
    Q 310-701 T -391 [92m☑[0m -391
    Q 269-538 T -269 [92m☑[0m -269
    Q 316+897 T 1213 [92m☑[0m 1213
    Q 492-308 T 184  [92m☑[0m 184 
    Q 269-538 T -269 [92m☑[0m -269
    Q 158-14  T 144  [92m☑[0m 144 
    Q 789-46  T 743  [91m☒[0m 753 
    Q 40+471  T 511  [92m☑[0m 511 
    
    --------------------------------------------------
    Iteration 96
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 9s 255us/step - loss: 0.0525 - acc: 0.9882 - val_loss: 0.1941 - val_acc: 0.9364
    Q 99+420  T 519  [92m☑[0m 519 
    Q 54-350  T -296 [91m☒[0m -286
    Q 448-623 T -175 [92m☑[0m -175
    Q 71-542  T -471 [91m☒[0m -460
    Q 226+36  T 262  [92m☑[0m 262 
    Q 555+978 T 1533 [92m☑[0m 1533
    Q 620+86  T 706  [92m☑[0m 706 
    Q 160+934 T 1094 [92m☑[0m 1094
    Q 88+83   T 171  [92m☑[0m 171 
    Q 28+734  T 762  [92m☑[0m 762 
    
    --------------------------------------------------
    Iteration 97
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 232us/step - loss: 0.0542 - acc: 0.9874 - val_loss: 0.1339 - val_acc: 0.9537
    Q 233-481 T -248 [92m☑[0m -248
    Q 931-623 T 308  [92m☑[0m 308 
    Q 54+359  T 413  [92m☑[0m 413 
    Q 842+80  T 922  [92m☑[0m 922 
    Q 762+638 T 1400 [91m☒[0m 1300
    Q 26-137  T -111 [92m☑[0m -111
    Q 708+808 T 1516 [91m☒[0m 1518
    Q 562+42  T 604  [92m☑[0m 604 
    Q 6+160   T 166  [92m☑[0m 166 
    Q 443+856 T 1299 [91m☒[0m 1399
    
    --------------------------------------------------
    Iteration 98
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 222us/step - loss: 0.0507 - acc: 0.9883 - val_loss: 0.1323 - val_acc: 0.9557
    Q 360-584 T -224 [92m☑[0m -224
    Q 812+250 T 1062 [91m☒[0m 1742
    Q 86-664  T -578 [92m☑[0m -578
    Q 14+824  T 838  [92m☑[0m 838 
    Q 53-971  T -918 [92m☑[0m -918
    Q 83-544  T -461 [92m☑[0m -461
    Q 7+99    T 106  [92m☑[0m 106 
    Q 768-611 T 157  [92m☑[0m 157 
    Q 260-668 T -408 [92m☑[0m -408
    Q 521+84  T 605  [92m☑[0m 605 
    
    --------------------------------------------------
    Iteration 99
    Train on 36000 samples, validate on 4000 samples
    Epoch 1/1
    36000/36000 [==============================] - 8s 220us/step - loss: 0.0477 - acc: 0.9893 - val_loss: 0.1491 - val_acc: 0.9494
    Q 716-28  T 688  [92m☑[0m 688 
    Q 68+114  T 182  [92m☑[0m 182 
    Q 73-672  T -599 [91m☒[0m -609
    Q 970+434 T 1404 [91m☒[0m 1304
    Q 899-875 T 24   [91m☒[0m 14  
    Q 626-946 T -320 [92m☑[0m -320
    Q 858+573 T 1431 [91m☒[0m 1331
    Q 965-753 T 212  [92m☑[0m 212 
    Q 80+239  T 319  [92m☑[0m 319 
    Q 14-461  T -447 [92m☑[0m -447
    

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
    Q 684+55  T 739  [92m☑[0m 739 
    Q 6+674   T 680  [92m☑[0m 680 
    Q 345-842 T -497 [92m☑[0m -497
    Q 752-133 T 619  [91m☒[0m 629 
    Q 988+227 T 1215 [92m☑[0m 1215
    Q 606-94  T 512  [91m☒[0m 511 
    Q 728+48  T 776  [92m☑[0m 776 
    Q 913-55  T 858  [91m☒[0m 868 
    Q 45+21   T 66   [92m☑[0m 66  
    Q 708+33  T 741  [92m☑[0m 741 
    

# Testing 2 (by new question)
Q: 760+172, 529+39 , 227-530, 866+777, 10-879 , 630-342, 235-111, 688+524, 999+166


```python
newQ = ['760+172', '529+39 ', '227-530', '866+777','10-879 ','630-342', '235-111','688+524', '999+166']
newA = ['932', '568 ', '-303', '1643', '-869', '288', '124', '1212', '1165']
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
    Q 760+172 T 932  [92m☑[0m 932 
    Q 529+39  T 568  [92m☑[0m 568 
    Q 227-530 T -303 [92m☑[0m -303
    Q 866+777 T 1643 [92m☑[0m 1643
    Q 10-879  T -869 [92m☑[0m -869
    Q 630-342 T 288  [92m☑[0m 288 
    Q 235-111 T 124  [92m☑[0m 124 
    Q 688+524 T 1212 [92m☑[0m 1212
    Q 999+166 T 1165 [92m☑[0m 1165
    


```python

```
