import keras.models as mod
import keras.layers as lay

bicell = 64 #超参数

# 负责生成bilstm模型
def model_make(maxlen, chars, wordsize, infer=False):
    seq = lay.Input(shape=(maxlen,), dtype='int32')
    embed = lay.Embedding(len(chars) + 1, wordsize,
                          input_length=maxlen, mask_zero=True)(seq)
    bilstm = lay.Bidirectional(
        lay.LSTM(bicell, return_sequences=True), merge_mode='sum')(embed)
    output = lay.TimeDistributed(lay.Dense(5, activation='softmax'))(bilstm)

    resultmodel = mod.Model(input=seq, output=output)
    if not infer:
        resultmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
                            'accuracy'])  # 采用crossentropy, adam优化器
    return resultmodel
