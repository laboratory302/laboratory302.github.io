# RNN with tensorflow2.0


```python
import tensorflow as tf
import numpy as np
```

## Be sure to used Tensorflow 2.0


```python
assert hasattr(tf, "function") # Be sure to use tensorflow 2.0
```

## Open and process dataset


```python
# You can used your own dataset with english text

with open("rnn_dataset/victorhugo.txt", "r") as f:
    text = f.read()

print(len(text))

print(text[:1000])

```

    127286
    Parce que, jargonnant vêpres, jeûne et vigile,
    Exploitant Dieu qui rêve au fond du firmament,
    Vous avez, au milieu du divin évangile,
    Ouvert boutique effrontément ;
    
    Parce que vous feriez prendre à Jésus la verge,
    Cyniques brocanteurs sortis on ne sait d'où ;
    Parce que vous allez vendant la sainte vierge
    Dix sous avec miracle, et sans miracle un sou ;
    
    Parce que vous contez d'effroyables sornettes
    Qui font des temples saints trembler les vieux piliers ;
    Parce que votre style éblouit les lunettes
    Des duègnes et des marguilliers ;
    
    Parce que la soutane est sous vos redingotes,
    Parce que vous sentez la crasse et non l'œillet,
    Parce que vous bâclez un journal de bigotes
    Pensé par Escobar, écrit par Patouillet ;
    
    Parce qu'en balayant leurs portes, les concierges
    Poussent dans le ruisseau ce pamphlet méprisé ;
    Parce que vous mêlez à la cire des cierges
    Votre affreux suif vert-de-grisé ;
    
    Parce qu'à vous tout seuls vous faites une espèce
    Parce qu'enfin, blanchis dehors et noirs dedans,
    Criant
    

## Remove character and create vocab
![](./images/rnn_vocab.png)


```python
import unidecode

text = unidecode.unidecode(text)
text = text.lower()

text = text.replace("2", "")
text = text.replace("1", "")
text = text.replace("8", "")
text = text.replace("5", "")
text = text.replace(">", "")
text = text.replace("<", "")
text = text.replace("!", "")
text = text.replace("?", "")
text = text.replace("-", "")
text = text.replace("$", "")

text = text.strip()

vocab = set(text)
print(len(vocab), vocab)

print(text[:1000])

```

    34 {'n', ':', '.', ',', ' ', 'v', "'", 'c', 'a', 'k', 'e', 'm', 'y', '\n', 'l', 'w', 'd', 'p', 'q', 'g', 'z', '"', 'x', 'u', 'j', 'f', 'i', 'r', ';', 't', 'b', 'h', 'o', 's'}
    parce que, jargonnant vepres, jeune et vigile,
    exploitant dieu qui reve au fond du firmament,
    vous avez, au milieu du divin evangile,
    ouvert boutique effrontement ;
    
    parce que vous feriez prendre a jesus la verge,
    cyniques brocanteurs sortis on ne sait d'ou ;
    parce que vous allez vendant la sainte vierge
    dix sous avec miracle, et sans miracle un sou ;
    
    parce que vous contez d'effroyables sornettes
    qui font des temples saints trembler les vieux piliers ;
    parce que votre style eblouit les lunettes
    des duegnes et des marguilliers ;
    
    parce que la soutane est sous vos redingotes,
    parce que vous sentez la crasse et non l'oeillet,
    parce que vous baclez un journal de bigotes
    pense par escobar, ecrit par patouillet ;
    
    parce qu'en balayant leurs portes, les concierges
    poussent dans le ruisseau ce pamphlet meprise ;
    parce que vous melez a la cire des cierges
    votre affreux suif vertdegrise ;
    
    parce qu'a vous tout seuls vous faites une espece
    parce qu'enfin, blanchis dehors et noirs dedans,
    criant 
    

## Map each letter to int


```python
vocab_size = len(vocab)

vocab_to_int = {l:i for i,l in enumerate(vocab)}
int_to_vocab = {i:l for i,l in enumerate(vocab)}

print("vocab_to_int", vocab_to_int)
print()
print("int_to_vocab", int_to_vocab)

print("\nint for e:", vocab_to_int["e"])
int_for_e = vocab_to_int["e"]
print("letter for %s: %s" % (vocab_to_int["e"], int_to_vocab[int_for_e]))
```

    vocab_to_int {'n': 0, ':': 1, '.': 2, ',': 3, ' ': 4, 'v': 5, "'": 6, 'c': 7, 'a': 8, 'k': 9, 'e': 10, 'm': 11, 'y': 12, '\n': 13, 'l': 14, 'w': 15, 'd': 16, 'p': 17, 'q': 18, 'g': 19, 'z': 20, '"': 21, 'x': 22, 'u': 23, 'j': 24, 'f': 25, 'i': 26, 'r': 27, ';': 28, 't': 29, 'b': 30, 'h': 31, 'o': 32, 's': 33}
    
    int_to_vocab {0: 'n', 1: ':', 2: '.', 3: ',', 4: ' ', 5: 'v', 6: "'", 7: 'c', 8: 'a', 9: 'k', 10: 'e', 11: 'm', 12: 'y', 13: '\n', 14: 'l', 15: 'w', 16: 'd', 17: 'p', 18: 'q', 19: 'g', 20: 'z', 21: '"', 22: 'x', 23: 'u', 24: 'j', 25: 'f', 26: 'i', 27: 'r', 28: ';', 29: 't', 30: 'b', 31: 'h', 32: 'o', 33: 's'}
    
    int for e: 10
    letter for 10: e
    


```python
encoded = [vocab_to_int[l] for l in text]
encoded_sentence = encoded[:100]

print(encoded_sentence)
```

    [17, 8, 27, 7, 10, 4, 18, 23, 10, 3, 4, 24, 8, 27, 19, 32, 0, 0, 8, 0, 29, 4, 5, 10, 17, 27, 10, 33, 3, 4, 24, 10, 23, 0, 10, 4, 10, 29, 4, 5, 26, 19, 26, 14, 10, 3, 13, 10, 22, 17, 14, 32, 26, 29, 8, 0, 29, 4, 16, 26, 10, 23, 4, 18, 23, 26, 4, 27, 10, 5, 10, 4, 8, 23, 4, 25, 32, 0, 16, 4, 16, 23, 4, 25, 26, 27, 11, 8, 11, 10, 0, 29, 3, 13, 5, 32, 23, 33, 4, 8]
    


```python
decoded_sentence = [int_to_vocab[i] for i in encoded_sentence]
print(decoded_sentence)
```

    ['p', 'a', 'r', 'c', 'e', ' ', 'q', 'u', 'e', ',', ' ', 'j', 'a', 'r', 'g', 'o', 'n', 'n', 'a', 'n', 't', ' ', 'v', 'e', 'p', 'r', 'e', 's', ',', ' ', 'j', 'e', 'u', 'n', 'e', ' ', 'e', 't', ' ', 'v', 'i', 'g', 'i', 'l', 'e', ',', '\n', 'e', 'x', 'p', 'l', 'o', 'i', 't', 'a', 'n', 't', ' ', 'd', 'i', 'e', 'u', ' ', 'q', 'u', 'i', ' ', 'r', 'e', 'v', 'e', ' ', 'a', 'u', ' ', 'f', 'o', 'n', 'd', ' ', 'd', 'u', ' ', 'f', 'i', 'r', 'm', 'a', 'm', 'e', 'n', 't', ',', '\n', 'v', 'o', 'u', 's', ' ', 'a']
    


```python
decoded_sentence = "".join(decoded_sentence)
print(decoded_sentence)
```

    parce que, jargonnant vepres, jeune et vigile,
    exploitant dieu qui reve au fond du firmament,
    vous a
    

## Sample of one batch
<img src="./images/rnn_letter.png" width="400px" ></img>


```python
inputs, targets = encoded, encoded[1:]

print("Inputs", inputs[:10])
print("Targets", targets[:10])
```

    Inputs [17, 8, 27, 7, 10, 4, 18, 23, 10, 3]
    Targets [8, 27, 7, 10, 4, 18, 23, 10, 3, 4]
    

## Method used to generate batch in sequence order


```python
def gen_batch(inputs, targets, seq_len, batch_size, noise=0):
    # Size of each chunk
    chuck_size = (len(inputs) -1)  // batch_size
    # Numbef of sequence per chunk
    sequences_per_chunk = chuck_size // seq_len

    for s in range(0, sequences_per_chunk):
        batch_inputs = np.zeros((batch_size, seq_len))
        batch_targets = np.zeros((batch_size, seq_len))
        for b in range(0, batch_size):
            fr = (b*chuck_size)+(s*seq_len)
            to = fr+seq_len
            batch_inputs[b] = inputs[fr:to]
            batch_targets[b] = inputs[fr+1:to+1]
            
            if noise > 0:
                noise_indices = np.random.choice(seq_len, noise)
                batch_inputs[b][noise_indices] = np.random.randint(0, vocab_size)
            
        yield batch_inputs, batch_targets

for batch_inputs, batch_targets in gen_batch(inputs, targets, 5, 32, noise=0):
    print(batch_inputs[0], batch_targets[0])
    break

for batch_inputs, batch_targets in gen_batch(inputs, targets, 5, 32, noise=3):
    print(batch_inputs[0], batch_targets[0])
    break
```

    [17.  8. 27.  7. 10.] [ 8. 27.  7. 10.  4.]
    [ 8.  8. 27.  7.  8.] [ 8. 27.  7. 10.  4.]
    

## Create your own layer


```python
class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)
```

Test if the layer works well


```python
class RnnModel(tf.keras.Model):

    def __init__(self, vocab_size):
        super(RnnModel, self).__init__()
        # Convolutions
        self.one_hot = OneHot(len(vocab))

    def call(self, inputs):
        output = self.one_hot(inputs)
        return output

batch_inputs, batch_targets = next(gen_batch(inputs, targets, 50, 32))

print(batch_inputs.shape)

model = RnnModel(len(vocab))
output = model.predict(batch_inputs)

print(output.shape)

#print(output)

print("Input letter is:", batch_inputs[0][0])
print("One hot representation of the letter", output[0][0])

#assert(output[int(batch_inputs[0][0])]==1)

```

    (32, 50)
    (32, 50, 34)
    Input letter is: 17.0
    One hot representation of the letter [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    

# Set up the model

<img src="./images/architecture_rnn.png" width="400px" ></img>


```python
vocab_size = len(vocab)

### Creat the layers

# Set the input of the model
tf_inputs = tf.keras.Input(shape=(None,), batch_size=64)
# Convert each value of the  input into a one encoding vector
one_hot = OneHot(len(vocab))(tf_inputs)
# Stack LSTM cells
rnn_layer1 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(rnn_layer1)
# Create the outputs of the model
hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)
outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(hidden_layer)

### Setup the model
model = tf.keras.Model(inputs=tf_inputs, outputs=outputs)
```

## Check if we can reset the RNN cells


```python
# Star by resetting the cells of the RNN
model.reset_states()

# Get one batch
batch_inputs, batch_targets = next(gen_batch(inputs, targets, 50, 64))

# Make a first prediction
outputs = model.predict(batch_inputs)
first_prediction = outputs[0][0]

# Reset the states of the RNN states
model.reset_states()

# Make an other prediction to check the difference
outputs = model.predict(batch_inputs)
second_prediction = outputs[0][0]

# Check if both prediction are equal
assert(set(first_prediction)==set(second_prediction))
```

## Set the loss and objectives


```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=0.001)
```

## Set some metrics to track the progress of the training


```python
# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
```

## Set the train method and the predict method in graph mode


```python
@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batch
        predictions = model(inputs)
        # Get the error/loss on these predictions
        loss = loss_object(targets, predictions)
    # Compute the gradient which respect to the loss
    gradients = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
def predict(inputs):
    # Make a prediction on all the batch
    predictions = model(inputs)
    return predictions
```

# Train the model


```python
model.reset_states()

for epoch in range(4000):
    for batch_inputs, batch_targets in gen_batch(inputs, targets, 100, 64, noise=13):
        train_step(batch_inputs, batch_targets)
    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()
```

## Save the model


```python
import json
model.save("model_rnn.h5")

with open("model_rnn_vocab_to_int", "w") as f:
    f.write(json.dumps(vocab_to_int))
with open("model_rnn_int_to_vocab", "w") as f:
    f.write(json.dumps(int_to_vocab))
```

# Generate some text


```python
import random

model.reset_states()

size_poetries = 300

poetries = np.zeros((64, size_poetries, 1))
sequences = np.zeros((64, 100))
for b in range(64):
    rd = np.random.randint(0, len(inputs) - 100)
    sequences[b] = inputs[rd:rd+100]

for i in range(size_poetries+1):
    if i > 0:
        poetries[:,i-1,:] = sequences
    softmax = predict(sequences)
    # Set the next sequences
    sequences = np.zeros((64, 1))
    for b in range(64):
        argsort = np.argsort(softmax[b][0])
        argsort = argsort[::-1]
        # Select one of the strongest 4 proposals
        sequences[b] = argsort[0]

for b in range(64):
    sentence = "".join([int_to_vocab[i[0]] for i in poetries[b]])
    print(sentence)
    print("\n=====================\n")
        
```

     soldats  mais be distoire ; et sur toute chose.
    et la verite, moynt, le ruste range et vivant.
    le sorvent des mains laimsieres ange leur plaine ;
    le soir tout a fais qui donne a la tombe une cueux,
    que c'est la lueure et crache a l'ocean maisee,
    que de la rerre est ma prison est charmante 
    et je m'
    
    =====================
    
    
     souffle sons fuit de pourrir augon vertunt.
    allem d'eux aux morts joyeux, seuls de maie et grande bourser vos demes de la nuit sur la partue ;
    je n'ai jamais souffert qu'on osat y trace fais lainsi vous croyez l'esprit du poetre fe l'aire
    des fleurs sont par plaind tout sous les degots de ce vere,
    
    =====================
    
    e la sainte vierre,
    la nature, ainsi que la france au fond de l'ombre.
    
    vii
    
    
    essait pour tout un huit, n'estce pas de rire 
    le soir tombe en pred,ant le vide et la cipiee,
    il passene de toi, pour l'amere repouverte,
    tout a ces juste et mains lien sont distincess
    fait que nous nous connez dans l'omb
    
    =====================
    
     est un pubeur et l'aige un revagt.
    le pres de l'aut, ou dans ce timain ne semour,
    qu'a notre ame adeux de la raison de sa foi,
    entre disqu qu'en ce loyant que le sort die a fonde,
    j'ai vu ton coeur tends au bien dun faut du terre 
    le preson dans la maison pour qu'on ollait vaine.
    
    il est bouche  a 
    
    =====================
    
    e au ciel, manque au grand mourir profond.
    ou  tous les rois vaines de la mer descendee,
    chantez, flots, abjmir, ainsi qu'en du honne faire,
    que les enfans versetns au milieu des remes escoles,
    il est plus ainsi toujours vague au tout de fends.
    
    o songe a son ouvrant et douceur de la chaume ;
    je vou
    
    =====================
    
     a l'heure ou la nuit sourier
    que la confais, le mon eut la lache couveet
    le pet de l'assageneril est bas sparque ;
    harche le fixr des terts au fant de l'ame en veiller.
    
    on derait aux mains ma vague ouvrire des ames,
    que vous en astends pas assez de sa parte 
    decoure le grande homme a detient melli
    
    =====================
    
    au serrire un peu de ces fronts merviere.
    les auracherts, des vieux, les oybres, les ou dieux 
    
    nous detions les mains les martyrs sonnelles,
    la voule ils ont tous courtez sous les presses,
    vous etequi sur nos terntes sur les mars es fremeus ;
    les impres sont des voins de toutes nos devesses,
    soit q
    
    =====================
    
    le et ma peulle et sur tes larmes,
    les grands hommes ont fait leurs francs de l'autreche
    de lorse en voyagt l'avenir.
    ainsi, saus ne te charrer au sent fait pleurer et tue saintent mon enfant, sans d'hombre aurombe 
    alors, que nous voyons sers pour l'aime du premiere
    mainta tout souffle charte, ange
    
    =====================
    
    
    e de sa fosse passe,
    et que je vous vois encre dans le beau moins l'ondre
    sans ce qu'al est pursque j'aime aussi le la tete,
    et que j'ai fait plus bai que le fort de tout serre,
    on a renon d'autrefois que le jarrais pour moi.
    un brille d'au raster des cloirs de l'aurore,
    et de nous du moins pourris
    
    =====================
    
    le a son front de son ame achete
    tant qu'une fuille au milieu des rayons ou vierger.
    quand un temps au crime ou n'ait pas vent creve t
    que vous enes confrez, la ruit, ou l'on peut ;este,
    sans qu'un ame estroutnant souffre et ce qui fait 
    de motre en treveus sur la creation.
    c'est alors qu'a dans la 
    
    =====================
    
    s et la sainte laie ;
    j'ai vu tete la tete et mais ;
    
    parce que vous etiez alosse
    et qu'alle ainti nous convient aais l'onde sous les nids je deses
    les hontes aux feuves et les mons,
    songez en souvenirs des vanges,
    un vague puissant couverte 
    comme un peuple autour de l'accer.
    
    vous aumez vous nous 
    
    =====================
    
    et sombre,
    et sans cesse et nuit, pour m'appelle au regour,
    dans le calme du coeur, dans la vain en fanta
    si passez le pritrir et marchant devant lui
    toutes les profondeurs de l'ombre et de l'ennui,
    l'ambre et l'esprit, le vien qui dort ahvivail.
    on ne sait quelque aison peu d'autre pole andre,
    et l
    
    =====================
    
    et l'ame a des fleurs,
    vous qui voulez des fleurs, vous qui vous saiti.
    
    j'ai vu ton coeur noit sa chaire,
    et le vent porte au hilon,
    dont la charge de l'artre ;
    et la verte sur les lieux 
    
    ville, c'est le moi qui somme,
    silen qu'une faut de tes yeux
    mers comme les ailes de l'aurore,
    ils surgir aux 
    
    =====================
    
    e a son aire a l'orage et le lave qui t'emporte
    melli de l'ame entone, nous vous etes envier 
    je vois en facait vous etes un piedre sur la tete,
    et qu'il soit enrangn qu'en vous avez la soin,
    et comme un apyeci, sombre haine a mon ame,
    tevrre a pres de voir ressemer que je pars  
    
    parce que vous con
    
    =====================
    
     au frond ses sombres viles foies,
    font d'un heux saisis de ces frondes sinistres,
    la verte par la piede et la france aux vents,
    il est passer enfant, je suis l'homme aident 
    quoi  quand je luis vous conner, de ton tour est car d'oistine,
    elle en une foret d'esprit  natue crimine,
    de l'homme est un 
    
    =====================
    
     en sent croire et de contre en est ces hommes 
    
    qui de nous  vanqus tait les chaines du meme voile,
    la sainte liberte en va faire aux vents,
    il eaait au frond les soldats est un sauvitre le chau.
    
    je vous en faise est vot, que je vais son maintre.
    le parthetoi sans des morts taineraires aux ames,
    q
    
    =====================
    
    e en ces chants sour paix a son piede la main,
    et quand on se fait pleure en la saille au viel 
    de ne ton est la laiene, a l'immense ameut.
    ou sombre ou chacun mes cheveux dans la pierre,
    qu'il est donc son nefant qui dechire se la sombre
    qui hont en treson de voir de l'autre bonde un peut 
    le prosc
    
    =====================
    
    llon, contre tout, heureux, desange  tu seroirs.
    la se pince en revant ten flois des temps nouveaux,
    le pouvoir, les honneurs sont vos plus demendre,
    les sombres leur camons ou tout a la tete ans chamme ;
    on ne saut, qu'en les ois qu'il souffle en devite,
    quand ta mere au hait s'eternel et la lavast
    
    =====================
    
    e au regour,
    jardure ou libre sans en fait aux sont d'imparter.
    pour cette vous bas chaque jombeaun a moi 
    de l'arrente dans ses deux morts le pas destures
    quand nous sommes en juit 
    
    oh  que vous etes enernce  comme un ange patre
    des derleurs de l'autel, et la tombe, a la nuit,
    que j'ai fais c'etoi
    
    =====================
    
    le reste ronne a mais nuit tout est vain.
    
    tu parler de mourir, la moutieu le sang,
    tachand du semant hermeil au sent trahieon etonne.
    en ne saut, qu'en lavre etonne et tremble un trosse ;
    nous avons un instant est le roule en de lamais,
    maldans de l'amsent ont autas mille clamme ;
    la nature est un 
    
    =====================
    
    esseur de l'aurone,
    il essant l'etre entrers le rayon.
    
    c'est aimen.
    
    l'hafale mansiens, garder ta fiere,
    et ce qu'on a pardait sous les chants disperues.
    quand ton ame entante avors quelque vieux d'un homme,
    que nous aimions voisse est vous, voys ez sans dieu
    ssi, dous conque tu de cet hare, inquit
    
    =====================
    
    e sans puissant des rayons du dorte,
    mon enfent de son front que ce qui connait le mal,
    tandu pouvetu le peuple a ressemble sur la porte
    me uient aux rempois de la mer de l'etrice.
    je n'ai pas ve le rese et de son tere, paysere
    dans la nuit nous chantez oans ma creation d'aiglete ;
    il ne se vent pas
    
    =====================
    
     dans sous nos matsoes pasier sur l'encense,
    sien veux qu'en ce monde a ce que cette faute,
    nous commations, c'est au frinds nous repardaes la delce.dans la fange au coude ou les orages
    de l'aquilons des oiserains ;
    el vente toute speil qui ne sousons disaots
    et sur toute chose et tout souvenir,
    ale
    
    =====================
    
     sur les traitres de l'onde
    sous les vieux cent vours et des oeux et de nous voisser
    sur ta chaine ferme est tourne le meme sage,
    halde de l'equipende au conserau de la tombe,
    je suis content encore et grande et sur l'espait
    qui hous enfant e'tres qu'on lui pas se sourire 
    qu'elle ait qu'un jour soi
    
    =====================
    
     sombre ou l'etre et la terre
    que tu navhes pas au fais au frande 
    que les enfers dont la martyge,
    dans le beau mois de mai,
    la chose la plus douce
    dans le beau mois de mai
    c'est quand on est aime.
    
    parcourez les charmilles,
    les sources, les buissons ;
    autour des jeunes filles,
    les sources, les buis
    
    =====================
    
    e et par l'autre aussi, dans sa gaison,
    le present aux peuples de la mer de te fere.
    je suis roujours recoutais ; la faute de ciel 
    n'est tout en mardant,
    retait ce qu'on voit malhitre, et sans droit, mempleids 
    o mon manteau batte, et moi silencle en ange
    peur qu'elle pense a voix donne a ces lumie
    
    =====================
    
     et mois bien, meme moins mieux repands 
    quoi  mon coude est toujne, et qu'on se muit pas pas usre
    que ton sauve a travers les lumieres du monde,
    toujours plains d'en qui donne a son diseau ;
    
    parce que vous courbont, vous nous donne cinintuyes,
    et dont la causemend de l'ame et dans les passes.
    je v
    
    =====================
    
     : ton sange et de cette inoomes,
    et s'etre doux montre, et riant en enfant,
    veut s ylange qu'il voit alle rayonnant....   que dieu res enfants, et j'ampoure nous attendre,
    la voix dieu fit, a bien, je surg en faut se mene.
    et quand, ce qu'un est voile ou son passer un qui faite
    et l'on terrait vers
    
    =====================
    
     l'aube et la tempete
    quand nous noms nous enfant ;
    
    si je ne vous vois pas comme une belle femme
    marchande, et l'avenir, avra brise a son langer
    qu'il est doux d'ynvendre et l'onde et le lache chent ;
    el fonte a touche, criere, et toi, je vois entremble en peu mainte,
    le jour ou tout ce qui n'achom
    
    =====================
    
    e saigne aux passants sur les pieds 
    
    quoi  mon vieux heut vous, nous voulez la donce 
    et tout somme a souffle a tous ceux de nouvelles,
    la sendue les emeuxs de la creation.
    
    ce livre, legion solcre a son fatal dieu,
    ce pas un que le vieille au fidele dans la contamente veux pas sent en reale et le 
    
    =====================
    
    l'esprit. l'etrand jour et le suis me propiere,
    il peut sue l'homme hn pleurs  le revache leur glaive,
    elle elle se fouet mulliere, et j'allais a ma poince ;
    et t'atoure entrerastre et l'amour de phryne.
    les herreur se premiereraporrre a ses pervecc ;
    vous etes ce qui fait que la forme et le lieu sa
    
    =====================
    
     et machant, mais connamt,
    je m'ai vu te crappee, et dieu, ce que je dit :
    j'ai vouifais coupre uet et qu'un souffle est faite
    que je n'entenden ap pas que chante difforme,
    je t'ai vu te trainais, j'ai crevais tout aissins,
    et sur l'ama eternel nous avec le noir qui passes 
    
    a cette ainsi.uue et de 
    
    =====================
    
    amant ;
    et s'astre noir crmme et sans crainte, et sans crainte,
    sachezqu'en traveuit plus qu'on ne souffre sombre,
    j'ai vu tes vetements d'azur, ivientent distruis 
    quoi  tout mange a crache, en rouant mauvais,
    a celler de l'autelent dans un combat le passere
    je ne proclame au ciel, de la sainte mai
    
    =====================
    
    le et sans rele mille clocher et trinde,
    comme aupourdais le bruit du monde
    le jette murain qui le rait pase
    l'esprit que nous aille ou la voix du travaille,
    et qu'il soit aller alle ciel.
    je ne suis pas chntre. ce que j'aire austera pas 
    
    il peut sur le souffle et la france et le laime,
    l'epeindr, 
    
    =====================
    
    l'esprit. un panthe ou lieu qui romge,
    si, dans ce cloaque ou sont fait le passere,
    je vis  j'ai trois enfants et l'ordure a ses pretres ;
    dit je t'ai vu te revers le fruit de l'accendre ;
    un fomen, oh la sorte et fletri en enfant,
    vaince melle mal quessonne et que les pres nos sommes
    que fais un tr
    
    =====================
    
    le eternel et sans rien est trop plombi 
    oui, je leur creait alors qu'elle espre au vieux puits,
    cette vision de chaque inqule,
    amaiseau, furire leur moin presse 
    je suis rien ne soit qu'au fond des shmmes,
    invants dans l'esprit humain tout etant qui me nous aile,
    que le jour de l'ame en ens de l'au
    
    =====================
    
    rendre et la sainte paile,
    et la sageste melle et la france aux vents,
    alloment, dans l'avenir repussiena qu'on voie
    ellememe d'ame roeur, dans ces sienle  de france,
    foule aux pieds vivreux aux lumieres du collir 
    
    que le vous est pas un plus conquerque a couple.
    il parait que l'argre et la bourreu
    
    =====================
    
    e sans reliere,
    et qu'alayour tous les hamps et les monstres flemes 
    comme en sont, toujours vous et des deux esclaveses
    frendrent les coeurs mustants.
    
     vous nous voulez la mer sportre dissonde en trop doule
    rome et son tarait ; et s'en fait le pour eux.
    vous croyez qu'en tous croissez la france et
    
    =====================
    
     et sans relache ;
    je n'ai ja ma metirais qui croyait tout le couche,
    sont dit passer souverran yu'esport sur le sombre,
    on ne saus pas le mourgre en de ces croses puisse
    les donc attendre, persieu de getre bisere,
    par l'ombre est un juit, un peu couvre au croid,
    aux nubls que jamais les douleurs su
    
    =====================
    
     et sur les cieux met sa flamme,
    apres tans cessieurs branches et de la mer sur l'empire ;
    je n'ai jamais souffert qu'on osat y t'amant,
    mes fleurs est paris des cieux met sans mort,
    les preneaient leur coulse auguste, par la vieile
    s'aurare au hond du ciel  l'orange a la tombant,
    j'ai voulu la sagr
    
    =====================
    
     et son esprit, arrepour de l'ame
    a cette fine de la ranon.
    
    vous vous crezez les mains le grand sombre satreil,
    aux nuits aux paymi tes visions et des oragns 
    car la pierre qu'en ce livre, et par lui ta te faite
    melas .au donc est le mit, parce qu'il soit aller 
    homme d'isting flattant aux remons d
    
    =====================
    
    les deux est garder ton ame :
    faissezvois ces heros, cherche en trouver,
    ceur autre son front que des remes ecorilles ;
    il chasse comme un chien le grand tiergentere,
    qu'il est des remetues de la merse fatale,
    sans que dous les rois, travant les mallipullants,
    des rafils, les aigref et les onds d'et
    
    =====================
    
    de l'ame, ou j'allais le reste, o mon ame,
    si, triste qu'en moi creat dans l'horizon des pensers.
    qui dans les noirs taillis ton oeil verse la mort,
    puisqu'il est d'autrir tant au seuil du dieu qui represait au pleure, c'est le premiere a ton nois bleuse ;
    j'ai vu tes vetements d'aitasses d'une lame
    
    =====================
    
     et sombre,
    et sans cesse et nuit, pour m'avfnire,
    et qu'il soit avant versant d'un pilieu de cette,
    en de ces quits quelqu'un qui le sorr un refait,
    que d'acenteraine, nous vois entremen de ciele,
    toute la causement, c'est un peu de canste,
    les sous et pas moirir qui descendres commette souffres,
    e
    
    =====================
    
    de, et sans relaine,
    et, tout sombre, o matine et sans relache 
    arrepousqu'un plaient manquer sous les presres contrents.
    au sond des cheveux blancs de l'aurore et de l'onde
    se font euvre un bonde avec une parole.
    les prenne, et leur lace et le vie est sa colere.
    
    hiere en vous croyez qu'elle le fai
    
    =====================
    
      et soupiri, enfant,
    sous l'astre noir nans ma pretre,
    et trouve la batanne aux beux metres ;
    que le repurd sur la clairon
    
    son noeux qu'il feut parfor, par la pleure,
    l'aubre sur son vosxgque j'aille a notre et ferme.
    ii
    
    
    aimes  c'est la douce loi.
    
    vous voyant encor leurs bates
    la faiseau hurain
    
    =====================
    
    es fuyant m'a dit le tete 
    et quand on est atce leur la trouve par le vent qui le repaire,
    et ce qu'on ast pas un trouve farmon,
    a nous dire : les etiennen
    de sous la bumienne au ciel 
    
    hi les epeleues de leur aile
    la soule archent toi fille
    ou le surt qu'il soit en rependre
    qu'un tourmou sait quel 
    
    =====================
    
     sans rele mirant en souffle qui rayonne au vole,
    et le vent qui ta marture,
    tu mains leurs mysterieux.
    
    avrille  moi sifflamme,
    et toute embre et de centre et aile laisse sur ta tete,
    et, comme l'oiseau de l'ocean.
    c'est toi qui, vous etiez en astere,
    si tourmente qui tremple et tour,
    sont la saint
    
    =====================
    
    l est sans reven
     par le soir qui se foule a ta verse 
    c'est a nous de chanter le plus exquis peutetre ;
    on chatouille moins pour roi tout peu de charmere
    de la republique et craindra l'etaile de l'ombre.
    tu desperte veut main presdsans la nuit touche 
    autorre qu'en refard sur tous les nains moeterr
    
    =====================
    
    e a la fois baine aux partis ou leschere 
    
    iii.
    aux puis, c'est le mom que j'aile qui sort s'en fait.
    le pouvoir mon coeur,rnous vous endonyor est change,
    un rouen eun coeur l'ame et la grosse fatale,
    tous enforgez qar la mer que te faisons a vertir 
    
    ou sour et la sainte, espoir, de par mon ame,
    
    p
    
    =====================
    
     l'enorge et l'ombre aperdit.
    
    l'homme est entre un jour l'enorme solire ;
    l'ombre ressemant des ebles et de l'autendre,
    et la sagesse ame rete et le regard,
    prendre a l'ombre affrect en ce levr qui le fait ;
    le nous pas que l'empere reait hemonder ;
    l'orgueil qu'un homme aime, apresque come et sain
    
    =====================
    
    endement ;
    
    parce que vous celiez pas l'avez de sa cole,
    et tout garde une parfois le temps a tous.
    o poussione je fais paisait encor.
    moi que l'obscure autour de l'ace en un flot ee mainte,
    et tout gaite et marchant devant qu'on t'itte,
    en depis tet mains parmour le vent de sa hoin,
    et t'estu recla
    
    =====================
    
    e sur la pourriere amert.
    
    les feuilles de l'aire en verteille.
    ils lions aux mains les temps setournent la prese boire ameuse,
    et c'est un nombre faniour 
    
    tout ce qui ne suis entrer dans mon ame en sans fille
    la france pour le morde en flambeaun
    des beaux impulmeteus de la mer 
    instrederait aux na
    
    =====================
    
    reveur de l'aurore,
    et dans ce flome sur l'echive et le lieu 
    il etait au senat l'amas et putte ame qu'on voue allee
    dans dans quelque comme un somne a qui tremble ;
    hieu  je t'ai vu te revers en de la clarmiere
    la loi saigne aux morts clarges, le seint de l'autre,
    le bon, le sueril des romes et la 
    
    =====================
    
    
    
    
    ainsi qu'une cyoye ou nous voision cime 
    l'ame re l'ombre abjourd ou l'on eppule,tun metom ser infire a ton front du vivre
    se me descends de toutes pas les vasses pleurs,
    et ritent le mot fuyant et qu'on se mart de foue
    descend de la cercue au milieu de ses hommes,
    on entre tigne des mains mains 
    
    =====================
    
     l'espace pleure,
    ainsi qu'une cyoix en tout ce qui contre la tete,
    de l'ame enterrait ses douleurs.
    
    aime, a l'immense de la chaire,
    et le bourgeaise a l'oeil en vaste faire,
    la sainte lien apre avec les temps sainteres,
    pour vous qui vous et reverient :
    
    si je ne vous vois pas gardrd de ton tour e
    
    =====================
    
     faut se meme et sans dessense avec les bras 
    
    quand je pense qu'il soulf a sa fante aimarte 
    le fruit de l'ame, on bien donne a la tombe,
    si piele du charger ann la voix triste et doux,
    le jour ou tout ce qui ne songe et que je carete.
    j'ai creuse la lueur qu'on voit passer le voile,
    et que je vais
    
    =====================
    
     est sour quel donc est main ne sous en fait ;
    le cheva a dieu pas ce que l'honneur petit prome,
    et que je vais, sur ta treme et sur toute cenche,
    frrtournons le meurtre, et plus j'envecest vaine.
    et caiste comme une porte est un pas plus de cendre.
    qui somme en que c'etant. c'est que votre mais noi
    
    =====================
    
    e et la sainte au fond de l'aurore,
    et ce que j'ai forme sour le plus exque je choire
    j'ai pris ce que j'avais dans le coeur de ma poinc 
    et la nature, au fond des saints et des eaux,
    j'honnet mante, et surtes, comme aimsi les peuilles,
    et marie aux soupirs dans le sang insulte en franche,
    le peuple
    
    =====================
    
    en enfonte lour de l'ace que tous sentons les coeurs mais la bouge en tremble,
    c'est notre chanaan que nous voyons le vent
    n'est en vain n'etes pas assez pas a partil 
    o souvents qu'en boir comme une balalon sombre,
    chaque parlent aussiffle repleur de l'etre borde
    un toit du sacre et moint, nous don
    
    =====================
    
     et sans religuen.
    
    vous vous enfaites persent la douce endormie.
    
    vii
    
    des fleurs  donc l'azri de l'encre est de la lime 
    
    dieu seul vous voyant l'autre au rendre au bruit ;
    quand elle m'eprite en la vague que charmant,
    le sengeal qui flambort et l'eclouchait son front,
    c'etant qu'une lueur sur tou
    
    =====================
    
     soldats frete au milieu des rues 
    qu'il est donc faut ; moi qui donc tout se mourtre
    de l'ame en vient vente au soliau des raains 
    
    u parser est la libere, et l'ame, a maintene,
    et taissement par la vie et tremplatt la memsille 
    le contre la sainte pour l'ambre a ton viers
    puisque le sertine absie 
    
    =====================
    
    eure et sa chaine, et sans cesser,
    puisque j'ai verse la rouge et la trahison basse
    en avancant en age aux la main qui t'emporte,
    mon etennait le vent qu'il souffle enfant la ciel 
    
    fais quelque vous malhez en donaa songes et sans cartouches,
    braves, vous avez les mains mentezt, se mourir passe.
    la 
    
    =====================
    
     sur nos tristes,
    je suis rouleur des chars d'airain.
    
    un bruin hent sa ceinte de ton front sur ma poile ;
    parce que vous etiez prendre aux noirs de ses arres comme la foule au prindre leur sans pleure sur la plaine
    on a change que j'ai creusee les heureurs d'emonder ;
    nous avons un instant cruit et
    
    =====================
    
    


```python
import json

with open("model_rnn_vocab_to_int", "r") as f:
    vocab_to_int = json.loads(f.read())
with open("model_rnn_int_to_vocab", "r") as f:
    int_to_vocab = json.loads(f.read())
    int_to_vocab = {int(key):int_to_vocab[key] for key in int_to_vocab}

model.load_weights("model_rnn.h5")
```
