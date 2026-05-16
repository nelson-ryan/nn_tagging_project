### Overview

This repo contains the final code used for a course project,
which was a competition hosted on CodaBench,
in INFO 557 *Neural Networks* in Spring 2026.  
The task was to train a neural model to predict emotion tag labels
(e.g. 'gratitude,' 'anger,' 'love,' 'disapproval') for Reddit comments.

### Approach

My approach was to use SpaCY for tokenization and for its Explosion embeddings,
then to pass those through an n-gram convolution and single hidden layer.

Once the basic architecture was established, I used *Weights and Biases* to 
identify the most effective hyperparameters, including the n-grams' *n*, which
was 2 (of course, it wasn't necessary to limit it to a single *n*; that just
happened to perform well vs. including more convolution windows).

#### More post-project discussion

1. What approaches did you try?

- First off, the basis of all that I did was using SpaCy for tokenization and
for its word vectors. My first approach to the model design was to go big,
combining sizeable convolutional and recurrent portions, trying one or the other,
and a number of manual variations.

2. What approaches worked?

- I got my first reasonable results after I sized it down considerably,
with 1D global max pooling and another hidden dense layer.
This was actually a form of the convolution attempt, but with just unigrams
(so not really a convolution) which I ended up expanding to only bigrams.
Weights and Biases (wandb) was instrumental in discovering an effective
combination of convolutions, activations, optimizer (I went with rmsprop,
but Adam also tended to look good; similarly, I didn't have to use only bigrams,
but it appeared to work well), and other hyperparameters.
Early on, I also added some data augmentation by copying and making tiny
adjustments to the vectors of tagged examples, which I was surprised to find
helped, if not by much.

3. Where did you spend the most time?

- I definitely spent a good deal of time stuck at an F1 around .06,
struggling to get anything to work... until I discovered that I had accidentally
converted all of my vectors to ints, making for some really bad input.
After I figured that out and sized down the network (which was inviting
overfitting), then I spent a bit of time figuring out how wandb is supposed to
work, which was absolutely worth it: let it do its thing for while until you
realize that it doesn't have to try every possible combination because it
already found many that do a pretty good job.

4. What mistakes did you make?

- Starting too big with an oversized network was surely one.
The biggest mistake, though, was surely the corrupted inputs,
which slowed me down a lot. 
There's also a couple of ways I could have eked out some better performance,
but not having done those things isn't really a mistake because the model
performed well enough for its purpose.
