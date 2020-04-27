#. Machine Translation

    - What is the task?
        - Translate a sentence $x$ from the source language to a sentence $y$ in the target language.
    
        
    - What is alignment?

        - how do words correspond to one another from source to target?
        - we actually want to learn $P(x,a|y)$ where $a$ is the **alignment**
        - i.e., "What is the probability of this particular word aligning with that one"
        - relations can be...
            - one-to-one
            - one-to-many ("fertile")
            - many-to-one
            - many-to-many (phrase-to-phrase)

    ##. Statistical machine translation
        - learn a conditional probability distribution from data
        - given a sentence x, find best sentence y : $argmax_yP(y|x)$
        
        - equivalent to $argmax_yP(x|y)P(y)$ by Bayes rule

            - $P(y)$ is a monolingual LM
            - $P(x|y)$ is a translation model learned from parallel data

        - How do you maximize the probability of $y$?
            - "decoding": apply a search heuristic and prune low-probability hypotheses as you traverse the possibility tree
        - What are disadvantages of Statistical machine translation?
            - feature engineering is very involved
            - resource-intensive
            - many separate sub-components 
    
    ##. Neural machine translation

        - directly calculates $P(y|x) = $P(y^{2}|y^{1}, x)P(y^{3}|y^{2},y^{1},x)...P($yT|yT-1,...,y1,x)$
        - why is seq2seq a conditional language model
            - conditional because it is conditioned on source sentence x
            - language model because it is predicting the next word in y

        - how to generate text
            - encoder produces an encoding source sentence (e.g. as its final hidden state)
            - decoder is a conditional language model: produces target sentence conditional on the source embedding
            - first state of decoder is the encoding of the source sentence; first token is <START>
            - first output is probability distribution of next word. take the argmax.
            - feed the output back into the decoder on the next time step
            - recursively generate text until decoder produces the <END> token.

        - advantages over statistical machine translation
            - fluency
            - use of context
            - generalization over similar phrases
            - no separate subcomponents (end-to-end)
            - less feature engineering
            - similar method for many language pairs
        
        - disadvantages to neural machine translation
            - less interpretable
            - harder to debug and attribute error than statistical systems
            - hard to control or override with rules (a safety concern)

        - problems in NMT
            - large output vocabulary requires calculating softmax for every vocabulary item on every timestep
            - but word generation fails if the training vocabulary is too small (<UNK>s in the target language)
            - possible approaches
                - hierarchical softmax, noise-contrastive estimation
                - train on a subset of vocab at a time and test on a smart set of of possible translations
                - use attention to look up the source word and cross-reference it in a dictionary

    ##. Evaluating machine translation

        - Comparative judgments
            - gold standard of machine translation
            - humans rank machine translations

        - Use MTs in a downstream task
            - 'score' is how well the downstream task performs
            - problem: the downstream task may not correlate well with quality of MT
            - certain aspects of MT (like syntax) may be indirectly related if at all to performance on the task

        - What is bilingual evaluation understudy (BLEU)?
            - an automatic measure of goodness of machine translation
            - compares machine-written translation to human-written translation and computes similarity score 
            - weighted geometric mean of **n-gram precision**
                - it is precision-oriented
            - adds a **brevity penalty** if machine translation is much shorter than the human one 
            - rewards high n-gram overlap, so it misses valid translations that are paraphrases

        - Outstanding issues in machine translation
            - maintaining context over longer text than 1 sentence
            - domain mismatch
            - out-of-vocabulary words
            - low-resource language pairs
            - learning biases in source data
            - low resource languages + uninterpretable systems = weirdness
                - parallel corpus of Biblical text + nonsense input means nothing to condition on -> generate random text conditioned on nonsense 

#. Sequence to Sequence

    - What is a problem with the architecture?
        - Two RNNs, encoder and decoder
        - Informational bottleneck: forces all the info about the input sequence into one output vector

    - use cases
        - machine translation
        - parsing (in: input text, out: output parse as sequence)
        - summarization (in: long text, out: short text)
        - code generation (in: language sequence, out: code sequence)
    
    ##. Training the model 
    
    - how to train 
        - feed source sequence into encoder; feed target sequence into decoder
        - for each step of decoder, produce probability distribution of next word and compute the loss using cross entropy.
        - average losses to get total loss for the example
    
    - what does 'end to end' mean?
        - the entire seq2seq model learns with respect to a single loss (we backpropagate through both RNNs)

    - differences in training vs. testing seq2seq
        - training: do not recursively feed decoder output into the model. feed the gold/target sentence into the decoder as $y$ to compute $y-y_hat$ (loss).
        - testing: recursively feed decoder output back into the model.

    ##. Decoding methods

    - How does greedy decoding choose outputs?
        - take argmax on every step of decoder
        - no way to backtrack

    - How does exhaustive search work?
        - which y maximizez P(x|y)?
        - track V^t possible partial translations on every time step

    - How does beam search work?
        - track k most probable partial translations at each step ("hypotheses")
        - k is the beam size (usually 5-10) / size of the search space
        - score of a hypothesis is its log probability (all are negative, higher is better)


#. Attention

    - What is attention?
        - given a set of vector **values** and a vector **query**, compute a weighted sum of the values dependent on the query. (the query "attends to" the values.)
        - on each time step in the decoder, use a direct connection to the encoder to focus on a particular part of the source sentence
        - dot product between decoder hidden state and all the encoder hidden states (one scalar for each source word). 
        - the weighted sum of the values is a **selective summary** of the information in the values, dependent on the query
        - produces a fixed-size representation from an arbitrary-sized input 

    - What are the advantages of attention?
        - solves the bottleneck problem by focusing on parts of the source
        - helps solve the vanishing gradient problem by creating shortcut connections
        - provides some interpretability via the attention distribution
        - unsupervised learning of soft-alignment structure
        
    - What is the attention distribution?
        - the probability distribution obtained by inputting the dot product of the encoder hidden states and the decoder hidden state through softmax
        - highest probability mass will be on most likely word
        - produces **soft alignment**

    - What is the attention output?
        - the weighted sum of encoder hidden states, in which weighting is the attention distribution
        - used to sample the next word (instead of only using decoder hidden state)
        - sometimes, use the previous step's attention output as well

    - What is soft alignment?
        - each word has a distribution over the corresponding words in the source sentence 