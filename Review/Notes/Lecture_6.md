#. Vanishing gradient

	- Vanishing gradient proof

    - Vanishing gradient in practice
    	- We can interpret the gradient as measuring the connection between past and future events.
    	- Impossible to distinguish between two cases:
    	  "There is no dependency between step $t$ and step $t-1$" and
    	  "The model has not learned the true dependency between $t$ and $t-1$".
    
        - What is its effect on RNN language models?
    		- The main problem is preserving information over time.
    		- If RNN-LMs can't learn long-distance dependencies, they won't be able to predict them at test time.
    		- Creates a bias toward sequential recency over syntactic recency:
    		e.g., "The writer of the books are" vs.
    		      "The writer of the books is"
	
	- What kinds of neural architectures does it affect?
		- all neural architectures that use backpropagation (including feed-forward and convolutional networks)
		- especially deep ones
		- due to the chain rule or choice of nonlinearity, lower layers have a far smaller gradient and change much slower during SGD
		- RNNs are **particularly** unstable because they multiply **the same** weight matrix by itself many times
	- Architectural solutions to vanishing gradients

		What are residual connections?
			- a.k.a. skip connections
			- preserve information by default with identity function
			- add direct connections from lower layers to higher ones / allow gradient to flow 
			- instead of just transforming x to f(x), transform it to f(x) + x
			- without resnet, deep network perfoms worse than shallow ones

		What are dense connections?
			- connect all layers to all layers with skip connections

		What are highway connections?
			- Use a gate to control balance between adding x / identity and computing the transformation
				e.g., gate*f(x) + (1-gate)*x

#. Exploding gradient

	- recall: stochastic gradient descent minimizes the loss $J(\Theta)$ by taking a step in the direction of negative gradients, $- \Delta_{\Theta}J(\Theta)$:
	$\Theta^{new} = \Theta^{old} - \alpha\Delta_{\Theta}J(\Theta)$

	If the gradient is big, the update step could (1) overshoot into really bad parameters or (2) veer off into `inf` or `NaN`.
	- In what scenarios does it occur?
		- Where the loss changes very suddenly
		- Common in RNNs where the network behaves approximately linearly 

	- What is gradient clipping?
		- A technique for avoiding exploding gradients
		- If norm of the gradient is above a threshold (a hyperparameter), then scale down the gradient by $\text{threshold}/g$ before applying the SGD update.

#. Long Short-Term Memory (LSTM)

	- What is an LSTM?
		- A type of RNN meant to solve the vanishing gradient problem (proposed 1997)
		- Unlike vanilla RNN, does not rewrite the hidden layer on every time step

	- What is its structure?
		- At each time step $t$, we have a **hidden state** $h^{(t)}$ and **cell state** $c^{(t)}$ of length $n$.
		- The cell state acts like memory and stores information from previous time steps' hidden states.
		- The LSTM can read, write, and update information in the cell, using three dynamic gates.

	- What are the 3 LSTM gates and their functionality?
		- input gate: controls what new information to write from the current input to the cell
		- forget gate: controls what to write/discard from the previous hidden state to/from the cell
		- output gate: controls what parts of the cell to read from the cell to the hidden state

	- LSTM algorithm
		- Compute new content based on previous hidden state and current input c~
		- Compute elementwise product of $f_t$ and $c^{t-1}$ to mask out information from previous cell state ("forget") and $i_t$ ("input")
		(both are vectors of length n: n in {0, 1} after sigmoid nonlinearity)
		- Compute new hidden state from elementwise product of $o_t$ and $tanh(c^{t})$
	
		- hidden states are like "outputs" of RNN; cell states are like internal memory not accessible to the outside

	- LSTM vs. vanishing gradients
		- The forget gate can be set to remember everything on every time step, and preserve all information indefinitely
		- It's harder for a vanilla RNN to preserve information as-is from step to step
		- Does not guarantee a solution to vanishing/exploding gradients
		- Forget gate operation **adds** $c_{t-1}$, creating a direct linear connection with $c_t$.
		
	- Use cases
		- Dominant approach 2013-2015 (transformers have dominated since they showed up)
		- handwriting recognition
		- speech recognition
		- machine translation
		- parsing
		- image captioning

#. Gated Recurrent Units (GRU)

	- What is a GRU?
		- A simpler alternative to LSTM (proposed 2014)
		- faster to compute and fewer parameters than LSTM
		- neither consistently outperforms the other
		- On each timestep $t$ we have input $x^{t}$ and hidden state $h^{(t)}$, but no cell state.

	- 2 GRU gates
		- update gate: forget gate + input gate
		- reset gate: which parts of previous hidden states will be used to compute output and which will be discarded
		- unlike attention, creates shortcut connections that retain positional info (rather than averaging)

	- GRU algorithm
		- calculate new hidden state content using reset gate and current input 
		- calculate output using update gate (1-u times previous h + u times new h)
			- unlike LSTM, information to keep from previous state and new information are not separate values; in a GRU, if you have more of one, you have less of the other.
	
	- GRU vs. vanishing gradients
		- if the update gate is set to 0, then the hidden state will be kept the same on every step (i.e., will retain information long-term)

#. Bidirectional RNNs

	- Why and when to use a bidirectional RNN?

		- in a vanilla RNN, at time step $t$, the prediction for step $t+1$ can only take into account additional information from time steps 1 through $(t-1)$.
		- can only use when you have access to the entire input sequence (e.g., can't use them for language modeling)
		- if you have the entire sequence, it can learn more context (e.g. for encoding)

	- architecture
		- two separate RNNs with separate weights:
			- 'forward RNN' encodes text left-to-right (words are conditioned on the words that came before)
			$P(w^{t}) = P(w^{t}|w^{(t-1)},...,w^{1})
			- 'backward RNN' encodes sentence right to left (words are conditioned on words that come after them)
			$P(w^{t}) = P(w^{t}|w^{(t+1)}, ..., w^{V})$
		- concatenate the output hidden states
			$[h^{t};h^{t}]$

#. Multi-layer RNNs

	- Why stack RNNs?
		- Appling RNNs one after another can compute more complex representations
		- Earlier RNNs should compute lower-level features, later ones should compute higher-level features

	- How to stack RNNs?
		- The output hidden states of one RNN layer becomes the input of the next RNN layer
		- use 2-4 layers (not as deep as convolutional networks)
		- more skip connections needed the deeper the network

