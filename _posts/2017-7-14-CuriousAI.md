---
layout: post
title: <center>The curious case of curious AI</center>
---
<center>
<h2 style="text-align:center;">A take on Curiosity-driven Exploration by Self-supervised Prediction.</h2>
<blockquote>
<p style="text-align:center;">This article attempts to give the reader (you) a brief overview of the excellent research paper on 'curious AI' written by <span class="word">Deepak</span> <span class="word">Pathak, </span><span class="word">Pulkit</span> <span class="word">Agrawal, </span><span class="word">Alexei</span> <span class="word">A.</span> <span class="word">Efros, </span><span class="word">Trevor</span> <span class="word">Darrell.</span></p>
</blockquote>
<p style="text-align:center;">In real life, rewards extrinsic to an agent is extremely sparse. Curiosity acts as in 'intrinsic reward signal'.
The way curiosity has been expressed in the paper is the error in agent's ability to predict the consequence of its own actions in a visual feature space learned by a 'self-supervised inverse dynamics model'.</p>
<p style="text-align:center;">The point is to encourage the agent to reach 'novel' states and help it get better at reducing the error in predicting consequence of its actions.
Measuring novelty requires a statistical model. So does measuring prediction error. This is difficult due to stochasticity in the environment.</p>

<blockquote>
<h3 style="text-align:center;">A bit on A3C</h3>
<p style="text-align:left;">You will be encountering A3C a lot in this article. Put simply, it made DQN obsolete.
Unlike DQN, where a single agent represented by a single neural network
interacts with a single environment, A3C utilizes multiple incarnations
of the above in order to learn more efficiently. In A3C there is a global
network, and multiple worker agents which each have their own set of
network parameters.
<a href="https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2" target="_blank" rel="noopener noreferrer">More on this</a></p>
</blockquote>
<p style="text-align:left;">The key in the paper was to only predict the consequences of its own actions. A transformation of the sensory input to a feature space was performed. The feature space was learned using self-supervision. This was done by predicting the action that would lead to a transformation from the current state to the next state. (A proxy<strong> inverse dynamics task</strong> to be fancy).
A <strong>forward dynamics model</strong> predicting the feature representation of the next state, given current state and the action was also used.</p>
<p style="text-align:left;">In the paper, comparison between an A3C agent with and without the curiosity signal on a 3D navigation task is performed. (In Viz-Doom)</p>

<h2 style="text-align:center;">The 'Design'</h2>
<img class="alignnone size-full wp-image-188" src="https://quirkyai.files.wordpress.com/2017/05/curiosity-system.png" alt="Curiosity system.png" />

There were two subsystems:
<ul>
	<li>A reward generator: Outputs a curiosity-driven intrinsic reward signal, and  an extrinsic reward system.</li>
	<li>A policy that outputs a sequence of actions to maximize that reward signal.</li>
</ul>
Intrinsic curiosity reward at time t: $latex r_{t}^{i}$, extrinsic reward: $latex r_{t}^{e}$.
Policy sub-system to maximize r = $latex r_{t}^{i}$ + $latex r_{t}^{e}$.

The policy $latex \pi (s_{t};\Theta_{p})$ is represented by a DNN with parameters $latex \Theta_{p}$. At state $latex s_{t}$, it executes action $latex a_{t} ~ \pi (s_{t};\Theta_{p})$. $latex \Theta_{p}$ is optimized to maximize the expected sum of rewards.

<strong>
$latex max_{\theta_{p}} \mathbb{E}_{\pi (s_{t};\Theta_{p})}\left [ \sum _{t}r_{t}\right ]$
</strong>

The problem with giving prediction error as a curiosity reward is that the agent is cannot distinguish between features that cannot be modeled and ones that can. Thus the agent can fall into an 'artificial curiosity trap' and stall. A good feature space should not model things that are out of agent's control and does not affect it.

Such a feature space can be learned by training a deep neural network with two sub modules:
<ul>
	<li>Raw state $latex s_{t}$ to feature vector $latex \Phi(s_{t})$ encoder</li>
	<li>Takes features encoding $latex \Phi(s_{t})$, $latex \Phi(s_{t+1})$ and predicts action $latex a_{t}$ taken by agent to move from $latex s_{t}$ to $latex s_{t+1}$</li>
</ul>
<h3>Inverse dynamics model</h3>
This prediction of action gives a learning function g:
[latex]\hat{a}_{t} = g\left ( s_{t}, s_{t+1}; \theta_l \right )[/latex]
Here $latex \hat{a}_{t}$ is the predicted estimate of the action, and the neural network parameters $latex \theta_{l}$ are trained to optimize:
[latex]min_{\theta_{I}}L_{I}\left ( \hat{a}_{t}, a_{t} \right )[/latex]
What you see above is the loss function. The learned function g is also known as the<strong> inverse dynamics model. </strong>
<h3>Forward dynamics model</h3>
We have another model that takes as inputs $latex a_{t}$ and $latex \Phi(s_{t})$ and predicts feature encoding of the state at time step t+1.

<img class=" size-full wp-image-221 aligncenter" src="https://quirkyai.files.wordpress.com/2017/05/2017-05-30-00_30_43-icml17.png" alt="2017-05-30 00_30_43-icml17.png" />

Here [latex]\hat\Phi(s_{t+1})[/latex] is the predicted estimate of [latex]\Phi(s_{t+1})[/latex] and the neural network parameters [latex]\theta _{F}[/latex] are optimized by optimizing the following loss function:
<img class=" size-full wp-image-222 aligncenter" src="https://quirkyai.files.wordpress.com/2017/05/2017-05-30-00_33_00-icml17.png" alt="2017-05-30 00_33_00-icml17.png" />

The intrinsic reward is computed as the product of $latex \eta$ and $latex L_{F}$. $latex \eta$ is a scaling factor. We jointly optimize the <strong>forward and inverse </strong><strong>dynamics loss</strong>.

<hr />



<hr />

<h2 style="text-align:center;">The Architecture</h2>
The input RGB inputs are converted to grey-scale 42 by 42 size images. To model <strong>temporal dependencies, </strong>each state representation is concatenated with three previous frames.

Asynchronous training protocol: 20 workers using SGD.
<h3 style="text-align:center;">A3C architecture:</h3>
Input state => four convolutional layers with 32 filters each, kernel size 3x3; stride 2; padding 1. ELU used after each convolutional layer. Output of last convolution fed into a LSTM with 256 units. ==> 2 seperate fully connected layers to predict the value function and action from LSTM feature representation.
<h3 style="text-align:center;">ICM architecture:</h3>
<strong>Inverse model architecture: </strong>Maps input state to feature vector using 4 conv. layers, with 32 filters each, kernel size 3x3; stride 2; padding 1. ELU is used. Dimensionality of feature vector: 288. $latex \Phi(s_{t})$ & $latex \Phi(s_{t+1})$ are concatenated to a single feature vector and passed to a fully connected layer of 256 units with fully connected layer with 4 units to predict one of the 4 possible actions.

<strong>Forward model architecture: </strong>Feature vector concatenated with action $latex a_{t}$, passing it to a sequence of two fully connected layers with 256 & 288 units respectively.
<h2>Results:</h2>
<img class="alignnone size-full wp-image-230" src="https://quirkyai.files.wordpress.com/2017/05/result-1.png" alt="result 1.png" />
Rewards provided were made more sparse, which cause performance to deteriorate.
ICM (pixels) + A3C consistently performed worse than ICM + A3C because any change in texture style could render the method ineffective.

<strong>Mario with no rewards:</strong>

After training the algorithm on level 1, three methods of evaluating its robustness could be considered: Playing "as is" on the next levels, Fine tune the policies a bit,  Fine tune with extrinsic reward.

<img class=" size-full wp-image-232 aligncenter" src="https://quirkyai.files.wordpress.com/2017/05/result-2.png" alt="result 2.png" />

Running "as is" on level 2 after level 1 proved to be unsuccessful. This was because level 2 is visually very distinct from level 1 in terms  of lighting. Thus after a little bit of fine tuning,  Very promising results were seen.

Training the algorithm from scratch on level 2 found less effective results. Could be due to increased difficulty of level 2.

Level 3 with fine tuning or from scratch did not see very good results. This is because level 3 is very hard - The agent hits a 'curiosity blockade', as it stops receiving curiosity rewards, thus the policy stops working. This has been referred to as analogous to 'boredom' in the paper. Halting exploration upon no reward from effort.

<hr />

NOTE: This entire article has been published for people who wanted a slightly concise version of the paper, also because i really enjoyed the paper and would like people to see it one way or another. None of the work above is done by me. Hope you enjoyed it nevertheless!

<hr />

 
