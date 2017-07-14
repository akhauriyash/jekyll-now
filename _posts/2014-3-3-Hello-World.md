---
layout: post
title: CODING YOUR FIRST NEURAL NETWORK FROM SCRATCH
---
<p style="text-align: center;">The key in my opinion, to coding your first neural network is to sidestep the landmine that is using Tensorflow, and using nothing but Python, and numpy.</p>
<p style="text-align: center;">The code given below 'learns' that the output 'y' will be 0 if the input is an odd number in binary representation, and 1 if the input is an even number.</p>


[caption id="attachment_101" align="aligncenter" width="591"]<img class=" size-full wp-image-101 aligncenter" src="https://quirkyai.files.wordpress.com/2017/05/code1.png" alt="code1.png" width="591" height="605" /> The code. (Image uploaded as i do not expect anyone to copy paste this.)[/caption]
<blockquote>
<p style="text-align: center;">We begin the journey by understanding the sigmoid function:</p>
<p style="text-align: center;"><img class="" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/34ef0bf30a2cd7e90e87221cfc796dcf4f265d56" alt="{\displaystyle S(x)={\frac {1}{1+e^{-x}}}.}" width="101" height="82" /></p>
</blockquote>
<p style="text-align: center;"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png" /></p>
<p style="text-align: center;">This function is very widely used in neural networks. The purpose is to bring non-linearity into the model.</p>
<p style="text-align: center;">Another reason for this choice is that the derivative of the Sigmoid function:</p>


[caption id="attachment_44" align="aligncenter" width="198"]<img class="alignnone size-full wp-image-44" src="https://quirkyai.files.wordpress.com/2017/05/sigmoid-deriv.png" alt="sigmoid deriv.png" width="198" height="61" /> eqn(1)[/caption]
<p style="text-align: center;">This means that one can easily create the following function:</p>
<p style="text-align: center;"><img class="alignnone  wp-image-48" src="https://quirkyai.files.wordpress.com/2017/05/sigderivfunc.png" alt="sigderivfunc.png" width="227" height="55" /></p>
<p style="text-align: center;">This would return the derivative, as proven by eqn (1)</p>

<h2 style="text-align: center;">Weighing your errors</h2>
<p style="text-align: center;"><strong>The derivative is useful because: </strong>whenever the neural network makes a 'prediction', it is important that the error in is adjusted correctly.</p>


[caption id="attachment_62" align="aligncenter" width="467"]<img class="alignnone  wp-image-62" src="https://quirkyai.files.wordpress.com/2017/05/sigmoid-deriv-explanation.png" alt="sigmoid deriv explanation.png" width="467" height="352" /> <strong>Weighted errors</strong>[/caption]
<p style="text-align: center;">The concept of <strong>weighted error </strong>comes into play here. Penalizing what the neural network is less confident about heavily, and penalizing its most confident outputs the least. Which can be articulated in the following statement/code best:</p>
<p style="text-align: center;">Error = y - prediction
prediction_delta = Error*sigmoid_to_deriv(prediction)</p>
<p style="text-align: center;">The red boxes signify what might qualify as high confidence predictions. This implies that the multiplying the error with the derivative at that prediction leads to a small prediction_delta. Which implies lesser weightage to the error.</p>

<h2 style="text-align: center;">Synapses</h2>
<p style="text-align: center;">It is extremely important that anyone attempting to code a neural network assimilates the concept of synapses.
Our first priority, after obviously the inputs is creating synapses. Synapses can be imagined to be <strong>weight matrices </strong>for the neural network.  For purposes pertaining to the task ahead of us, we shall load random numbers into the matrix, which will act as weights initially, and we will update it with every iteration.</p>
<p style="text-align: center;">I hope everyone is familiar with matrix multiplication here. If not, I'll give you the very basic, and a recommendation to look up some Khan Academy videos.</p>
<p style="text-align: center;"><img class="" src="https://quirkyai.files.wordpress.com/2017/05/3efde-matrix_multi.png" width="333" height="121" /></p>
<p style="text-align: center;">If we are to create a neural network with many hidden layers, this becomes essential.</p>
<p style="text-align: center;">After inspecting the code on top, as well as the picture below, you will have a perfect understanding of the process underlying neural networks.</p>

<blockquote>
<p style="text-align: center;"><strong>Make sure you spend as much time as you need understanding this. Refer to the code simultaneously as well.</strong></p>
</blockquote>
<p style="text-align: center;"><img class="alignnone size-full wp-image-134" src="https://quirkyai.files.wordpress.com/2017/05/explanationtry2.jpg" alt="explanationtry2" width="2000" height="1500" /></p>


<hr />

<h2 style="text-align: center;">Backpropogation</h2>
<img class="aligncenter size-full wp-image-133" src="https://quirkyai.files.wordpress.com/2017/05/2017-05-29-15_57_55-spyder-python-3.png" alt="2017-05-29 15_57_55-Spyder (Python 3.png" width="336" height="137" />

Looking at line 22 might make the concept of error obvious. However, can you guess what is happening on line 24?
What we do here is simply establish a metric of how each synapse in the network contributed to the overall error in each iteration.
<p style="text-align: center;">For a concrete understanding of this concept, i'd suggest an exceptional source: <a href="http://cs231n.github.io/optimization-2/" target="_blank" rel="noopener noreferrer">Optimization: CS231n</a></p>

<h2 style="text-align: center;">Alphas &
Gradient Descent</h2>
Say I'm jumping around in a bowl shaped like a parabola. If the length i can jump is too long, and my purpose is to get to the lowest point, I might never get there, as i may keep overshooting the bottom. But say the bowl was not parabola shaped, but irregular with many local minimas, and say I jump extremely small distances, then the probability that I will get caught in a local minima for several iterations is extremely high.

[caption id="attachment_138" align="alignnone" width="743"]<img class="alignnone size-full wp-image-138" src="https://quirkyai.files.wordpress.com/2017/05/graddesc.png" alt="graddesc.png" width="743" height="396" /> Source: http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html[/caption]

This impacts our ability to train in a massive manner.

So, an alpha parameter is introduced, which reduces the size of each 'leap'. This is crucial to training speed, but also needs to be selected carefully.

This is just one of the several optimization techniques, among SGD, AdaGrad, ADAM, to name a few.

 
![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.
