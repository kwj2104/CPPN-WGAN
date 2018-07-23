# CPPN

Generate abstract images with vanilla Compositional Pattern Producing Networks

Generate images by running:

```
python CPPN/cppn.py
```

## Examples

<p align='center'>
<img src="generated_img/sample_bw.png" width='400'/>
<img src="generated_img/sample_color.png" width='400'/>
</p>

# CPPN-Gan

Generate abstract images with Vanilla GANs

Code is pretty unstable. Feel free to experiment with what is in the CPPN-GAN-OLD folder.

## Examples:

<p align='center'>
<img src="generated_img/sample1 inv.png" width='300'/>
<img src="generated_img/sample2 inv.png" width='300'/>
</p>

<p align='center'>
<img src="generated_img/cppn-gan interpolation.gif" width='400'/>
</p>





# CPPN-WGAN

Generate images with CPPNs and GANs.

See the CPPN+GAN blog post here, and then the WGAN variation here.
Vanilla CPPN/CPPN-GAN repo is here, though the CPPN-GAN does not quite work properly (but it does generate very interesting pictures).  

## Working with datasets (CIFAR10 and CASIA)

**CIFAR-10**: Download CIFAR-10 (Python version) at https://www.cs.toronto.edu/~kriz/cifar.html and place files into `cifar-10-batches-py/`

**CASIA**: CASIA is a bit cumbersome to use. Currently `tflib/casia.py` create a custom PyTorch dataloader to download the data, and then transform them into squares (since the raw data comes in various dimensions). The file makes use of the pycasia library. CASIA will automatically download; however the file may take a long time due to being hosted on Chinese servers. After the file is downloaded `pycasia.load_dataset()` also takes forever to load all of the `competition-gnt` file. The whole file is broken up into ~60 chunks, so I recommend first taking one chunk and loading that (move all the other files away from the data folder temporarily), before trying the entire dataset, to save time.

## Training the models

You can train by running e.g. :
```
python gan_cppn_cifar10.py
```
Models take a few hours to train even on a GPU. Repo comes with pre-trained models of the generator.

## Generating Images

Run `Interpolator.py` to generate images and gifs. In the main method you can edit the dimensions and samples you wish to create.

## Examples

#### Large Generated Images:
<p align='center'>
<img src="generated_img/large_sample_cifar10.png" width='400'/>
<p align='center'>
<img src="generated_img/large_sample_casia.png" width='400'/>
</p>


#### Interpolations:
<p align='center'>
<img src="generated_img/cifar10_movie.gif" width='400'/>
<img src="generated_img/movie_CASIA.gif" width='400'/>
</p>

#### Trained Samples:
<p align='center'>
<img src="generated_img/samples_mnist_2.png" width='200'/>
<img src="generated_img/samples_cifar10.jpg" width='200'/>
<img src="generated_img/samples_63299.png" width='200'/>
</p>

## Special Thanks
Many thanks to:
- @Hardmaru for inspiring this work
- @caogang for their WGAN-GP implementation which I heavily relied on for my code.
- @lucaskjaero for their pycasia library
