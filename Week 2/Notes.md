## Thresholding

A lot of computer vision work involves simplifying the visual data you are presented
with. A useful tool for this is thresholding, where we convert an image to black
or white based on a threshold value. Say we have a greyscale image we want to threshold
with a value of 125, every pixel with a value below 125 will be converted to white and
every pixel above this will be converted to black.

## Color Filtering

Color filtering involves processing an image to only filter out a specific color.
This uses the bitwise operations we learnt in week one to create the filter. In
this tutorial we use the HSV color space to represent the color we are trying to
aim for. Here is a visualisation of HSV color space, ![alt text](https://en.wikipedia.org/wiki/HSL_and_HSV#/media/File:HSV_color_solid_cylinder_saturation_gray.png "HSV").

In the code, first we convert the frames to HSV, then create the upper and lower
HSV values for the color we are trying to filter, then we create the mask that
only shows white pixels where they are within the range we specified. Next we
perform a bitwise AND between the frame and the mask to get our result.

## Blurring and Smoothing

To further simplify our filtered image, we can reduce the noise that is occurring
by applying a blur. We have many different types of blur algorithms at our
disposal so picking the best one really depends on what works best in each case.
The first one averages the pixels with each block, where you can specify the
block size. There's also Gaussian blur, median blur and bilateral filter which
each give different results.

## Morphological Transformations

The first two Transformations we cover are called Erosion and Dilation. Both use
a sliding window called a kernel, whos size you can specify, in this case it's
5x5 pixels. In the case of erosion, if all the pixels within the window are white
then the entire window is white, otherwise the entire window becomes black.
This may eliminate some white noise.

In Dilation, if the entire window isn't completely black then it get converted
to white. Think of this as pushing the white out until it can't, whereas erosion
is trying to erode away as much white as it can.


Lastly, we have opening and closing. Opening is used to remove false positives,
while closing is to remove false negatives.
