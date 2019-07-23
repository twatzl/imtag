package vggPreprocessing

import (
	"fmt"
	"github.com/pkg/errors"
	"image"
	"math"

	tg "github.com/galeone/tfgo"
	tgi "github.com/galeone/tfgo/image"
	log "github.com/sirupsen/logrus"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// based on https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py

const (
	R_MEAN = 123.68
	G_MEAN = 116.78
	B_MEAN = 103.94

	RESIZE_SIDE_MIN = 256
	RESIZE_SIDE_MAX = 512
)

func VGGPreprocessingForEval(image *image.RGBA, output_height, output_width int64, resize_side int32) (*tf.Tensor, error) {
	log.Debugln("VGGPreprocessingForEval")
	s := op.NewScope()

	imageInputTensor, err := tf.NewTensor(image.Pix)
	if err != nil {
		return nil, err
	}

	input := op.Placeholder(s, tf.Uint8)

	// load image as tensor
	rect := image.Bounds()
	height := rect.Max.Y - rect.Min.Y
	width := rect.Max.X - rect.Min.X

	// make tensor that resembles the shape of the image (with alpha channel, because we load from rgba)
	img := op.Reshape(s, op.Cast(s, input, tf.Float), tg.Const(s,[3]int32{int32(height), int32(width), 4}))
	// remove alpha channel
	img = op.Slice(s, img, tg.Const(s, [3]int32{0,0,0}), tg.Const(s, [3]int32{int32(height), int32(width), 3}))

	resizedImage := _aspect_preserving_resize(s, img, resize_side)


	croppedImages, err := _central_crop(s, []*tf.Output{resizedImage}, output_height, output_width)
	if err != nil {
		return nil, err
	}

	//image.set_shape([output_height, output_width, 3])
	//image = tf.to_float(image)
	outputImage, err := _mean_image_subtraction(s, croppedImages[0], []float32{R_MEAN, G_MEAN, B_MEAN})
	if err != nil {
		return nil, err
	}

	outputImageBatch := imageToBatch(s, outputImage)

	log.Println(outputImage.Shape())

	graph, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	log.Infoln("preprocessing graph built")

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	log.Infoln("starting preprocessing session")

	defer session.Close()
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{input: imageInputTensor},
		[]tf.Output{*outputImageBatch},
		nil)
	if err != nil {
		return nil, err
	}

	log.Infoln("preprocessing finished")

	return output[0], nil
}

func imageToBatch(s *op.Scope, image *tf.Output) *tf.Output {
	s = s.SubScope("imageToBatch")
	outputImageBatch := op.ExpandDims(s,
		// Use pixel values
		*image,
		op.Const(s, int32(0)))
	return &outputImageBatch
}

func _aspect_preserving_resize(s *op.Scope, image tf.Output, smallest_side int32) (*tf.Output) {
	log.Debugln("_aspect_preserving_resize")
	//"""Resize images preserving the original aspect ratio.
	//Args:
	//image: A 3-D image `Tensor`.
	//smallest_side: A python integer or scalar `Tensor` indicating the size of
	//the smallest side after resize.
	//Returns:
	//resized_image: A 3-D tensor containing the resized image.
	//"""

	shape := image.Shape()
	height := shape.Size(0) // was 0
	width := shape.Size(1) // was 1
	new_height, new_width := _smallest_size_at_least(float64(height), float64(width), float64(smallest_side))

	// add batch dimension to image
	imageBatch := op.ExpandDims(s, image, tg.Const(s,  int32(0)))

	tgiImage := tgi.NewImage(s, imageBatch)
	resized_image := tgiImage.ResizeBilinear(
		tgi.Size{float32(new_height), float32(new_width)}).Output
	resized_image = op.Squeeze(s, resized_image)
	//resized_image.set_shape([None, None, 3])

	return &resized_image
}

func _central_crop(s *op.Scope, image_list []*tf.Output, crop_height, crop_width int64) ([]*tf.Output, error) {
	log.Debugln("_central_crop")
	s = s.SubScope("centralcrop")
	//"""Performs central crops of the given image list.
	//Args:
	//image_list: a list of image tensors of the same dimension but possibly
	//varying channel.
	//crop_height: the height of the image following the crop.
	//crop_width: the width of the image following the crop.
	//Returns:
	//the list of cropped images.
	//"""
	outputs := []*tf.Output{}
	for _, image := range image_list {
		image_height := image.Shape().Size(0)
		image_width := image.Shape().Size(1)

		offset_height := (image_height - crop_height) / 2
		offset_width := (image_width - crop_width) / 2

		croppedImage, err := _crop(s, *image, offset_height, offset_width, crop_height, crop_width)
		if err != nil {
			log.Errorf("Could not crop image: %s\n", err.Error())
			return nil, err
		}

		outputs = append(outputs, croppedImage)
	}
	return outputs, nil
}

func _mean_image_subtraction(s *op.Scope, image *tf.Output, means []float32) (*tf.Output, error) {
	const methodName = "_mean_image_subtraction"
	log.Debugln(methodName)
	s = s.SubScope("meanimagesubtraction")
	//"""Subtracts the given means from each image channel.
	//For example:
	//means = [123.68, 116.779, 103.939]
	//image = _mean_image_subtraction(image, means)
	//Note that the rank of `image` must be known.
	//Args:
	//image: a tensor of size [height, width, C].
	//means: a C-vector of values to subtract from each channel.
	//Returns:
	//the centered image.
	//Raises:
	//ValueError: If the rank of `image` is unknown, if `image` has a rank other
	//than three or if the number of channels in `image` doesn't match the
	//number of values in `means`.
	//"""

	if image.Shape().NumDimensions() != 3 {
		const message = "Input must be of size [height, width, C > 0]"
		fmt.Println(message)
		return nil, errors.New(message)
	}
	num_channels := image.Shape().Size(image.Shape().NumDimensions()-1)
	if len(means) != int(num_channels) {
		const message = "len(means) must match the number of channels"
		fmt.Println(message)
		return nil, errors.New(message)
	}

	//image_o := tg.Const(s, *image)
	channels := op.Split(s, tg.Const(s, int32(2)), *image, int64(num_channels))
	log.Println(s.Err())
	for i, _ := range channels {
		scopeName := fmt.Sprintf("channel%d", i)
		op.Sub(s.SubScope(scopeName), channels[i], tg.Const(s, means[i]))
	}
	resultImage := op.Concat(s, tg.Const(s, int32(2)), channels)
	return &resultImage, nil
}

func _smallest_size_at_least(height, width, smallest_side float64) (new_height, new_width int) {
	//"""Computes new shape with the smallest side equal to `smallest_side`.
	//Computes new shape with the smallest side equal to `smallest_side` while
	//preserving the original aspect ratio.
	//Args:
	//height: an int32 scalar tensor indicating the current height.
	//width: an int32 scalar tensor indicating the current width.
	//smallest_side: A python integer or scalar `Tensor` indicating the size of
	//the smallest side after resize.
	//Returns:
	//new_height: an int32 scalar tensor indicating the new height.
	//new_width: and int32 scalar tensor indicating the new width.
	//"""

	scale := 0.0
	if height > width {
		scale = smallest_side / width
	} else {
		scale = smallest_side / height
	}

	new_height = int(math.Round(height * scale))
	new_width = int(math.Round(width * scale))
	return
}

func _crop(s *op.Scope, image tf.Output, offset_height, offset_width, crop_height, crop_width int64) (*tf.Output, error) {
	log.Debugln("_crop")
	//"""Crops the given image using the provided offsets and sizes.
	//Note that the method doesn't assume we know the input image size but it does
	//assume we know the input image rank.
	//Args:
	//image: an image of shape [height, width, channels].
	//offset_height: a scalar tensor indicating the height offset.
	//offset_width: a scalar tensor indicating the width offset.
	//crop_height: the height of the cropped image.
	//crop_width: the width of the cropped image.
	//Returns:
	//the cropped (and resized) image.
	//Raises:
	//InvalidArgumentError: if the rank is not 3 or if the image dimensions are
	//less than the crop size.
	//"""
	original_shape := image.Shape()

	if image.Shape().NumDimensions() != 3 {
		const message = "Rank of image must be equal to 3."
		fmt.Println(message)
		return nil, errors.New(message)
	}

	//cropped_shape := tf.MakeShape(crop_height, crop_width, original_shape.Size(2))

	if original_shape.Size(0) <= crop_height && original_shape.Size(1) <= crop_width {
		const message = "Crop size greater than the image size."
		fmt.Println(message)
		return nil, errors.New(message)
	}

	//offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
	offsets := tg.Const(s, []int64{int64(offset_height), int64(offset_width), 0})
	log.Infoln(offsets.Shape())

	// Use tf.slice instead of crop_to_bounding box as it accepts tensors to
	// define the crop size.
	cropped_shape_o := tg.Const(s, []int64{crop_height, crop_width, original_shape.Size(2)})
	cropped_shape_rank := cropped_shape_o.Shape()
	log.Infoln(cropped_shape_rank)
	slicedImage := op.Slice(s, image, offsets, cropped_shape_o)
	fmt.Println(s.Err())

	// TODO: reshape should be done in case image is smaller or has rank other than 3
	//reshapedImage := op.Reshape(s, slicedImage, cropped_shape_o)
	return &slicedImage, nil
}
