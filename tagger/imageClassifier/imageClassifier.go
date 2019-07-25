// The package imageClassifier contains the common interface for
// image classifiers, as well as implementations of certain classifiers.
// The idea is that for each new classifier a new subpackage is created in this
// package and all of the classifiers satisfy the common interface.
package imageClassifier

import (
	"github.com/twatzl/EmbeddingImageTagger/tagger/image"
	"github.com/twatzl/EmbeddingImageTagger/tagger/tag"
)

type ImageClassifier interface {
	// ClassifyImages takes a batch of images as input and returns a slice of tags for each
	// of the images.
	// The implementations will probably run some kind of machine learning model to
	// estimate the likelyhood of different tags.
	// ClassifyImages returns the whole vector of tags.
	ClassifyImages(image []image.Image) [][]tag.Tag

	// ClassifyImages top k works the same as ClassifyImages, except that it
	// takes an additional parameter k. The classifier will then only return the
	// k tags with the highest priority.
	// This function is provided by the classifiers, because some of the ML
	// frameworks like TensorFlow already provide an efficient way to get the
	// top k results.
	ClassifyImagesTopK(image []image.Image, k int) [][]tag.Tag
}
