package tagger

import (
	"github.com/twatzl/imtag/tagger/image"
	"github.com/twatzl/imtag/tagger/word2vec"
)

func embedImageFlat(word2vec word2vec.Word2Vec, image image.Image) []float32 {
	tags := image.GetTags()
	var sumProbabilities float32 = 0.0 // this corresponds to Z in the paper
	embeddedVector := make([]float32, word2vec.GetDim())

	for _,tag := range tags {
		label := tag.GetLabel()
		confidence := tag.GetConfidence()
		vec := word2vec.Word2Vec(label)

		for idx, val := range vec {
			embeddedVector[idx] += val * confidence
		}

		sumProbabilities += confidence
	}

	sumProbabilities = 1/sumProbabilities

	for idx, val := range embeddedVector {
		embeddedVector[idx] = val * sumProbabilities
	}

	return embeddedVector
}

func embedImageHierarchical() {

}
