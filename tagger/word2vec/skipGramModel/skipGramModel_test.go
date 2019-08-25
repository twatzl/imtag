package skipGramModel

import (
	"testing"
)

const base_path = "../../../data/models/flickr4m.word2vec/word2vec/tagvec500"


func Test_skipGramModel_integration(t *testing.T) {
	w := skipGramModel{}
	w.init(base_path)

	vec := w.Word2Vec("test")
	println(vec)

	println("dimensions: ", w.GetDim())
	println("number of words: ", len(w.name2Index))
}