package word2vec

type Word2Vec interface {
	Word2Vec(word string) []float32
	GetDim() int
}
