package tagger

import (
	"bufio"
	"github.com/fluhus/gostuff/nlp/wordnet"
	"os"
)

func loadSynsetsForClassifierLabels(wn *wordnet.WordNet, path string) (synsets []*wordnet.Synset, err error) {
	file, err := os.Open(path)
	if err != nil {
		// TODO error
		return nil, err
	}
	defer file.Close()

	synsets = []*wordnet.Synset{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		synsetId := scanner.Text()
		synset := wn.Synset[synsetId]
		if synset == nil {
			//todo write warting
			continue
		}
		synsets = append(synsets, synset)
	}

	return synsets, scanner.Err()
}

func loadAncestorLabelsForSynset(wn *wordnet.WordNet, synset *wordnet.Synset) []string {
	hypernyms := []string{} // Synset IDs of hypernyms

	synsetsToSearch := []*wordnet.Synset{synset}

	for len(synsetsToSearch) > 0 {
		// deque next element from synsets
		synset = synsetsToSearch[0]
		synsetsToSearch[0] = nil // Erase element (write zero value) to avoid memory leaks
		synsetsToSearch = synsetsToSearch[1:]

		for _, p := range synset.Pointer {
			if p.Symbol == wordnet.Hypernym {
				synsetId := p.Synset
				hypernymSynset := wn.Synset[synsetId]
				synsetsToSearch = append(synsetsToSearch, hypernymSynset)
				hypernyms = append(hypernyms, hypernymSynset.Word[0])
			}
		}
	}

	return hypernyms
}