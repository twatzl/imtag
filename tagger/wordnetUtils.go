package tagger

import (
	"bufio"
	"github.com/fluhus/gostuff/nlp/wordnet"
	"os"
	"sort"
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
	// map because wordnet allows duplicates
	hypernyms := map[string]interface{}{}

	synsetsToSearch := []*wordnet.Synset{synset}

	for len(synsetsToSearch) > 0 {
		// deque next element from synsets
		synset = synsetsToSearch[0]
		synsetsToSearch = synsetsToSearch[1:]

		for _, p := range synset.Pointer {
			if p.Symbol == wordnet.Hypernym {
				synsetId := p.Synset
				hypernymSynset := wn.Synset[synsetId]
				synsetsToSearch = append(synsetsToSearch, hypernymSynset)

				word := hypernymSynset.Word[0]
				if p.Target != -1 {
					word = hypernymSynset.Word[p.Target]
				}

				hypernyms[word] = ""
			}
		}
	}

	hypernymSlice := make([]string, 0)

	for  k, _ := range hypernyms {
		hypernymSlice = append(hypernymSlice, k)
	}

	return hypernymSlice
}

type LabelHierarchy struct {
	HierarchyLevel int
	Label string
}

func loadAncestorHierarchyForSynset(wn *wordnet.WordNet, synset *wordnet.Synset) []LabelHierarchy {
	// map because there might be different paths to a label and we want to only mark the
	// shortest one
	hypernymDict := map[string]LabelHierarchy{}

	synsetsToSearch := []struct {
		*wordnet.Synset;
		int
	}{{synset, 0}}
	var level int

	for len(synsetsToSearch) > 0 {
		// deque next element from synsets
		synset = synsetsToSearch[0].Synset
		level = synsetsToSearch[0].int + 1
		synsetsToSearch = synsetsToSearch[1:]

		for _, p := range synset.Pointer {
			if p.Symbol == wordnet.Hypernym {
				synsetId := p.Synset
				hypernymSynset := wn.Synset[synsetId]
				synsetsToSearch = append(synsetsToSearch, struct {
					*wordnet.Synset;
					int
				}{hypernymSynset, level})

				words := hypernymSynset.Word
				if p.Target != -1 {
					words = []string{hypernymSynset.Word[p.Target]}
				}

				for _, word := range words {
					if val, hasKey := hypernymDict[word]; hasKey && val.HierarchyLevel < level {
						continue
					}

					hypernymDict[word] = LabelHierarchy{level, word}
				}
			}
		}
	}

	hypernyms := []LabelHierarchy{}
	for _, val := range hypernymDict {
		hypernyms = append(hypernyms, val)
	}

	// map has random order so we need to sort
	sort.SliceStable(hypernyms, func(i, j int) bool {
		return hypernyms[i].HierarchyLevel < hypernyms[j].HierarchyLevel
	})

	return hypernyms
}