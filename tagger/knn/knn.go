package knn

import (
	"github.com/twatzl/imtag/tagger/label"
	"math"
	"sort"
)

type knnMapping struct {
	distance float32
	labelIndex int
}

type KnnDistFunc func([]float32, []float32) float32

/**
 * KnnSearch will search the vectorspace (given by all labels which have been embedded in the vectorspace) and
 * return the k labels with the smallest distance to the embedded image.
 * This knn is implemented in a naive way, but can be improved by using either a more sophisticated algorithm
 * or by using goroutines to utilize multicore systems.
 */
func KnnSearch(vectorspace []label.Label, searchTarget [][]float32, k int, distanceFunction KnnDistFunc) [][]label.Label {
	var distances = make([][]knnMapping, len(searchTarget))

	// calculate distances to all elements in vectorspace
	for labelIdx, l := range vectorspace {
		for targetIdx, target := range searchTarget {
			distances[targetIdx][labelIdx] = knnMapping{
				distanceFunction(l.GetVector(), target),
				labelIdx,
			}
		}
	}

	result := make([][]label.Label, len(searchTarget))

	// sort result, take first k elements and map them back to labels
	for targetIdx := range searchTarget {
		// sort results
		sort.Slice(distances[targetIdx], func(i, j int) bool {
			return  distances[targetIdx][i].distance < distances[targetIdx][j].distance
		})
		distances[targetIdx] = distances[targetIdx][:k]

		// take min of k or length
		x := len(distances[targetIdx])
		if k < x {
			x = k
		}

		result[targetIdx] = make([]label.Label, x)

		// mapping results to labels
		for i := 0; i < x; i++ {
			result[targetIdx][i] = vectorspace[distances[targetIdx][i].labelIndex]
		}
	}

	return result
}

/**
 * Chi2Dist computes the chi square distance between vectors a and b.
 * The chi square distance is given as chi2(x,y) = sum( (xi - yi)^2 / (xi+yi))/2
 */
func Chi2Dist(x []float32, y []float32) float32 {
	var result float32
	for i := range x {
		result += (x[i] - y[i]) * (x[i] - y[i]) / (x[i] + y[i])
	}

	return result / 2
}

/**
 * cosDist(x,y) = 1 - (x * y)/(norm(x) * norm(y)
 */
func CosDist(x []float32, y []float32) float32 {
	var dProd float32
	var normX float32
	var normY float32
	for i := range x {
		dProd += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}

	normX = float32(math.Sqrt(float64(normX)))
	normY = float32(math.Sqrt(float64(normY)))

	return 1 - (dProd / (normX * normY))
}

/**
 * euDist(x,y) = sqrt(sum((xi - yi)^2))
 */
func EuclideanDist(x []float32, y []float32) float32 {
	var dist float32

	for i := range x {
		dist += (x[i] - y[i]) * (x[i] - y[i])
	}

	return float32(math.Sqrt(float64(dist)))
}

/**
 * manDist(x,y) = sum(abs(xi-yi))
 */
func ManhattanDist(x []float32, y [] float32) float32 {
	var dist float32

	for i := range x {
		dist += float32(math.Abs(float64(x[i] - y[i])))
	}
	return dist
}
