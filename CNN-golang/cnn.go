package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"math/rand"
	"os"
	"time"
)

// Convolutional Layer
type ConvLayer struct {
	Filters [][][][]float64 // [numFilters][inChannels][kernelH][kernelW]
	Biases  []float64
	Stride  int
	Padding int
}

func (cl *ConvLayer) Forward(input [][][]float64) [][][]float64 {
	numFilters := len(cl.Filters)
	inChannels := len(input)
	kernelH := len(cl.Filters[0][0])
	kernelW := len(cl.Filters[0][0][0])

	inputH := len(input[0])
	inputW := len(input[0][0])

	// Calculate output dimensions
	outH := (inputH + 2*cl.Padding - kernelH)/cl.Stride + 1
	outW := (inputW + 2*cl.Padding - kernelW)/cl.Stride + 1

	output := make([][][]float64, numFilters)
	for f := range output {
		output[f] = make([][]float64, outH)
		for i := range output[f] {
			output[f][i] = make([]float64, outW)
		}
	}

	// Apply padding to each input channel
	padded := make([][][]float64, inChannels)
	for c := range padded {
		padded[c] = pad2D(input[c], cl.Padding)
	}

	// Perform convolution
	for f := 0; f < numFilters; f++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				var sum float64
				for c := 0; c < inChannels; c++ {
					yStart := y * cl.Stride
					xStart := x * cl.Stride
					for ky := 0; ky < kernelH; ky++ {
						for kx := 0; kx < kernelW; kx++ {
							sum += padded[c][yStart+ky][xStart+kx] * cl.Filters[f][c][ky][kx]
						}
					}
				}
				output[f][y][x] = relu(sum + cl.Biases[f])
			}
		}
	}
	return output
}

// Max Pooling Layer
type MaxPoolLayer struct {
	PoolSize int
	Stride   int
}

func (mpl *MaxPoolLayer) Forward(input [][][]float64) [][][]float64 {
	channels := len(input)
	h := len(input[0])
	w := len(input[0][0])

	outH := (h - mpl.PoolSize)/mpl.Stride + 1
	outW := (w - mpl.PoolSize)/mpl.Stride + 1

	output := make([][][]float64, channels)
	for c := range output {
		output[c] = make([][]float64, outH)
		for i := range output[c] {
			output[c][i] = make([]float64, outW)
		}
	}

	for c := 0; c < channels; c++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				yStart := y * mpl.Stride
				xStart := x * mpl.Stride
				maxVal := input[c][yStart][xStart]
				for ky := 0; ky < mpl.PoolSize; ky++ {
					for kx := 0; kx < mpl.PoolSize; kx++ {
						val := input[c][yStart+ky][xStart+kx]
						if val > maxVal {
							maxVal = val
						}
					}
				}
				output[c][y][x] = maxVal
			}
		}
	}
	return output
}

// Flatten Layer
type FlattenLayer struct{}

func (fl *FlattenLayer) Forward(input [][][]float64) []float64 {
	size := len(input) * len(input[0]) * len(input[0][0])
	output := make([]float64, size)
	i := 0
	for c := range input {
		for y := range input[c] {
			for x := range input[c][y] {
				output[i] = input[c][y][x]
				i++
			}
		}
	}
	return output
}

// Dense (Fully Connected) Layer
type DenseLayer struct {
	Weights [][]float64
	Biases  []float64
}

func (dl *DenseLayer) Forward(input []float64) []float64 {
	output := make([]float64, len(dl.Biases))
	for i := range output {
		output[i] = dl.Biases[i]
		for j := range input {
			output[i] += dl.Weights[i][j] * input[j]
		}
		output[i] = relu(output[i])
	}
	return output
}

// Utility functions
func pad2D(input [][]float64, padding int) [][]float64 {
	if padding == 0 {
		return input
	}
	h := len(input)
	w := len(input[0])
	padded := make([][]float64, h+2*padding)
	for i := range padded {
		padded[i] = make([]float64, w+2*padding)
	}
	for i := 0; i < h; i++ {
		copy(padded[i+padding][padding:], input[i])
	}
	return padded
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

// Helper functions for initialization
func randomFilters(num, channels, height, width int) [][][][]float64 {
	filters := make([][][][]float64, num)
	for f := range filters {
		filters[f] = make([][][]float64, channels)
		for c := range filters[f] {
			filters[f][c] = make([][]float64, height)
			for h := range filters[f][c] {
				filters[f][c][h] = randomArray(width)
			}
		}
	}
	return filters
}

func randomMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = randomArray(cols)
	}
	return m
}

func randomArray(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.NormFloat64() * 0.1 // Small random values
	}
	return arr
}

func printDimensions(label string, x [][][]float64) {
	fmt.Printf("%s: %d channels %dx%d\n", 
		label, 
		len(x), 
		len(x[0]), 
		len(x[0][0]))
}

func main() {
	// Load and prepare image
	file, err := os.Open("android_Ninja.png")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}

	// Convert to 28x28 grayscale
	resized := image.NewGray(image.Rect(0, 0, 28, 28))
	bounds := img.Bounds()
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			// Simple resize by cropping/scaling
			srcX := bounds.Min.X + x*bounds.Dx()/28
			srcY := bounds.Min.Y + y*bounds.Dy()/28
			resized.SetGray(x, y, color.GrayModel.Convert(img.At(srcX, srcY)).(color.Gray))
		}
	}

	// Create input tensor
	input := make([][][]float64, 1)
	input[0] = make([][]float64, 28)
	for y := 0; y < 28; y++ {
		input[0][y] = make([]float64, 28)
		for x := 0; x < 28; x++ {
			input[0][y][x] = float64(resized.GrayAt(x, y).Y) / 255.0
		}
	}

	// Initialize network
	rand.Seed(time.Now().UnixNano())

	conv1 := &ConvLayer{
		Filters: randomFilters(3, 1, 3, 3),
		Biases:  randomArray(3),
		Stride:  1,
		Padding: 1,
	}

	pool1 := &MaxPoolLayer{PoolSize: 2, Stride: 2}

	conv2 := &ConvLayer{
		Filters: randomFilters(5, 3, 3, 3),
		Biases:  randomArray(5),
		Stride:  1,
		Padding: 1,
	}

	pool2 := &MaxPoolLayer{PoolSize: 2, Stride: 2}

	flat := &FlattenLayer{}

	// Forward pass with dimension checks
	x := conv1.Forward(input)
	printDimensions("After conv1", x)

	x = pool1.Forward(x)
	printDimensions("After pool1", x)

	x = conv2.Forward(x)
	printDimensions("After conv2", x)

	x = pool2.Forward(x)
	printDimensions("After pool2", x)

	flatX := flat.Forward(x)
	fmt.Printf("Flattened size: %d\n", len(flatX))

	// Dynamic dense layer initialization
	dense1 := &DenseLayer{
		Weights: randomMatrix(128, len(flatX)),
		Biases:  randomArray(128),
	}

	denseOut := dense1.Forward(flatX)
	
	dense2 := &DenseLayer{
		Weights: randomMatrix(10, 128),
		Biases:  randomArray(10),
	}

	output := dense2.Forward(denseOut)
	fmt.Println("Final output:", output)
}