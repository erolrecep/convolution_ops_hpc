package main

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
)

var kernel = [3][3]float32{
	{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0},
	{2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0},
	{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0},
}

func minVal(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxVal(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s <technique 1-4> <input_image> <output_image>\n", os.Args[0])
		os.Exit(1)
	}

	technique, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid technique: %v\n", err)
		os.Exit(1)
	}
	inputPath := os.Args[2]
	outputPath := os.Args[3]

	file, err := os.Open(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open image %s: %v\n", inputPath, err)
		os.Exit(1)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to decode image %s: %v\n", inputPath, err)
		os.Exit(1)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	channels := 3
	rowLen := width * channels
	totalBytes := width * height * channels

	// Flatten image to contiguous byte slice
	inBytes := make([]byte, totalBytes)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			offset := y*rowLen + x*channels
			inBytes[offset+0] = byte(r >> 8)
			inBytes[offset+1] = byte(g >> 8)
			inBytes[offset+2] = byte(b >> 8)
		}
	}

	outBytes := make([]byte, totalBytes)
	numWorkers := runtime.NumCPU()

	startTime := time.Now()

	var wg sync.WaitGroup

	if technique == 3 || technique == 1 {
		chunkSize := (height + numWorkers - 1) / numWorkers
		for i := 0; i < numWorkers; i++ {
			yStart := i * chunkSize
			yEnd := minVal((i+1)*chunkSize, height)
			if yStart >= height {
				break
			}
			wg.Add(1)
			go func(ys, ye int) {
				defer wg.Done()
				for y := ys; y < ye; y++ {
					for x := 0; x < width; x++ {
						for c := 0; c < channels; c++ {
							var sum float32 = 0.0
							for ky := -1; ky <= 1; ky++ {
								py := minVal(maxVal(y+ky, 0), height-1)
								rowOffset := py * rowLen
								for kx := -1; kx <= 1; kx++ {
									px := minVal(maxVal(x+kx, 0), width-1)
									pVal := float32(inBytes[rowOffset+px*channels+c])
									sum += pVal * kernel[ky+1][kx+1]
								}
							}
							val := byte(minVal(maxVal(int(sum+0.5), 0), 255))
							outBytes[y*rowLen+x*channels+c] = val
						}
					}
				}
			}(yStart, yEnd)
		}
		wg.Wait()
	} else {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					var sum float32 = 0.0
					for ky := -1; ky <= 1; ky++ {
						py := minVal(maxVal(y+ky, 0), height-1)
						rowOffset := py * rowLen
						for kx := -1; kx <= 1; kx++ {
							px := minVal(maxVal(x+kx, 0), width-1)
							pVal := float32(inBytes[rowOffset+px*channels+c])
							sum += pVal * kernel[ky+1][kx+1]
						}
					}
					val := byte(minVal(maxVal(int(sum+0.5), 0), 255))
					outBytes[y*rowLen+x*channels+c] = val
				}
			}
		}
	}

	elapsed := time.Since(startTime).Seconds()

	if outputPath != "none" {
		outImg := image.NewRGBA(image.Rect(0, 0, width, height))
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*rowLen + x*channels
				outImg.Set(x, y, color.RGBA{
					R: outBytes[offset+0],
					G: outBytes[offset+1],
					B: outBytes[offset+2],
					A: 255,
				})
			}
		}
		outFile, err := os.Create(outputPath)
		if err == nil {
			defer outFile.Close()
			png.Encode(outFile, outImg)
		}
	}

	fmt.Printf("EXECUTION_TIME: %.6f\n", elapsed)
}
