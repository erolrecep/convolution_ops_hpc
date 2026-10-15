package main

/*
#cgo CFLAGS: -O3 -I../c_project/include
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../c_project/include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../c_project/include/stb_image_write.h"
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
	"unsafe"
)

var kernel = [3][3]float32{
	{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0},
	{2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0},
	{1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0},
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
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

	cInputPath := C.CString(inputPath)
	defer C.free(unsafe.Pointer(cInputPath))

	var w, h, ch C.int
	cData := C.stbi_load(cInputPath, &w, &h, &ch, 0)
	if cData == nil {
		fmt.Fprintf(os.Stderr, "Failed to load image: %s\n", inputPath)
		os.Exit(1)
	}
	defer C.stbi_image_free(unsafe.Pointer(cData))

	width := int(w)
	height := int(h)
	channels := int(ch)
	totalBytes := width * height * channels

	// Convert C buffer to Go slice without copying
	inBytes := unsafe.Slice((*byte)(unsafe.Pointer(cData)), totalBytes)
	outBytes := make([]byte, totalBytes)

	numWorkers := runtime.NumCPU()

	startTime := time.Now()

	var wg sync.WaitGroup

	if technique == 3 || technique == 1 {
		// Row-based strip partitioning across goroutines
		chunkSize := (height + numWorkers - 1) / numWorkers
		for i := 0; i < numWorkers; i++ {
			yStart := i * chunkSize
			yEnd := min((i+1)*chunkSize, height)
			if yStart >= height {
				break
			}
			wg.Add(1)
			go func(ys, ye int) {
				defer wg.Done()
				rowLen := width * channels
				for y := ys; y < ye; y++ {
					for x := 0; x < width; x++ {
						for c := 0; c < channels; c++ {
							var sum float32 = 0.0
							for ky := -1; ky <= 1; ky++ {
								py := min(max(y+ky, 0), height-1)
								rowOffset := py * rowLen
								for kx := -1; kx <= 1; kx++ {
									px := min(max(x+kx, 0), width-1)
									pVal := float32(inBytes[rowOffset+px*channels+c])
									sum += pVal * kernel[ky+1][kx+1]
								}
							}
							val := byte(min(max(int(sum+0.5), 0), 255))
							outBytes[y*rowLen+x*channels+c] = val
						}
					}
				}
			}(yStart, yEnd)
		}
		wg.Wait()
	} else {
		// Sequential single-worker fallback for technique comparison
		rowLen := width * channels
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				for c := 0; c < channels; c++ {
					var sum float32 = 0.0
					for ky := -1; ky <= 1; ky++ {
						py := min(max(y+ky, 0), height-1)
						rowOffset := py * rowLen
						for kx := -1; kx <= 1; kx++ {
							px := min(max(x+kx, 0), width-1)
							pVal := float32(inBytes[rowOffset+px*channels+c])
							sum += pVal * kernel[ky+1][kx+1]
						}
					}
					val := byte(min(max(int(sum+0.5), 0), 255))
					outBytes[y*rowLen+x*channels+c] = val
				}
			}
		}
	}

	elapsed := time.Since(startTime).Seconds()

	if outputPath != "none" {
		cOutputPath := C.CString(outputPath)
		defer C.free(unsafe.Pointer(cOutputPath))
		C.stbi_write_jpg(cOutputPath, w, h, ch, unsafe.Pointer(&outBytes[0]), 90)
	}

	fmt.Printf("EXECUTION_TIME: %.6f\n", elapsed)
}
