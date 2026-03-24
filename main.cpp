// Programming Assignment #4: Sequential Histogram Equalization for Tone Mapping (CPU)

#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int BINS = 256;

// ------------------------------------------------------------
// Sequential CPU implementations of each kernel/stage.
// ------------------------------------------------------------

// TODO 1: RGB -> luminance Y in [0,1].
// Input: h_rgb_u8 (interleaved RGB, 0..255)
// Output: h_y (float, 0..1)
static void rgb_to_luminance(const uint8_t* h_rgb_u8, float* h_y, int width, int height) {
  int N = width * height;
  for (int idx = 0; idx < N; idx++) {
    // Load R,G,B as floats in [0,1].
    float R = (float)(h_rgb_u8[idx * 3 + 0]) / 255.0f;
    float G = (float)(h_rgb_u8[idx * 3 + 1]) / 255.0f;
    float B = (float)(h_rgb_u8[idx * 3 + 2]) / 255.0f;
    // Y = 0.299R + 0.587G + 0.114B.
    float y = 0.299f * R + 0.587f * G + 0.114f * B;
    // Write Y to h_y[idx].
    h_y[idx] = y;
  }
}

// TODO 2: Sequential reduction for min and max.
static float reduce_min(const float* h_in, int N) {
  float result = 1e30f;
  for (int i = 0; i < N; i++) {
    if (h_in[i] < result) result = h_in[i];
  }
  return result;
}

static float reduce_max(const float* h_in, int N) {
  float result = -1e30f;
  for (int i = 0; i < N; i++) {
    if (h_in[i] > result) result = h_in[i];
  }
  return result;
}

// TODO 3: Histogram (sequential).
// Input luminance h_y, min/max, output h_hist[256].
static void histogram256(const float* h_y, int* h_hist, int N, float y_min, float y_max) {
  // Zero histogram.
  for (int i = 0; i < BINS; i++) h_hist[i] = 0;
  // Map y -> bin in [0,255] using formula.
  // NOTE: Handle y_max == y_min case to avoid divide-by-zero.
  for (int i = 0; i < N; i++) {
    int bin;
    if (y_max == y_min) {
      bin = 0;
    } else {
      bin = (int)floorf((h_y[i] - y_min) / (y_max - y_min) * 255.0f);
    }
    if (bin < 0) bin = 0;
    if (bin > 255) bin = 255;
    h_hist[bin]++;
  }
}

// TODO 4: Exclusive prefix sum (scan) for n=256.
// Input: h_hist[256], Output: h_cdf[256] (exclusive prefix sum).
static void exclusive_scan_256(const int* h_hist, int* h_cdf) {
  h_cdf[0] = 0;
  for (int i = 1; i < BINS; i++) {
    h_cdf[i] = h_cdf[i - 1] + h_hist[i - 1];
  }
}

// TODO 5: Compute CDF_min (the smallest CDF[k] where hist[k] > 0).
static int compute_cdf_min(const int* h_hist, const int* h_cdf) {
  // Find the smallest cdf[k] such that hist[k] > 0.
  // Return that value (cdf_min).
  for (int k = 0; k < BINS; k++) {
    if (h_hist[k] > 0) {
      return h_cdf[k];
    }
  }
  return 0;
}

// TODO 5 (continued): Remap luminance using the equalized CDF.
static void remap_luminance(const float* h_y, float* h_y_new, const int* h_cdf,
                            int N, float y_min, float y_max, int cdf_min) {
  for (int idx = 0; idx < N; idx++) {
    // y -> bin.
    int bin;
    if (y_max == y_min) {
      bin = 0;
    } else {
      bin = (int)floorf((h_y[idx] - y_min) / (y_max - y_min) * 255.0f);
    }
    if (bin < 0) bin = 0;
    if (bin > 255) bin = 255;
    // y_new = (cdf[bin] - cdf_min) / (N - cdf_min).
    float y_new;
    if (N - cdf_min == 0) {
      y_new = 0.0f;
    } else {
      y_new = (float)(h_cdf[bin] - cdf_min) / (float)(N - cdf_min);
    }
    // Clamp to [0,1] and write to h_y_new[idx].
    if (y_new < 0.0f) y_new = 0.0f;
    if (y_new > 1.0f) y_new = 1.0f;
    h_y_new[idx] = y_new;
  }
}

// TODO 6: Restore RGB using ratio Y_new / Y and write uint8 output.
// Input: original RGB uint8, original Y, new Y.
// Output: uint8 RGB image.
static void restore_rgb(const uint8_t* h_rgb_in, const float* h_y, const float* h_y_new,
                        uint8_t* h_rgb_out, int width, int height) {
  int N = width * height;
  for (int idx = 0; idx < N; idx++) {
    // Load R,G,B in [0,1].
    float r = h_rgb_in[idx * 3 + 0] / 255.0f;
    float g = h_rgb_in[idx * 3 + 1] / 255.0f;
    float b = h_rgb_in[idx * 3 + 2] / 255.0f;
    // ratio = (Y > 0) ? Y_new / Y : 0.
    float ratio = 0.0f;
    if (h_y[idx] > 0.0f) {
      ratio = h_y_new[idx] / h_y[idx];
    }
    // Rn = clamp(R * ratio, 0, 1) similarly for G,B.
    float rn = std::min(std::max(r * ratio, 0.0f), 1.0f);
    float gn = std::min(std::max(g * ratio, 0.0f), 1.0f);
    float bn = std::min(std::max(b * ratio, 0.0f), 1.0f);
    // Convert to uint8 and write.
    h_rgb_out[idx * 3 + 0] = (uint8_t)(rn * 255.0f);
    h_rgb_out[idx * 3 + 1] = (uint8_t)(gn * 255.0f);
    h_rgb_out[idx * 3 + 2] = (uint8_t)(bn * 255.0f);
  }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  std::string input_path = "input.jpg";
  std::string output_path = "output_equalized.jpg";
  if (argc >= 2) input_path = argv[1];
  if (argc >= 3) output_path = argv[2];

  // Load input image
  int width = 0, height = 0, channels = 0;
  uint8_t* h_img = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
  if (!h_img) {
    fprintf(stderr, "Failed to load image: %s\n", input_path.c_str());
    return 1;
  }
  channels = 3;
  const int N = width * height;
  fprintf(stdout, "Loaded %s (%d x %d), N=%d\n", input_path.c_str(), width, height, N);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // Allocate host memory
  std::vector<float> h_y(N);
  std::vector<float> h_y_new(N);
  int h_hist[BINS];
  int h_cdf[BINS];
  std::vector<uint8_t> h_out((size_t)N * 3);

  // 1) RGB -> luminance
  rgb_to_luminance(h_img, h_y.data(), width, height);

  // 2) min/max by sequential reduction
  float y_min = reduce_min(h_y.data(), N);
  float y_max = reduce_max(h_y.data(), N);
  fprintf(stdout, "y_min=%f y_max=%f\n", y_min, y_max);

  // 3) histogram
  histogram256(h_y.data(), h_hist, N, y_min, y_max);

  // 4) scan -> cdf
  exclusive_scan_256(h_hist, h_cdf);

  // 5) compute CDF_min
  int cdf_min = compute_cdf_min(h_hist, h_cdf);
  fprintf(stdout, "cdf_min=%d\n", cdf_min);

  // 6) remap luminance
  remap_luminance(h_y.data(), h_y_new.data(), h_cdf, N, y_min, y_max, cdf_min);

  // 7) restore RGB and write output
  restore_rgb(h_img, h_y.data(), h_y_new.data(), h_out.data(), width, height);
  
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
  // Save output JPG
  if (!stbi_write_jpg(output_path.c_str(), width, height, 3, h_out.data(), 95)) {
    fprintf(stderr, "Failed to write output image: %s\n", output_path.c_str());
    return 1;
  }
  fprintf(stdout, "Wrote %s\n", output_path.c_str());

  // Cleanup
  stbi_image_free(h_img);

  return 0;
}