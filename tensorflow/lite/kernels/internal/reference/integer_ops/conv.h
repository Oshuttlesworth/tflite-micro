/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "mvm/mvm.h"
#include "stm32h7xx_hal.h"

namespace tflite {
namespace reference_integer_ops {

// 'distributed convolution' function by Orhun Tamyigit
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data)
{

  // Requirements:
  // batch size must be 1
  // filters and input must be square matrices
  // Input image of size 28x28 is separated into 16 pieces of size 7x7
  // I use padding = 1 and kernel size 3 so output shape can be preserved after concatenating all of 16 pieces

  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = params.output_offset;
  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
	TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int input_dim = 7;
  const int padding = 1;
  const int piece_ID_y = 1;  // row ID: 0, 1, 2 or 3
  const int piece_ID_x = 1;  // column ID: 0, 1, 2 or 3. There are 16 pieces of size 7x7

  int8_t input_matrix[input_dim + 2 * padding][input_dim + 2 * padding];

  // pad input matrix:
  for (int i = 0; i < input_dim + 2 * padding; ++i)
  {
	input_matrix[0][i] = 0;
	input_matrix[input_dim + 2 * padding - 1][i] = 0;
	input_matrix[i][0] = 0;
	input_matrix[i][input_dim + 2 * padding - 1] = 0;
  }

  // construct input window as 2D array of shape 7x7 (including padding, the shape is 9x9):
  int y, x;
  y = 1;
  for (int in_y = piece_ID_y * input_dim; in_y < (piece_ID_y + 1) * input_dim; ++in_y)
  {
	x = 1;
	for (int in_x = piece_ID_x * input_dim; in_x < (piece_ID_x + 1) * input_dim; ++in_x)
	{
	  input_matrix[y][x] = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
	  x += 1;
	}
	y += 1;
  }

  if (piece_ID_x != 0)  // piece overlaps with the rest of the image from the left side
  {
	int in_x = piece_ID_x * input_dim - 1;
	x = 0;
	y = 1;
	for (int in_y = piece_ID_y * input_dim; in_y < (piece_ID_y + 1) * input_dim; ++in_y)
	{
	  input_matrix[y][x] = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
	  y += 1;
	}
  }

  if (piece_ID_x != 3)  // piece overlaps with the rest of the image from right side
  {
	int in_x = (piece_ID_x + 1) * input_dim;
	x = input_dim + 1;
	y = 1;
	for (int in_y = piece_ID_y * input_dim; in_y < (piece_ID_y + 1) * input_dim; ++in_y)
	{
	  input_matrix[y][x] = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
	  y += 1;
	}
  }

  if (piece_ID_y != 0)  // piece overlaps with the rest of the image from top
  {
	int in_y = piece_ID_y * input_dim - 1;
	y = 0;
	x = 1;
	for (int in_x = piece_ID_x * input_dim; in_x < (piece_ID_x + 1) * input_dim; ++in_x)
	{
	  input_matrix[y][x] = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
	  x += 1;
	}
  }

  if (piece_ID_y != 3)  // piece overlaps with the rest of the image from bottom
  {
	int in_y = (piece_ID_y + 1) * input_dim;
	y = input_dim + 1;
	x = 1;
	for (int in_x = piece_ID_x * input_dim; in_x < (piece_ID_x + 1) * input_dim; ++in_x)
	{
	  input_matrix[y][x] = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
	  x += 1;
	}
  }

  // this is how the padded-input-piece looks like now:
  	// 0,  0,  0,  0,  0,  0,  0,  0,  0,
	// 0,  0,  1,  2,  3,  4,  5,  6,  7,
	// 0, 28, 29, 30, 31, 32, 33, 34,  35,
	// 0, 56, 57, 58, 59, 60, 61, 62,  63,
	// 0, 84, 85, 86, 87, 88, 89, 90,  91,
	// 0,  0,  1,  2,  3,  4,  5,  6,  7,
	// 0, 28, 29, 30, 31, 31, 33, 34,  35,
	// 0, 56, 57, 58, 59, 60, 61, 62,  63,
    // 0, 84, 85, 86, 87, 88, 89, 90,  91,

  // convert each receptive field (with size 3x3) into column vectors of length 9
  // i.e. apply input-window-unrolling
  // there are 49 columns because there are 7*7 output entries

  int8_t input_unrolled[filter_height * filter_width][input_dim * input_dim];
  int8_t filter_array[output_depth][filter_height * filter_width];

  int out_idx = 0;
  int unrolling_idx;
  for (int out_i = 1; out_i < input_dim + 2 * padding - 1; out_i++)
  {
	for (int out_j = 1; out_j < input_dim + 2 * padding - 1; out_j++)
	{
	  unrolling_idx = 0;
	  for (int i = out_i - 1; i < out_i - 1 + filter_height; i++)
	  {
		for (int j = out_j - 1; j < out_j - 1 + filter_width; j++)
		{
		  input_unrolled[unrolling_idx][out_idx] = input_matrix[i][j];
		  unrolling_idx += 1;
		}
	  }
	  out_idx += 1;
	}
  }

  // construct filter matrix of shape: (num_channels,filter_length) i.e. (4,9)
  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
  {
	// construct 1D filter array:
	int filter_index = 0;
	for (int y = 0; y < filter_height; y++)
	{
	  for (int x = 0; x < filter_width; x++)
	  {
		filter_array[out_channel][filter_index] = filter_data[Offset(filter_shape, out_channel, y, x, 0)];
		filter_index += 1;
	  }
	}
  }

  // now, we have two matrices: (4,9) filter matrix and (9,49) unrolled input matrix
  // this is how the unrolled input matrix looks like:
			// 0,  0,  0,  ..., 32, 33
			// 0,  0,  0,  ..., 33, 34
			// 0,  0,  0,  ..., 34, 35
			// 0,  0,  1,  ..., 60, 61
			// 0,  1,  2,  ..., 61, 62
			// 1,  2,  3,  ..., 62, 63
			// 0,  28, 29, ..., 88, 89
			// 28, 29, 30, ..., 89, 90
			// 29, 30, 31, ..., 90, 91

  // flatten the unrolled-input matrix and filter matrix (for bit-sliced mvm):
  int8_t* input_unrolled_1d = new int8_t[filter_height * filter_width * input_dim * input_dim];
  int8_t* weights_1d = new int8_t[output_depth * filter_height * filter_width];
  int idx_1d = 0;
  for (int i=0; i<filter_height*filter_width; ++i)
  {
	for (int j=0; j<input_dim*input_dim; j++)
	{
		input_unrolled_1d[idx_1d] = input_unrolled[i][j];
		idx_1d += 1;
	}
  }

  idx_1d = 0;
  for (int i=0; i<output_depth; ++i)
  {
	for (int j=0; j<filter_height * filter_width; j++)
	{
		weights_1d[idx_1d] = filter_array[i][j];
		idx_1d += 1;
	}
  }

  /*
  int32_t conv_result_prev[output_depth][input_dim * input_dim];  // shape is 4x49
  // matrix multiplication (to be computed on the external device):
  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
  {
	for (int out_index = 0; out_index < input_dim * input_dim; out_index++)
	{
		conv_result_prev[out_channel][out_index] = 0;
	  for (int k = 0; k < filter_height * filter_width; k++)
	  {
		  conv_result_prev[out_channel][out_index] += filter_array[out_channel][k] * input_unrolled[k][out_index];
	  }
	}
  }
  */

  mvm_bitslicing::MVM mvm(filter_height*filter_width, output_depth, input_dim*input_dim);
  mvm.set_inputs(input_unrolled_1d);
  mvm.set_weights(weights_1d);
  mvm.multiply();
  int32_t** conv_result = mvm.get_mvm_result();

  for (int out_channel = 0; out_channel < output_depth; ++out_channel)
  {
	for (int out_index = 0; out_index < input_dim * input_dim; out_index++)
	{
	  int32_t curr_res = conv_result[out_channel][out_index];
	  curr_res = MultiplyByQuantizedMultiplier(curr_res, output_multiplier[out_channel], output_shift[out_channel]);
	  curr_res += output_offset;
	  curr_res = std::max(curr_res, output_activation_min);
	  curr_res = std::min(curr_res, output_activation_max);
	  int out_y = out_index / input_dim;
	  int out_x = out_index % input_dim;
	  output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] = static_cast<int8_t>(curr_res);
	}
  }

  // Convolution for the rest of the input image
  for (int out_y = 0; out_y < output_height; ++out_y)
  {
	  const int in_y_origin = out_y - padding;
	  for (int out_x = 0; out_x < output_width; ++out_x)
	  {
		if((out_x >= piece_ID_x * input_dim) &&
				(out_x < (piece_ID_x + 1) * input_dim) &&
				(out_y >= piece_ID_y * input_dim) &&
				(out_y < (piece_ID_y + 1) * input_dim))
			continue; // skip if this output entry is already calculated

		const int in_x_origin = out_x - padding;
		for (int out_channel = 0; out_channel < output_depth; ++out_channel)
		{
		  int32_t acc = 0;
		  for (int filter_y = 0; filter_y < filter_height; ++filter_y)
		  {
			const int in_y = in_y_origin +  filter_y;
			for (int filter_x = 0; filter_x < filter_width; ++filter_x)
			{
			  const int in_x = in_x_origin +  filter_x;

			  // Zero padding by omitting the areas outside the image.
			  const bool is_point_inside_image =
					  (in_x >= 0) && (in_x < input_width) &&
					  (in_y >= 0) && (in_y < input_height);

			  if (!is_point_inside_image)
				continue;

			  int8_t input_val = input_data[Offset(input_shape, 0, in_y, in_x, 0)];
			  int8_t filter_val = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, 0)];
			  // Accumulate with 32 bits accumulator:
			  acc += filter_val * (input_val + input_offset);
			}
		  }

		  if (bias_data)
			acc += bias_data[out_channel];

		  acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
		  acc += output_offset;
		  acc = std::max(acc, output_activation_min);
		  acc = std::min(acc, output_activation_max);

		  output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
		}
	  }
  }
}

/*
// Fixed-point per-channel-quantization convolution reference kernel.
// ===============================================================
// original convolution code of tensorflow
// ===============================================================
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data)

{

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            auto group = out_channel / filters_per_group;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;

                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }

                for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
                  int32_t input_val = input_data[Offset(input_shape, batch, in_y, in_x, in_channel + group * filter_input_depth)];
                  int32_t filter_val = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8_t, even though
                  // it is represented using int32_t. int32_t += int8_t *
                  // (int8_t - int8_t) so the highest value we can get from each
                  // accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(b/174275578): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }

            if (bias_data) {
              acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);

            output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
          }
        }
      }
    }
}
*/

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
