/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include "rkt_coefs.h"
#include "rkt_ml.h"

struct pipe_resource *
rkt_fill_weights(struct rkt_ml_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation)
{
   struct pipe_context *pcontext = subgraph->context;
   unsigned weights_width = poperation->conv.weight_tensor->dims[2];
   unsigned weights_height = poperation->conv.weight_tensor->dims[1];
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned input_channels_real = poperation->input_tensors[0]->dims[3];
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   unsigned output_channels_real = poperation->output_tensors[0]->dims[3];
   unsigned weights_size;
   int zero_point = rkt_zero_point(poperation->conv.weight_tensor);
   uint8_t offset = rkt_sign_offset(poperation->conv.weight_tensor->is_signed);
   struct pipe_transfer *transfer_out;
   void *map = poperation->conv.weight_tensor->data;
   uint8_t(*weights_in)[weights_height][weights_width][input_channels] = map;
   struct pipe_resource *rsc;
   uint8_t *weights_out;

   input_channels =
      rkt_weight_channels_padded(input_channels, rkt_is_depthwise(poperation));

   output_channels = align(output_channels, 2);
   if (rkt_is_depthwise(poperation))
      output_channels = 1;

   weights_size = weights_width * weights_height * output_channels *
                  align(input_channels, WEIGHT_ATOMIC_SIZE) * 2;

   rsc =
      pipe_buffer_create(pcontext->screen, 0, PIPE_USAGE_DEFAULT, weights_size);
   weights_out = pipe_buffer_map(pcontext, rsc, PIPE_MAP_WRITE, &transfer_out);

   unsigned input_channel_groups = WEIGHT_ATOMIC_SIZE;
   if (rkt_is_depthwise(poperation))
      input_channel_groups *= 2;

   unsigned input_channels_1 =
      DIV_ROUND_UP(input_channels, input_channel_groups);
   unsigned input_channels_2 = MIN2(input_channels, input_channel_groups);

   unsigned n = 0;
   for (int oc1 = 0; oc1 < DIV_ROUND_UP(output_channels, WEIGHT_ATOMIC_SIZE);
        oc1++) {
      for (int ic1 = 0; ic1 < input_channels_1; ic1++) {
         for (int y = 0; y < weights_height; y++) {
            for (int x = 0; x < weights_width; x++) {
               for (int oc2 = 0; oc2 < MIN2(output_channels, WEIGHT_ATOMIC_SIZE);
                    oc2++) {
                  for (int ic2 = 0; ic2 < input_channels_2; ic2++) {
                     unsigned oc = oc1 * WEIGHT_ATOMIC_SIZE + oc2;
                     unsigned ic = ic1 * input_channel_groups + ic2;
                     if (output_channels_real > 2 &&
                         oc >= align(output_channels_real, 2))
                        continue;

                     if (oc >= output_channels_real)
                        weights_out[n++] = 0x0;
                     else if (ic >= input_channels_real) {
                        /* Pad a tap only up to the aligned channel count.
                         * This branch runs for a whole weight group, and for
                         * depthwise input_channel_groups is
                         * 2*WEIGHT_ATOMIC_SIZE, so a group can reach past
                         * input_channels. The counter n lays the taps out one
                         * after the other, and advances by exactly
                         * input_channels, as padded above, per tap; emitting
                         * more than that shifted every tap after the first.
                         */
                        if (ic < input_channels)
                           weights_out[n++] = zero_point;
                     } else
                        weights_out[n++] = weights_in[oc][y][x][ic] - offset;
                  }
               }
            }
         }
      }
   }

   if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
      static int task = 0;
      rkt_dump_buffer(weights_out, "weights", 0, task++, 0, weights_size);
   }

   pipe_buffer_unmap(pcontext, transfer_out);

   return rsc;
}

static int32_t
calculate_bias_correction(struct rkt_ml_subgraph *subgraph,
                          const struct pipe_ml_operation *poperation,
                          unsigned oc, void *map)
{
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   int input_zero_point = rkt_zero_point(poperation->input_tensors[0]);
   unsigned weights_width = poperation->conv.weight_tensor->dims[2];
   unsigned weights_height = poperation->conv.weight_tensor->dims[1];
   int weight_zero_point = rkt_zero_point(poperation->conv.weight_tensor);
   uint8_t offset = rkt_sign_offset(poperation->conv.weight_tensor->is_signed);
   uint8_t(*weights)[weights_height][weights_width][input_channels] = map;

   int32_t correction = 0;
   if (rkt_is_depthwise(poperation)) {
      for (unsigned y = 0; y < weights_height; y++) {
         for (unsigned x = 0; x < weights_width; x++) {
            correction +=
               ((int8_t)(weights[0][y][x][oc] - offset) - weight_zero_point) *
               input_zero_point;
         }
      }
   } else {
      for (unsigned y = 0; y < weights_height; y++) {
         for (unsigned x = 0; x < weights_width; x++) {
            for (unsigned ic = 0; ic < input_channels; ic++) {
               correction +=
                  ((int8_t)(weights[oc][y][x][ic] - offset) - weight_zero_point) *
                  input_zero_point;
            }
         }
      }
   }

   return correction;
}

float
rkt_max_weights_scale(const struct pipe_ml_operation *poperation)
{
   const struct pipe_tensor *weight_tensor = poperation->conv.weight_tensor;
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   float scale = weight_tensor->scale;

   if (weight_tensor->scales) {
      for (unsigned oc = 0; oc < output_channels; oc++)
         scale = MAX2(scale, weight_tensor->scales[oc]);
   }

   return scale;
}

/* With a fused PReLU, BN is taken and BS applies the whole ratio */
static void
pack_per_channel(void *map, const int32_t *biases,
                 const struct pipe_ml_operation *poperation, bool prelu)
{
   const float *scales = poperation->conv.weight_tensor->scales;
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   float max_scale = rkt_max_weights_scale(poperation);

   int16_t *bn_muls =
      (int16_t *)((uint8_t *)map + rkt_bn_muls_offset(output_channels));

   for (unsigned oc = 0; oc < output_channels; oc++) {
      uint8_t *group = (uint8_t *)map + oc / BS_GROUP * BS_GROUP *
                                           (sizeof(int32_t) + sizeof(int16_t));
      int32_t *group_biases = (int32_t *)group;
      int16_t *group_muls = (int16_t *)(group + BS_GROUP * sizeof(int32_t));
      /* Channels whose weights are all 0 can have a scale of 0 */
      float ratio = max_scale > 0 ? scales[oc] / max_scale : 0;
      int k = prelu       ? 0
              : ratio > 0 ? CLAMP((int)floorf(-log2f(ratio)), 0, BN_MUL_SHIFT)
                          : BN_MUL_SHIFT;

      group_biases[oc % BS_GROUP] = biases[oc];
      group_muls[oc % BS_GROUP] =
         MIN2(lroundf(ratio * (1 << k) * (1 << BS_MUL_SHIFT)), INT16_MAX);
      bn_muls[oc] = 1 << (BN_MUL_SHIFT - k);
   }
}

static float
prelu_alpha_value(const struct pipe_tensor *alpha, unsigned channel)
{
   unsigned count = alpha->dims[0] * alpha->dims[1] * alpha->dims[2] *
                    alpha->dims[3];
   unsigned i = count == 1 ? 0 : channel;
   int q = alpha->is_signed ? ((const int8_t *)alpha->data)[i]
                            : ((const uint8_t *)alpha->data)[i];

   return alpha->scale * (q - alpha->zero_point);
}

/*
 * The operands of a fused PReLU: |alpha| for BN at the largest shift that keeps
 * them in int16, and the signs for EW. The multipliers shift positive and
 * negative results by different amounts, and in PReLU mode positive inputs
 * skip the multiplication but are still shifted, so a negative alpha can't be
 * applied in one stage: its results are positive, and would need the shift of
 * 0 of the unmultiplied positive values.
 */
static unsigned
pack_prelu(void *map, const struct pipe_tensor *alpha, unsigned output_channels,
           bool per_channel, bool *sign)
{
   int16_t *muls = (int16_t *)((uint8_t *)map +
                               rkt_bn_operands_offset(output_channels,
                                                      per_channel));
   int8_t *signs = (int8_t *)((uint8_t *)map +
                              rkt_ew_operands_offset(output_channels,
                                                     per_channel));
   float max_alpha = 0;
   unsigned shift = BN_MUL_SHIFT;

   for (unsigned oc = 0; oc < output_channels; oc++)
      max_alpha = MAX2(max_alpha, fabsf(prelu_alpha_value(alpha, oc)));

   while (shift > 0 && lroundf(max_alpha * (1 << shift)) > INT16_MAX)
      shift--;

   *sign = false;
   for (unsigned oc = 0; oc < output_channels; oc++) {
      float value = prelu_alpha_value(alpha, oc);

      muls[oc] = lroundf(fabsf(value) * (1 << shift));
      signs[oc] = value < 0 ? -1 : 1;
      *sign |= value < 0;
   }

   return shift;
}

struct pipe_resource *
rkt_fill_biases(struct rkt_ml_subgraph *subgraph,
                const struct pipe_ml_operation *poperation,
                const struct pipe_tensor *prelu_alpha,
                unsigned *truncate_bits, unsigned *prelu_shift,
                bool *prelu_sign)
{
   struct pipe_context *pcontext = subgraph->context;
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   unsigned weights_size = poperation->conv.weight_tensor->dims[1];
   struct pipe_transfer *transfer_out;
   int32_t *biases_in = (int32_t *)poperation->conv.bias_tensor->data;
   void *weights = poperation->conv.weight_tensor->data;
   bool per_channel = poperation->conv.weight_tensor->scales != NULL;
   struct pipe_resource *rsc;
   uint32_t *biases;
   unsigned size;
   void *map;

   if (prelu_alpha)
      size = rkt_ew_operands_offset(output_channels, per_channel) +
             output_channels;
   else if (per_channel)
      size = rkt_bn_muls_offset(output_channels) +
             output_channels * sizeof(int16_t);
   else
      size = output_channels * sizeof(uint32_t);

   rsc = pipe_buffer_create(pcontext->screen, 0, PIPE_USAGE_DEFAULT, size);
   map = pipe_buffer_map(pcontext, rsc, PIPE_MAP_WRITE, &transfer_out);
   memset(map, 0, size);
   biases = per_channel ? calloc(output_channels, sizeof(*biases)) : map;

   // DBG("weight_scale %x\n",
   // fui(poperation->conv.weight_tensor->scale));
   /* TODO: Figure out when exactly we need to truncate */
   /* From
    * http://nvdla.org/hw/v1/ias/unit_description.html#convolution-accumulator :
    *
    * The final result of accumulator in CACC is 48bits for INT16 and 34bits for
    * INT8. The bit width between CACC and SDP is 32. For precisions INT8 and
    * INT16, there is a round and saturation operation before sending the result
    * to SDP. The precision of rounding is configured by field CLIP_TRUNCATE in
    * register D_CLIP_CFG. For FP16, the value is just converted from FP48 to
    * FP32.
    */
   if (fui(poperation->conv.weight_tensor->scale) == 0x3a88323f ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c0060de ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c06022d ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c1642e3 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c1e3f51 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c5c8aa8 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c615e93 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c7326a2 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3c783013 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3d1748e6 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3d282992 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3d2e87ae ||
       fui(poperation->conv.weight_tensor->scale) == 0x3d77f5f6 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3a9a5956 ||
       fui(poperation->conv.weight_tensor->scale) == 0x3caebc56)
      *truncate_bits = 1;
   else
      *truncate_bits = 0;

   int32_t max_bias = 0;
   int32_t max_corr = 0;
   unsigned max_num_bits = 0;
   bool retry = true;
   while (retry) {
      for (int oc = 0; oc < output_channels; oc++) {
         int32_t corr =
            calculate_bias_correction(subgraph, poperation, oc, weights);
         biases[oc] = (biases_in[oc] - corr) / (1 << *truncate_bits);

         int64_t max_val =
            (biases_in[oc] - corr + 255 * 255 * weights_size * weights_size) /
            (1 << *truncate_bits);
         unsigned num_bits = ceil(log(abs((int32_t)max_val)) / log(2)) + 1;
         max_bias = MAX2(max_bias, biases[oc]);
         max_corr = MAX2(max_corr, corr);
         max_num_bits = MAX2(max_num_bits, num_bits);

         /* TODO: This doesn't actually work, num_bits doesn't go above 19, and the
          * blob sometimes truncates way below */
         if (num_bits > 32) {
            (*truncate_bits)++;
            retry = true;
         } else
            retry = false;
      }
   }

   if (per_channel) {
      pack_per_channel(map, (int32_t *)biases, poperation,
                       prelu_alpha != NULL);
      free(biases);
   }

   *prelu_shift = 0;
   *prelu_sign = false;
   if (prelu_alpha)
      *prelu_shift = pack_prelu(map, prelu_alpha, output_channels, per_channel,
                                prelu_sign);

   if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
      static int task = 0;
      rkt_dump_buffer(map, "biases", 0, task++, 0, size);
   }

   pipe_buffer_unmap(pcontext, transfer_out);

   return rsc;
}
