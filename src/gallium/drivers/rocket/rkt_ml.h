/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef RKT_ML_H
#define RKT_ML_H

#include <util/u_dynarray.h>

#include "rkt_device.h"

// http://nvdla.org/hw/v1/ias/unit_description.html#convolution-buffer
#define CBUF_BANK_SIZE        32768
#define CBUF_BANKS            12
#define CBUF_ENTRIES_PER_BANK 256
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)
#define FEATURE_ATOMIC_SIZE   16
#define WEIGHT_ATOMIC_SIZE    32
#define ATOMIC_K_SIZE         16

static inline unsigned
rkt_weight_channels_padded(unsigned channels, bool depthwise)
{
   unsigned padded = align(MAX2(channels, FEATURE_ATOMIC_SIZE),
                           FEATURE_ATOMIC_SIZE);

   if (depthwise &&
       padded % (2 * WEIGHT_ATOMIC_SIZE) == 3 * FEATURE_ATOMIC_SIZE)
      padded += FEATURE_ATOMIC_SIZE;

   return padded;
}

/* The hardware computes in int8: uint8 values and zero points get 0x80
 * subtracted where they are read, which keeps the real values. */
static inline uint8_t
rkt_sign_offset(bool is_signed)
{
   return is_signed ? 0 : 0x80;
}

static inline int
rkt_zero_point(const struct pipe_tensor *tensor)
{
   return tensor->zero_point - rkt_sign_offset(tensor->is_signed);
}

/*
 * BS and BN correspond to the X1 and X2 units of NVDLA's SDP, which can read
 * the operands of their multipliers per channel from memory:
 * http://nvdla.org/hw/v1/ias/unit_description.html#single-point-data-processor
 *
 * With per-channel weight scales, each channel's scale relative to the largest
 * one, r = 2^-k * m with m in (0.5, 1], is applied by the multipliers of BS
 * and BN, which read their int16 operands from the biases buffer:
 *
 *  - BS: m in units of 2^-BS_MUL_SHIFT. BS reads groups of BS_GROUP output
 *    channels: BS_GROUP int32 biases followed by BS_GROUP int16 multipliers.
 *  - BN: the exact power of two 2^-k, in units of 2^-BN_MUL_SHIFT, one int16
 *    per channel starting at rkt_bn_muls_offset().
 *
 * The shift is shared by all channels, so a single multiplier would lose the
 * precision of small ratios: in units of 2^-BS_MUL_SHIFT, 0.001 becomes 16 and
 * keeps 4 bits, while 0.512 * 2^-9 is 8389 in BS and exact in BN.
 *
 * A fused PReLU takes BN, and EW if an alpha is negative (see
 * rkt_fill_biases()), their operands in the biases buffer from
 * rkt_bn_operands_offset() and rkt_ew_operands_offset(). With per-channel
 * weight scales BS then applies the whole ratio r. That only loses precision
 * for ratios far below the 0.07 seen in real models, such as the ~1e-6 of
 * channels whose weights are all zero, where it makes no difference.
 */
#define BS_GROUP      8
#define BS_MUL_SHIFT  14
#define BN_MUL_SHIFT  14
#define BN_MULS_ALIGN 64

static inline unsigned
rkt_bn_muls_offset(unsigned output_channels)
{
   return align(DIV_ROUND_UP(output_channels, BS_GROUP) * BS_GROUP *
                   (sizeof(int32_t) + sizeof(int16_t)),
                BN_MULS_ALIGN);
}

/* Where BN's per-channel operands start in the biases buffer */
static inline unsigned
rkt_bn_operands_offset(unsigned output_channels, bool per_channel)
{
   return per_channel ? rkt_bn_muls_offset(output_channels)
                      : align(output_channels * sizeof(int32_t), BN_MULS_ALIGN);
}

/* EW's int8 operands of a fused PReLU, after BN's */
static inline unsigned
rkt_ew_operands_offset(unsigned output_channels, bool per_channel)
{
   return align(rkt_bn_operands_offset(output_channels, per_channel) +
                   output_channels * sizeof(int16_t),
                BN_MULS_ALIGN);
}

struct split_task {
   unsigned num;

   unsigned top_slice;
   unsigned bottom_slice;
   unsigned num_overlap_slices;
   unsigned num_retain_slices;
   unsigned convolutions;

   unsigned pad_top;
   unsigned pad_bottom;
   unsigned pad_left;
   unsigned pad_right;

   unsigned stride_x;
   unsigned stride_y;

   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   unsigned input_channels_real;
   int8_t input_zero_point;
   float input_scale;
   unsigned input_data_entries;
   int input_line_stride;
   int input_surface_stride;
   unsigned input_offset;

   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   unsigned output_channels_real;
   int8_t output_zero_point;
   float output_scale;
   int output_surface_stride;
   unsigned output_offset;

   unsigned weights_width;
   unsigned weights_height;
   unsigned weights_kernels;
   int8_t weights_zero_point;
   float weights_scale;

   unsigned input_banks;
   unsigned weights_banks;

   unsigned atomic_count;
   unsigned surfaces_per_row;

   unsigned regcfg_amount;
   uint32_t regcfg_addr;
};

struct rkt_operation {
   struct pipe_resource *regcmd;
   struct pipe_resource *weights;
   struct pipe_resource *biases;

   bool depthwise;
   bool relu;
   bool reuse_weights_cbuf;
   unsigned truncate_bits;
   unsigned padding_top;
   unsigned padding_bottom;
   unsigned padding_left;
   unsigned padding_right;
   unsigned stride;

   bool addition_input;
   int addition_offset;
   float addition_scale;

   unsigned input_index;
   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   int8_t input_zero_point;
   float input_scale;

   unsigned output_index;
   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   int8_t output_zero_point;
   float output_scale;

   unsigned weights_width;
   unsigned weights_height;
   int8_t weights_zero_point;
   float weights_scale; /* the largest one, with per-channel scales */
   bool per_channel;

   /* PReLU fused into BN, its |alpha| in units of 2^-prelu_shift, and EW
    * applying the signs of negative alphas */
   bool prelu;
   unsigned prelu_shift;
   bool prelu_sign;

   int add_tensor;

   struct util_dynarray tasks; /* struct split_task */
};

/* BN applies the per-channel power-of-two scales or a fused PReLU */
static inline bool
rkt_bn_used(const struct rkt_operation *operation)
{
   return operation->per_channel || operation->prelu;
}

/* BN's shifts of positive and of negative results */
static inline unsigned
rkt_bn_shift(const struct rkt_operation *operation)
{
   return operation->prelu ? 0 : BN_MUL_SHIFT;
}

static inline unsigned
rkt_bn_neg_shift(const struct rkt_operation *operation)
{
   return operation->prelu ? operation->prelu_shift : BN_MUL_SHIFT;
}

struct rkt_ml_subgraph {
   struct pipe_ml_subgraph base;

   struct pipe_context *context;
   struct util_dynarray operations; /* rkt_operation */
   struct util_dynarray tensors;    /* pipe_resource* */
};

bool
rkt_ml_operation_supported(struct pipe_ml_device *pdevice, const struct pipe_ml_operation *operation);

struct pipe_ml_subgraph *
rkt_ml_subgraph_create(struct pipe_ml_device *pdevice,
                       const struct pipe_ml_operation *poperations,
                       unsigned count);

void rkt_ml_subgraph_invoke(struct pipe_context *pcontext,
                            struct pipe_ml_subgraph *psubgraph,
                            unsigned inputs_count, unsigned input_idxs[],
                            void *inputs[], bool is_signed[]);

void rkt_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                                  struct pipe_ml_subgraph *psubgraph,
                                  unsigned outputs_count,
                                  unsigned output_idxs[], void *outputs[],
                                  bool is_signed[]);

void rkt_ml_subgraph_destroy(struct pipe_ml_device *pdevice,
                             struct pipe_ml_subgraph *psubgraph);

struct rkt_resource *rkt_get_tensor(struct rkt_ml_subgraph *subgraph,
                                    unsigned idx);

bool rkt_is_depthwise(const struct pipe_ml_operation *poperation);

void rkt_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                     int suboperation_nr, int offset, unsigned size);

#endif /* RKT_ML_H */
