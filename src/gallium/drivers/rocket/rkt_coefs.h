/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef RKT_COEFS_H
#define RKT_COEFS_H

#include "rkt_ml.h"

struct pipe_resource *
rkt_fill_weights(struct rkt_ml_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation);

float rkt_max_weights_scale(const struct pipe_ml_operation *poperation);

struct pipe_resource *
rkt_fill_biases(struct rkt_ml_subgraph *subgraph,
                const struct pipe_ml_operation *poperation,
                const struct pipe_tensor *prelu_alpha,
                unsigned *truncate_bits, unsigned *prelu_shift,
                bool *prelu_sign);

#endif /* RKT_COEFS_H */