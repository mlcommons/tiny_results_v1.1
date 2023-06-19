# ONNC Calibration Details for MLPerf Tiny v1.1

We employ a per-tensor symmetric uniform quantization scheme for the weight, bias, and activation tensors, as described by the following equation:
```math
T_{int8} = \text{clip} \biggl ( \text{round} \biggl ( \frac{T_{fp32}}{s_T}\biggr ), -128, 127 \biggr )
```

Here, $`T_{fp32}`$ represents the original 32-bit floating-point tensor, $`s_T`$ denotes the scale factor for tensor $`T`$, and $`T_{int8}`$ corresponds to the final quantized tensor.

## Determining $`s_T`$
For different tensors, we adopt different rules for determining $`s_T`$.

### For the weight/bias tensor $`T_w`$
1. We retain the full expression range of the weight/bias tensor:
    ```math
    s_T = \max(\text{abs}(T_w))
    ```
2. Then, we align this scale to a power of 2:
    ```math
    s_{T_w} = 2^{\text{round}(\log_2(s_T))}
    ```
    Note that this step is not necessary but is required by the CMSIS-NN q7 API we utilize.

### For activation tensor $`T_a`$
1. From the 2048 candidate scales:
    ```math
    s_{T_i} = i \cdot \frac{\max(\text{abs}(T_a))}{2048} \text{, where } i \in \mathbb{N} \text{ and } 0 < i \le 2048
    ```
    we select a scale $`s_T = s_{T_i}`$ that minimizes the L2 distance:
    ```math
    L_2D(s_{T_i} \cdot T_{int8}, T_{fp32})
    ```
2. We then align this scale to a power of 2:
    ```math
    s_{T_a} = 2^{\lceil \log_2(s_T) \rceil}
    ```
    Again, this step is not mandatory but is required by the CMSIS-NN q7 API we employ.
    Note that we use the "ceil" function instead of "round" in this context.

## Rounding
All rounding operations, denoted as $`round()`$, employ the "round half away from zero" method.