#include "nnue.h"

#include "../../../../shared/nnue/definitions.h"
#include "../../../../shared/simd.h"
#include "accumulator.h"

#ifdef _MSC_VER
#define SP_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif

#include "../../../third-party/incbin/incbin.h"
#include "sparse.h"

#ifdef SP_MSVC
#pragma pop_macro("_MSC_VER")
#undef SP_MSVC
#endif

INCBIN(EVAL, EVALFILE);

namespace nnue {

[[nodiscard]] I32 CReLU(I16 value) {
  return std::clamp<I32>(value, 0, arch::kFtQuantization);
}

[[nodiscard]] float CReLU(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

void LoadFromIncBin() {
  if (gEVALSize != sizeof(Network)) {
    fmt::println("Invalid embedded network size: {} bytes; expected {}",
                 gEVALSize,
                 sizeof(Network));
    std::abort();
  }
  // Load the preprocessed network from embedded binary data
  network = reinterpret_cast<Network *>(const_cast<unsigned char *>(gEVALData));
}

Score Evaluate(Board &board) {
  auto &state = board.GetState();
  auto &accumulator = *board.GetAccumulator();

  accumulator.ApplyChanges(state);
  const auto bucket = accumulator.GetOutputBucket(state);
  const int hmc_bucket = GetHmcBucket(state.fifty_moves_clock);

  constexpr int kFtShift = 9;

#if BUILD_HAS_SIMD and !defined(SPARSE_PERMUTE)
  constexpr int kI32Lanes = simd::kNativeLanes<I32>;
  constexpr int kI16Lanes = simd::kNativeLanes<I16>;
  constexpr int kI8Lanes = simd::kNativeLanes<I8>;
  constexpr int kF32Lanes = simd::kNativeLanes<float>;

  const auto quantise_vector = simd::Set<I16>(arch::kFtQuantization);

  // Allow every activated feature to be present in the sparse index list.
  std::array<U16, arch::kL1Size + 8> nnz_indices;
  int nnz_count = 0;

  // Activate the feature layer neurons
  alignas(simd::kAlignment) std::array<U8, arch::kL1Size> feature_output;
  const auto transform_features = [&]<bool kHasHmc>() {
    for (int them = 0; them <= 1; them++) {
      const auto perspective = static_cast<Color>(state.turn ^ them);
      const auto &stm_accumulator = accumulator[perspective];
      const auto hmc = kHasHmc ? network->hmc_weights[hmc_bucket].data() : nullptr;

      const auto load_neurons = [&](int idx) {
        auto value = simd::Load<I16>(&stm_accumulator.psqt[idx]) +
                     simd::Load<I16>(&stm_accumulator.threat[idx]);
        if constexpr (kHasHmc) {
          value = value + simd::Load<I16>(&hmc[idx]);
        }
        return value;
      };

      for (int i = 0; i < arch::kL1Size / 2; i += kI8Lanes) {
        // Clip first accumulator values
        const auto accumulator_value = load_neurons(i);
        const auto pair_accumulator_value = load_neurons(i + arch::kL1Size / 2);

        const auto clipped_value =
            simd::Clip(accumulator_value, arch::kFtQuantization);
        const auto clipped_pair_value =
            simd::Min(pair_accumulator_value, quantise_vector);

        // Clip second accumulator values
        const auto accumulator_value1 = load_neurons(i + kI16Lanes);
        const auto pair_accumulator_value1 =
            load_neurons(i + arch::kL1Size / 2 + kI16Lanes);
        const auto clipped_value1 =
            simd::Clip(accumulator_value1, arch::kFtQuantization);
        const auto clipped_pair_value1 =
            simd::Min(pair_accumulator_value1, quantise_vector);

        // Perform a left-shift on them and multiply the products using the
        // higher 16 bits
        const auto first_product = simd::MulhiEpi16(
            clipped_value << (16 - kFtShift), clipped_pair_value);
        const auto second_product = simd::MulhiEpi16(
            clipped_value1 << (16 - kFtShift), clipped_pair_value1);

        // Pack the two I16 vectors into a U8 vector, which will clamp negative
        // values to 0 because of unsigned saturation. This is why we didn't
        // clamp the pair values to 0 earlier, effectively saving us an
        // operation
        auto &features =
            simd::AsVector<U8>(&feature_output[i + them * arch::kL1Size / 2]);
        features = simd::PackusEpi16(first_product, second_product);

        // Sparse Processing / NNZ (Number of Non-Zero).
        // Each I32 lane represents a group of four U8 features. Store the
        // group index; the forward pass multiplies it by four to obtain the
        // feature offset used by the Dpbusd weight layout.
        const auto nnz_mask = simd::NonZeroMask(simd::Cast<I32>(features));
        const auto nnz_base = U16(i / 4 + them * (arch::kL1Size / 2) / 4);
        for (int chunk = 0; chunk < kI32Lanes; ++chunk) {
          if (nnz_mask & (U16(1) << chunk))
            nnz_indices[nnz_count++] = nnz_base + U16(chunk);
        }
      }
    }
  };

  if (hmc_bucket == arch::kHmcBucketCount) {
    transform_features.template operator()<false>();
  } else {
    transform_features.template operator()<true>();
  }

#ifdef SPARSE_PERMUTE
  sparse::CountActivations(feature_output);
#endif

  // Forward the feature layer neurons to the 2nd layer
  alignas(simd::kAlignment) std::array<I32, arch::kL2Size> l1_sums{};
  {
    int i = 0;
    for (; i < nnz_count - 1; i += 2) {
      const int idx = nnz_indices[i] * 4, idx_two = nnz_indices[i + 1] * 4;
      const auto feature_vector = simd::Cast<U8>(
          simd::Set(*reinterpret_cast<I32 *>(&feature_output[idx])));
      const auto feature_vector_two = simd::Cast<U8>(
          simd::Set(*reinterpret_cast<I32 *>(&feature_output[idx_two])));
      for (int j = 0; j < arch::kL2Size; j += kI32Lanes) {
        const auto weight_vector =
            simd::AsVector<I8>(&network->l1_weights[bucket][idx + j / 4][0]);
        const auto weight_vector_two = simd::AsVector<I8>(
            &network->l1_weights[bucket][idx_two + j / 4][0]);
        auto &features = simd::AsVector<I32>(&l1_sums[j]);
        features = simd::DpbusdEpi32x2(features,
                                       feature_vector,
                                       weight_vector,
                                       feature_vector_two,
                                       weight_vector_two);
      }
    }

    // Handle the remaining features
    for (; i < nnz_count; i++) {
      const int idx = nnz_indices[i] * 4;
      const auto feature_vector = simd::Cast<U8>(
          simd::Set(*reinterpret_cast<I32 *>(&feature_output[idx])));
      for (int j = 0; j < arch::kL2Size; j += kI32Lanes) {
        const auto weight_vector =
            simd::AsVector<I8>(&network->l1_weights[bucket][idx + j / 4][0]);
        auto &features = simd::AsVector<I32>(&l1_sums[j]);
        features = simd::DpbusdEpi32(features, feature_vector, weight_vector);
      }
    }
  }

  // Quantisation constants to convert to float
  constexpr float kL1Normalization =
      static_cast<float>(1 << kFtShift) /
      static_cast<float>(arch::kFtQuantization * arch::kFtQuantization *
                         arch::kL1Quantization);
  const auto l1_multiplier_vector = simd::Set<float>(kL1Normalization);

  alignas(simd::kAlignment) std::array<float, arch::kL2Size> l1_output;
  for (int i = 0; i < arch::kL2Size; i += kF32Lanes) {
    const auto bias_vector =
        simd::AsVector<float>(&network->l1_biases[bucket][i]);
    const auto float_vector =
        simd::Convert<float>(simd::AsVector<I32>(&l1_sums[i]));
    const auto casted_sum =
        simd::MultiplyAdd(float_vector, l1_multiplier_vector, bias_vector);
    simd::AsVector<float>(&l1_output[i]) = simd::Clamp(casted_sum, 0.0f, 1.0f);
  }

  // Forward the feature layer neurons to the 2nd layer
  alignas(simd::kAlignment) std::array<float, arch::kL3Size> l2_sums;
  std::memcpy(
      l2_sums.data(), network->l2_biases[bucket].data(), sizeof(l2_sums));
  for (int i = 0; i < arch::kL2Size; i++) {
    const auto l1_vector = simd::Set<float>(l1_output[i]);
    for (int j = 0; j < arch::kL3Size; j += kF32Lanes) {
      const auto weight_vector =
          simd::AsVector<float>(&network->l2_weights[bucket][i][j]);
      auto &features = simd::AsVector<float>(&l2_sums[j]);
      features = simd::MultiplyAdd(weight_vector, l1_vector, features);
    }
  }

  alignas(simd::kAlignment) std::array<float, arch::kL3Size> l2_output;
  for (int i = 0; i < arch::kL3Size; i += kF32Lanes) {
    simd::AsVector<float>(&l2_output[i]) =
        simd::Clamp(simd::AsVector<float>(&l2_sums[i]), 0.0f, 1.0f);
  }

  // Forward the feature layer neurons to the 3rd (final) layer
  constexpr int kResultChunks = 64 / sizeof(simd::Vepf32);
  std::array<simd::Vepf32, kResultChunks> result_sums;
  result_sums.fill(simd::Zero<float>());
  for (int i = 0; i < arch::kL3Size / kF32Lanes; i += kResultChunks) {
    for (int chunk = 0; chunk < kResultChunks; chunk++) {
      const auto weight_vector = simd::AsVector<float>(
          &network->l3_weights[bucket][(i + chunk) * kF32Lanes]);
      const auto l2_vector =
          simd::AsVector<float>(&l2_output[(i + chunk) * kF32Lanes]);
      result_sums[chunk] =
          simd::MultiplyAdd(l2_vector, weight_vector, result_sums[chunk]);
    }
  }

  const auto l3_output =
      simd::ReduceAdd(result_sums) + network->l3_biases[bucket];
  return static_cast<Score>(l3_output * arch::kEvalScale);

#else
  // Activate the feature layer via pair-wise CReLU multiplication
  std::array<U8, arch::kL1Size> feature_output{};
  for (int them = 0; them <= 1; them++) {
    const auto perspective = static_cast<Color>(state.turn ^ them);
    const auto &stm_accumulator = accumulator[perspective];
    const I16 *hmc = network->hmc_weights[hmc_bucket].data();
    for (int i = 0; i < arch::kL1Size / 2; i++) {
      const auto first_val = CReLU(static_cast<I16>(
          stm_accumulator.psqt[i] + stm_accumulator.threat[i] + hmc[i]));
      const auto second_val =
          CReLU(static_cast<I16>(stm_accumulator.psqt[i + arch::kL1Size / 2] +
                                 stm_accumulator.threat[i + arch::kL1Size / 2] +
                                 hmc[i + arch::kL1Size / 2]));

      const auto product = (first_val * second_val) >> 9;
      feature_output[i + them * arch::kL1Size / 2] = static_cast<U8>(product);
    }
  }

#ifdef SPARSE_PERMUTE
  sparse::CountActivations(feature_output);
#endif

  const float kL1Normalization =
      static_cast<float>(1 << kFtShift) /
      static_cast<float>(arch::kFtQuantization * arch::kFtQuantization *
                         arch::kL1Quantization);

  // Forward the feature layer neurons to the 2nd layer
  std::array<I32, arch::kL2Size> l1_sums{};
  for (int i = 0; i < arch::kL1Size; i++) {
    if (!feature_output[i]) continue;

    for (int j = 0; j < arch::kL2Size; j++) {
      l1_sums[j] += feature_output[i] * network->l1_weights[bucket][i][j];
    }
  }

  // Activate 2nd layer neurons
  std::array<float, arch::kL2Size> l1_output{};
  for (int i = 0; i < arch::kL2Size; i++) {
    l1_output[i] = CReLU(static_cast<float>(l1_sums[i]) * kL1Normalization +
                         network->l1_biases[bucket][i]);
  }

  // Forward the 2nd layer neurons to the 3rd layer
  std::array<float, arch::kL3Size> l2_output{};
  std::memcpy(
      l2_output.data(), network->l2_biases[bucket].data(), sizeof(l2_output));
  for (int i = 0; i < arch::kL2Size; i++) {
    for (int j = 0; j < arch::kL3Size; j++) {
      l2_output[j] = std::fma(
          l1_output[i], network->l2_weights[bucket][i][j], l2_output[j]);
    }
  }

  // Forward 3rd layer neurons to output layer
  constexpr int kResultChunks = 64 / sizeof(float);
  std::array<float, kResultChunks> result_sums{};

  for (int i = 0; i < arch::kL3Size; i += kResultChunks) {
    for (int chunk = 0; chunk < kResultChunks; chunk++) {
      const float activated = CReLU(l2_output[i + chunk]);
      result_sums[chunk] = std::fma(activated,
                                    network->l3_weights[bucket][i + chunk],
                                    result_sums[chunk]);
    }
  }

  const float l3_output =
      network->l3_biases[bucket] + simd::ReduceAdd(result_sums);

  // Scale output
  return static_cast<Score>(l3_output * arch::kEvalScale);
#endif
}

}  // namespace nnue
