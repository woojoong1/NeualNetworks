// =============================
// include/vsnn/Loss.hpp  (수정본)
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>   // memcpy
#include "Matrix.hpp"
#include "Ops.hpp"

using namespace std;

namespace vsnn {
    // 기존 클래스/함수 시그니처 유지
    class SoftmaxCrossEntropy {
    private:
        // Backward에서 사용할 확률 버퍼 (행 단위 softmax 결과 저장)
        Matrix probs_;

    public:
        // logits: (N x C), y: 정답 인덱스(N)
        // 수치안정화(log-sum-exp)로 softmax와 loss를 한 번에 계산
        float Forward(const Matrix& logits, const vector<int>& y) {
            const int N = logits.Rows();
            const int C = logits.Cols();

            if (probs_.Rows() != N || probs_.Cols() != C) probs_.Reset(N, C);

            float loss = 0.0f;

            const float* x = &logits.Data()[0];
            float* p = &probs_.Data()[0];

            for (int n = 0; n < N; ++n) {
                const float* row = x + static_cast<size_t>(n) * C;
                float* prow = p + static_cast<size_t>(n) * C;

                // 1) row 최대값 (m) ? overflow/underflow 방지
                float m = row[0];
                for (int j = 1; j < C; ++j) if (row[j] > m) m = row[j];

                // 2) exp(row - m) 합
                float denom = 0.0f;
                for (int j = 0; j < C; ++j) {
                    const float e = std::exp(row[j] - m);
                    prow[j] = e;           // 일단 exp만 저장
                    denom += e;
                }

                // 3) 확률로 정규화 + loss 누적
                const float inv = 1.0f / denom;
                for (int j = 0; j < C; ++j) prow[j] *= inv;

                const int t = y[n];
                const float pt = std::max(1e-12f, prow[t]);   // 안전 clamp
                loss += -std::log(pt);
            }

            return loss / static_cast<float>(N);
        }

        // dLogits: (N x C) ? dL/d(logits) 채워서 반환
        void Backward(const vector<int>& y, Matrix& dLogits) {
            const int N = probs_.Rows();
            const int C = probs_.Cols();

            if (dLogits.Rows() != N || dLogits.Cols() != C) dLogits.Reset(N, C);

            const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(C);

            // 1) dLogits = probs_  (연속 메모리 복사로 가속)
            std::memcpy(dLogits.Raw().data(),
                probs_.Raw().data(),
                elems * sizeof(float));

            // 2) 정답 위치에서 -1
            for (int n = 0; n < N; ++n) {
                dLogits(n, y[n]) -= 1.0f;
            }

            // 3) 평균 스케일
            const float invN = 1.0f / static_cast<float>(N);
            float* g = dLogits.Raw().data();
            for (size_t i = 0; i < elems; ++i) g[i] *= invN;
        }
    };
}
