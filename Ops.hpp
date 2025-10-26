// =============================
// include/vsnn/Ops.hpp
// =============================
#pragma once
#include <cmath>
#include <algorithm>
#include "Matrix.hpp"

using namespace std;

namespace vsnn {
	class Ops {
	public:
		// Y = X * W with shapes: (N,in) * (in,out) = (N,out)
        static void MatMul(const Matrix& X, const Matrix& W, Matrix& Y) {
            assert(X.Cols() == W.Rows());
            const int N = X.Rows();
            const int K = X.Cols();
            const int M = W.Cols();

            if (Y.Rows() != N || Y.Cols() != M)
                Y.Reset(N, M);
            else
                Y.Fill(0.0f); // 누적 연산 대비 초기화

            // 타일 크기 (CPU 캐시 고려)
            const int TILE = 32;

            const float* Xd = X.Data();
            const float* Wd = W.Data();
            float* Yd = Y.Data();

            // 캐시 친화적 타일링 행렬곱
            for (int i0 = 0; i0 < N; i0 += TILE) {
                int iMax = std::min(i0 + TILE, N);
                for (int k0 = 0; k0 < K; k0 += TILE) {
                    int kMax = std::min(k0 + TILE, K);
                    for (int j0 = 0; j0 < M; j0 += TILE) {
                        int jMax = std::min(j0 + TILE, M);

                        // 타일 내부 계산
                        for (int i = i0; i < iMax; ++i) {
                            float* yRow = &Yd[i * M];
                            const float* xRow = &Xd[i * K];
                            for (int k = k0; k < kMax; ++k) {
                                float xval = xRow[k];
                                const float* wRow = &Wd[k * M];
                                for (int j = j0; j < jMax; ++j) {
                                    yRow[j] += xval * wRow[j];
                                }
                            }
                        }
                    }
                }
            }
        }

		static void AddRowBias(Matrix& Y, const Matrix& b) {
			assert(b.Rows() == 1 && b.Cols() == Y.Cols());
			for (i32 n = 0; n < Y.Rows(); ++n)
				for (i32 j = 0; j < Y.Cols(); ++j) Y(n, j) += b(0, j);
		}
		static void ReLUForward(const Matrix& X, Matrix& Y) {
			if (Y.Rows() != X.Rows() || Y.Cols() != X.Cols()) Y.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					Y(r, c) = (X(r, c) > 0.0f) ? X(r, c) : 0.0f;
		}
		static void ReLUBackward(const Matrix& X, const Matrix& dY, Matrix& dX) {
			if (dX.Rows() != X.Rows() || dX.Cols() != X.Cols()) dX.Reset(X.Rows(), X.Cols());
			for (i32 r = 0; r < X.Rows(); ++r)
				for (i32 c = 0; c < X.Cols(); ++c)
					dX(r, c) = (X(r, c) > 0.0f) ? dY(r, c) : 0.0f;
		}
		static void SoftmaxRow(const float* logits, float* probs, int C) {
			float m = logits[0];
			for (int i = 1; i < C; ++i) m = max(m, logits[i]);
			float s = 0.0f; for (int i = 0; i < C; ++i) { probs[i] = exp(logits[i] - m); s += probs[i]; }
			if (s == 0.0f) s = 1e-12f;
			for (int i = 0; i < C; ++i) probs[i] /= s;
		}
	};
}
