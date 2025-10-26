// =============================
// include/vsnn/Dense.hpp
// =============================
#pragma once
#include "Layer.hpp"
#include "Ops.hpp"
#include "Initializer.hpp"
#include <vector>
#include <omp.h>

namespace vsnn {
	class Dense : public Layer {
	private:
		Matrix W_, b_;
		Matrix gW_, gb_;

	public:
		Dense(i32 in_dim, i32 out_dim, float init_scale = 0.01f)
			: W_(in_dim, out_dim), b_(1, out_dim), gW_(in_dim, out_dim), gb_(1, out_dim) {
			Initializer::Uniform(W_, init_scale, 123);
			b_.Fill(0.0f); gW_.Fill(0.0f); gb_.Fill(0.0f);
		}

		void Forward(const Matrix& X, Matrix& Y) override {
			Ops::MatMul(X, W_, Y);
			Ops::AddRowBias(Y, b_);
		}

        void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
            const int N = X.Rows();      // batch size
            const int IN = X.Cols();     // input dimension
            const int OUT = dY.Cols();   // output dimension

            // gW = X^T * dY
            if (gW_.Rows() != IN || gW_.Cols() != OUT) gW_.Reset(IN, OUT);
            gW_.Fill(0.0f);

            const int TILE = 32;
            const float* Xd = X.Data();
            const float* dYd = dY.Data();
            float* gWd = gW_.Data();

            for (int k0 = 0; k0 < IN; k0 += TILE) {
                int kMax = std::min(k0 + TILE, IN);
                for (int j0 = 0; j0 < OUT; j0 += TILE) {
                    int jMax = std::min(j0 + TILE, OUT);
                    for (int i0 = 0; i0 < N; i0 += TILE) {
                        int iMax = std::min(i0 + TILE, N);
                        for (int k = k0; k < kMax; ++k) {
                            for (int j = j0; j < jMax; ++j) {
                                float acc = 0.0f;
                                for (int i = i0; i < iMax; ++i) {
                                    acc += Xd[i * IN + k] * dYd[i * OUT + j];
                                }
                                gWd[k * OUT + j] += acc;
                            }
                        }
                    }
                }
            }

            // gb = sum_rows(dY)
            if (gb_.Rows() != 1 || gb_.Cols() != OUT) gb_.Reset(1, OUT);
            gb_.Fill(0.0f);
            for (int j = 0; j < OUT; ++j) {
                float acc = 0.0f;
                for (int i = 0; i < N; ++i) acc += dY(i, j);
                gb_(0, j) = acc;
            }

            // dX = dY * W^T
            if (dX.Rows() != N || dX.Cols() != IN) dX.Reset(N, IN);
            dX.Fill(0.0f);
            const float* Wd = W_.Data();
            float* dXd = dX.Data();

            for (int i0 = 0; i0 < N; i0 += TILE) {
                int iMax = std::min(i0 + TILE, N);
                for (int k0 = 0; k0 < IN; k0 += TILE) {
                    int kMax = std::min(k0 + TILE, IN);
                    for (int j0 = 0; j0 < OUT; j0 += TILE) {
                        int jMax = std::min(j0 + TILE, OUT);
                        for (int i = i0; i < iMax; ++i) {
                            float* dxRow = &dXd[i * IN];
                            const float* dyRow = &dYd[i * OUT];
                            for (int j = j0; j < jMax; ++j) {
                                float dyv = dyRow[j];
                                for (int k = k0; k < kMax; ++k) {
                                    dxRow[k] += dyv * Wd[k * OUT + j];
                                }
                            }
                        }
                    }
                }
            }
        }


		void ZeroGrad() override { gW_.Fill(0.0f); gb_.Fill(0.0f); }
		void Step(float) override {}

		Matrix& WRef() { return W_; }
		Matrix& bRef() { return b_; }
		Matrix& gWRef() { return gW_; }
		Matrix& gbRef() { return gb_; }
		const Matrix& W() const { return W_; }
		const Matrix& b() const { return b_; }
		const Matrix& gW() const { return gW_; }
		const Matrix& gb() const { return gb_; }
	};
}
