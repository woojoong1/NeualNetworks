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
            const int TILE = 32;

            const float* Xd = X.Data();
            const float* dYd = dY.Data();

            // ================================================================
            // gW = X^T * dY  (row-major optimized)
            // ================================================================
            if (gW_.Rows() != IN || gW_.Cols() != OUT) gW_.Reset(IN, OUT);
            gW_.Fill(0.0f);
            float* gWd = gW_.Data();

            for (int i0 = 0; i0 < N; i0 += TILE) {
                int iMax = std::min(i0 + TILE, N);
                for (int k0 = 0; k0 < IN; k0 += TILE) {
                    int kMax = std::min(k0 + TILE, IN);
                    for (int j0 = 0; j0 < OUT; j0 += TILE) {
                        int jMax = std::min(j0 + TILE, OUT);

                        for (int i = i0; i < iMax; ++i) {
                            const float* xRow = &Xd[i * IN];
                            const float* dyRow = &dYd[i * OUT];
                            for (int k = k0; k < kMax; ++k) {
                                float xVal = xRow[k];
                                float* gWk = &gWd[k * OUT];
                                for (int j = j0; j < jMax; ++j) {
                                    gWk[j] += xVal * dyRow[j];
                                }
                            }
                        }
                    }
                }
            }

            // ================================================================
            // gb = sum_rows(dY)  (row-major optimized)
            // ================================================================
            if (gb_.Rows() != 1 || gb_.Cols() != OUT) gb_.Reset(1, OUT);
            gb_.Fill(0.0f);

            float* gbd = gb_.Data();

            for (int i = 0; i < N; ++i) {
                const float* dyRow = &dYd[i * OUT];  // i번째 행 시작 주소
                for (int j = 0; j < OUT; ++j) {
                    gbd[j] += dyRow[j];              // gb(0,j) += dY(i,j)
                }
            }

            // ================================================================
            // dX = dY * W^T  (fully row-major optimized)
            // ================================================================
            if (dX.Rows() != N || dX.Cols() != IN) dX.Reset(N, IN);
            dX.Fill(0.0f);
            float* dXd = dX.Data();
            const float* Wd = W_.Data();

            // Step 1: Transpose W into Wt (OUT × IN)
            std::vector<float> Wt(static_cast<size_t>(OUT) * IN);
            for (int j = 0; j < OUT; ++j) {
                const float* Wcol = &Wd[j];       // W(:, j)
                float* wtRow = &Wt[j * IN];       // Wt(j, :)
                for (int k = 0; k < IN; ++k) {
                    wtRow[k] = Wcol[k * OUT];     // Wt(j, k) = W(k, j)
                }
            }
            const float* Wtd = Wt.data();

            // Step 2: Compute dX = dY * W^T
            for (int i0 = 0; i0 < N; i0 += TILE) {
                int iMax = std::min(i0 + TILE, N);
                for (int j0 = 0; j0 < OUT; j0 += TILE) {
                    int jMax = std::min(j0 + TILE, OUT);
                    for (int k0 = 0; k0 < IN; k0 += TILE) {
                        int kMax = std::min(k0 + TILE, IN);

                        for (int i = i0; i < iMax; ++i) {
                            float* dxRow = &dXd[i * IN];
                            const float* dyRow = &dYd[i * OUT];
                            for (int j = j0; j < jMax; ++j) {
                                float dyv = dyRow[j];               // dY(i, j)
                                const float* wtRow = &Wtd[j * IN];  // Wt(j, :)
                                for (int k = k0; k < kMax; ++k) {
                                    dxRow[k] += dyv * wtRow[k];
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
