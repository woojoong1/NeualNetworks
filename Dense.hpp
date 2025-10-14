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
			if (gW_.Rows() != W_.Rows() || gW_.Cols() != W_.Cols()) gW_.Reset(W_.Rows(), W_.Cols());
			if (gb_.Rows() != 1 || gb_.Cols() != W_.Cols()) gb_.Reset(1, W_.Cols());
			if (dX.Rows() != X.Rows() || dX.Cols() != W_.Rows()) dX.Reset(X.Rows(), W_.Rows());

			const int N = X.Rows();
			const int Din = W_.Rows();
			const int Dout = W_.Cols();
			const int Ddense = 10;

			// Èñ¼Ò ÀÎµ¦½º Ä³½Ì
			std::vector<std::vector<int>> nz_cols_per_row(N);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < N; ++i) {
				auto& nz = nz_cols_per_row[i];
				for (int k = Ddense; k < Din; ++k)
					if (X(i, k) != 0.0f)
						nz.push_back(k);
			}

			// gW ÃÊ±âÈ­
			gW_.Fill(0.0f);

			// -------------------------------
			// gW = X^T * dY
			// (°¢ »ùÇÃ ´ÜÀ§·Î º´·ÄÈ­)
			// -------------------------------
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < N; ++i) {
				const auto& nz = nz_cols_per_row[i];
				for (int j = 0; j < Dout; ++j) {
					const float dy = dY(i, j);

					// (1) ¿¬¼ÓÇü Æ¯Â¡(0~9)
					for (int k = 0; k < Ddense; ++k) {
#pragma omp atomic
						gW_(k, j) += X(i, k) * dy;
					}

					// (2) Èñ¼Ò one-hot Æ¯Â¡(10~)
					for (int idx = 0; idx < (int)nz.size(); ++idx) {
						int k = nz[idx];
#pragma omp atomic
						gW_(k, j) += X(i, k) * dy;
					}
				}
			}

			// -------------------------------
			// gb = sum_rows(dY)
			// -------------------------------
#pragma omp parallel for schedule(static)
			for (int j = 0; j < Dout; ++j) {
				float acc = 0.0f;
				for (int i = 0; i < N; ++i)
					acc += dY(i, j);
				gb_(0, j) = acc;
			}

			// -------------------------------
			// dX = dY * W^T
			// -------------------------------
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < N; ++i) {
				for (int k = 0; k < Din; ++k) {
					float acc = 0.0f;
					for (int j = 0; j < Dout; ++j)
						acc += dY(i, j) * W_(k, j);
					dX(i, k) = acc;
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
