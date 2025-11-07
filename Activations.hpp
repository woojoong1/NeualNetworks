// =============================
// include/vsnn/Activations.hpp (수정본)
// =============================
#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
// #include "Ops.hpp"

namespace vsnn {

    class ReLU : public Layer {
    public:
        // 임시 버퍼/함수 호출 없이 바로 계산
        void Forward(const Matrix& X, Matrix& Y) override {
            const int R = X.Rows();
            const int C = X.Cols();

            if (Y.Rows() != R || Y.Cols() != C) Y.Reset(R, C);

            const float* x = X.Data();
            float* y = Y.Data();
            const size_t N = static_cast<size_t>(R) * static_cast<size_t>(C);

            for (size_t i = 0; i < N; ++i) {
                const float v = x[i];
                y[i] = (v > 0.0f) ? v : 0.0f;
            }
        }

        // dX = (X > 0) ? dY : 0
        void Backward(const Matrix& X, const Matrix& dY, Matrix& dX) override {
            const int R = X.Rows();
            const int C = X.Cols();

            if (dX.Rows() != R || dX.Cols() != C) dX.Reset(R, C);

            const float* x = X.Data();
            const float* gy = dY.Data();
            float* gx = dX.Data();
            const size_t N = static_cast<size_t>(R) * static_cast<size_t>(C);

            for (size_t i = 0; i < N; ++i) {
                gx[i] = (x[i] > 0.0f) ? gy[i] : 0.0f;
            }
        }
    };

}
