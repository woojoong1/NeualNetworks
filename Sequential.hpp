// =============================
// include/vsnn/Sequential.hpp (최적화 버전: 핑퐁 & 무복사)
// =============================
#pragma once
#include <memory>
#include <vector>
#include <cstring>      // memcpy
#include "Layer.hpp"
#include "Matrix.hpp"

using namespace std;

namespace vsnn {

    class Sequential {
    private:
        vector<unique_ptr<Layer>> layers_;
        vector<Matrix> acts_; // X0..X_L (Backward용 캐시)

    public:
        template<typename T, typename... Args>
        T* Add(Args&&... args) {
            layers_.push_back(make_unique<T>(forward<Args>(args)...));
            return static_cast<T*>(layers_.back().get());
        }

        // Forward: 중간 임시(cur/nxt) 제거, acts_[i+1]에 직접 기록
        void Forward(const Matrix& X, Matrix& out) {
            const size_t L = layers_.size();
            if (L == 0) {
                // 레이어가 없으면 out = X (memcpy로 한 번에)
                if (out.Rows() != X.Rows() || out.Cols() != X.Cols()) out.ResetNoInit(X.Rows(), X.Cols());
                std::memcpy(out.Data(), X.Data(),
                    sizeof(float) * static_cast<size_t>(X.Rows()) * X.Cols());
                return;
            }

            // X0..X_L 캐시 벡터 준비 (재할당 최소화)
            if (acts_.size() != L + 1) acts_.resize(L + 1);

            // X를 acts_[0]에 복사 (Backward용)
            if (acts_[0].Rows() != X.Rows() || acts_[0].Cols() != X.Cols()) acts_[0].ResetNoInit(X.Rows(), X.Cols());
            std::memcpy(acts_[0].Data(), X.Data(),
                sizeof(float) * static_cast<size_t>(X.Rows()) * X.Cols());

            // 각 레이어의 출력을 acts_[i+1]에 직접 생성
            for (size_t i = 0; i < L; ++i) {
                // 다음 출력 버퍼는 레이어 내부에서 크기를 결정할 수 있지만,
                // 최소한 입력과 동일 크기로 NoInit 해 두면 불필요한 제로필을 피할 수 있음.
                // (레이어 Forward에서 Reset/ResetNoInit 호출 시 덮어써짐)
                if (acts_[i + 1].Rows() != acts_[i].Rows() || acts_[i + 1].Cols() != acts_[i].Cols()) {
                    acts_[i + 1].ResetNoInit(acts_[i].Rows(), acts_[i].Cols());
                }
                layers_[i]->Forward(acts_[i], acts_[i + 1]);
            }

            // 최종 결과 out에 복사 (필요 크기만 맞추고 memcpy 한 번)
            const Matrix& last = acts_.back();
            if (out.Rows() != last.Rows() || out.Cols() != last.Cols()) out.ResetNoInit(last.Rows(), last.Cols());
            std::memcpy(out.Data(), last.Data(),
                sizeof(float) * static_cast<size_t>(last.Rows()) * last.Cols());
        }

        // Backward: grad 핑퐁 (임시 최소화)
        void Backward(const Matrix& dOut) {
            const size_t L = layers_.size();
            if (L == 0) return;

            Matrix grad_ping, grad_pong;

            // grad_ping ← dOut (memcpy 한 번)
            if (grad_ping.Rows() != dOut.Rows() || grad_ping.Cols() != dOut.Cols())
                grad_ping.ResetNoInit(dOut.Rows(), dOut.Cols());
            std::memcpy(grad_ping.Data(), dOut.Data(),
                sizeof(float) * static_cast<size_t>(dOut.Rows()) * dOut.Cols());

            // 역순으로 각 레이어에 전달 (acts_[i]는 Forward에서 캐시됨)
            for (int i = static_cast<int>(L) - 1; i >= 0; --i) {
                // 출력 크기와 동일 크기로 NoInit (Backward가 덮어씀)
                if (grad_pong.Rows() != acts_[i].Rows() || grad_pong.Cols() != acts_[i].Cols())
                    grad_pong.ResetNoInit(acts_[i].Rows(), acts_[i].Cols());

                layers_[i]->Backward(acts_[i], grad_ping, grad_pong);
                std::swap(grad_ping, grad_pong);
            }
            // 최종 dX는 grad_ping에 남아있음. 외부에서 필요하다면 반환 인터페이스를 확장하세요.
        }

        void ZeroGrad() { for (auto& L : layers_) L->ZeroGrad(); }
        void Step(float lr) { for (auto& L : layers_) L->Step(lr); }

        // ---- Introspection for Updater ----
        size_t NumLayers() const { return layers_.size(); }
        Layer* LayerAt(size_t i) { return layers_[i].get(); }
        const Layer* LayerAt(size_t i) const { return layers_[i].get(); }
    };

}
