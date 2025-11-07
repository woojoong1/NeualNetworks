#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstring>      // memcpy
#include "Sequential.hpp"
#include "Loss.hpp"
#include "Timer.hpp"

using namespace std;

namespace vsnn {

    struct TrainConfig {
        int epochs = 50;
        int batch = 64;
        float lr = 1e-2f;
        int warmup = 1;
        int repeats = 3;
        unsigned seed = 0;
    };

    struct TrainReport {
        double median_ms_per_epoch = 0.0;
        double median_update_ms_per_epoch = 0.0;
        float last_loss = 0.0f;
    };

    class Trainer {
    private:
        // 인덱스 기반 슬라이스: 원본 X/y는 그대로 두고, 선택 행만 배치로 복사
        static void SliceBatchIdx(const Matrix& X, const vector<int>& y,
            const vector<int>& idx, int beg, int end,
            Matrix& Xb, vector<int>& yb) {
            const int B = end - beg;
            const int D = X.Cols();
            if (Xb.Rows() != B || Xb.Cols() != D) Xb.ResetNoInit(B, D);
            yb.resize(B);

            const float* Xd = X.Data();
            float* Xbd = Xb.Data();
            for (int i = 0; i < B; ++i) {
                const int src = idx[beg + i];
                std::memcpy(&Xbd[static_cast<size_t>(i) * D],
                    &Xd[static_cast<size_t>(src) * D],
                    sizeof(float) * D);
                yb[i] = y[src];
            }
        }

        static double median_of(vector<double>& v) {
            sort(v.begin(), v.end());
            return v[v.size() / 2];
        }

    public:
        template<typename Updater = TrainUpdater>
        static TrainReport Train(Sequential& model, const Matrix& X, const vector<int>& y, const TrainConfig& cfg) {
            SoftmaxCrossEntropy CE;
            Matrix logits, dlogits;        // 배치 루프 전역 재사용
            Matrix Xb; vector<int> yb;     // 배치 버퍼 재사용
            Timer T, TU;
            mt19937 rng(cfg.seed);

            vector<double> epoch_ms_list, update_ms_list;
            float last_loss = 0.0f;

            // 전체 인덱스 0..N-1
            vector<int> base_idx(X.Rows());
            iota(base_idx.begin(), base_idx.end(), 0);

            for (int r = 0; r < cfg.repeats; ++r) {
                // 반복마다 독립 셔플을 위해 복사 후 셔플
                vector<int> idx = base_idx;

                double sum_epoch_ms = 0.0;
                double sum_up_ms = 0.0;

                for (int e = 0; e < cfg.epochs; ++e) {
                    shuffle(idx.begin(), idx.end(), rng);   // 에폭마다 인덱스만 셔플

                    T.Tic();
                    const int N = static_cast<int>(idx.size());

                    for (int beg = 0; beg < N; beg += cfg.batch) {
                        const int end = min(N, beg + cfg.batch);

                        // 인덱스 기반 슬라이스 (행 단위 memcpy)
                        SliceBatchIdx(X, y, idx, beg, end, Xb, yb);

                        // fwd -> loss -> bwd
                        model.Forward(Xb, logits);
                        last_loss = CE.Forward(logits, yb);
                        CE.Backward(yb, dlogits);
                        model.ZeroGrad();
                        model.Backward(dlogits);

                        // update 타이밍 분리 측정
                        TU.Tic();
                        Updater::Update(model, cfg.lr);
                        sum_up_ms += TU.TocMs();
                    }

                    double ep_ms = T.TocMs();
                    if (e >= cfg.warmup) sum_epoch_ms += ep_ms;
                }

                const int eff_epochs = max(0, cfg.epochs - cfg.warmup);
                const double avg_ep_ms = (eff_epochs > 0) ? (sum_epoch_ms / eff_epochs) : 0.0;
                const double avg_up_ms = (eff_epochs > 0) ? (sum_up_ms / eff_epochs) : 0.0;

                epoch_ms_list.push_back(avg_ep_ms);
                update_ms_list.push_back(avg_up_ms);
            }

            return { median_of(epoch_ms_list), median_of(update_ms_list), last_loss };
        }
    };
}