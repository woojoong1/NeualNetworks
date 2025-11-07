// =============================
// include/vsnn/Matrix.hpp (개선 버전)
// =============================
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace std;

namespace vsnn {
    using f32 = float;
    using i32 = int32_t;

    class Matrix {
    private:
        i32 rows_ = 0, cols_ = 0;
        vector<f32> data_; // 기본 vector 사용

        // 재할당 빈도를 줄이기 위한 내부 성장 전략
        inline void EnsureSize_(size_t need) {
            if (data_.size() >= need) return;
            // capacity를 넉넉히 확보하여 반복 재할당 방지
            size_t new_cap = data_.capacity();
            if (new_cap < need) {
                new_cap = max(need, new_cap * 2ull + 64ull);
            }
            vector<f32> tmp;
            tmp.reserve(new_cap);
            data_.swap(tmp);          // 기존 메모리 해제 & 새 capacity 확保
            data_.resize(need);       // 길이만 need로 설정 (값 초기화는 하지 않음)
        }

    public:
        Matrix() = default;
        Matrix(i32 r, i32 c) { Reset(r, c); }

        // 기존 Reset semantics 유지: 항상 0으로 채움
        inline void Reset(i32 r, i32 c) {
            rows_ = r; cols_ = c;
            const size_t need = static_cast<size_t>(r) * static_cast<size_t>(c);
            data_.assign(need, 0.0f);
        }

        // 제로필 없이 크기만 맞춤 (연산으로 덮어쓸 때 사용)
        inline void ResetNoInit(i32 r, i32 c) {
            rows_ = r; cols_ = c;
            const size_t need = static_cast<size_t>(r) * static_cast<size_t>(c);
            EnsureSize_(need);        // 필요 시에만 재할당(여유 capacity 확보)
            data_.resize(need);       // 값은 초기화하지 않음
        }

        // 필요할 때만 명시적으로 0으로 채움
        inline void Zero() {
            std::fill(data_.begin(), data_.end(), 0.0f);
        }

        inline i32 Rows() const { return rows_; }
        inline i32 Cols() const { return cols_; }
        inline f32* Data() { return data_.data(); }
        inline const f32* Data() const { return data_.data(); }

        inline f32& operator()(i32 r, i32 c) { return data_[static_cast<size_t>(r) * cols_ + c]; }
        inline f32  operator()(i32 r, i32 c) const { return data_[static_cast<size_t>(r) * cols_ + c]; }

        inline void Fill(f32 v) { std::fill(data_.begin(), data_.end(), v); }

        inline const vector<f32>& Raw() const { return data_; }
        inline vector<f32>& Raw() { return data_; }
    };
}
