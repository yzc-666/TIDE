# Copyright 2026 The TIDE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the TIDE objective. No GPU or model checkpoint required."""

import pytest
import torch

from verl.utils.tide_reward import (
    compute_tide_gates,
    hellinger_reward_from_log_probs,
    tide_scores,
)


# --------------------------------------------------------------------------- #
# Excess branch: Hellinger shaping
# --------------------------------------------------------------------------- #
def test_hellinger_matches_closed_form():
    student_p = torch.tensor([0.4, 0.9, 1e-6, 0.5], dtype=torch.float64)
    teacher_p = torch.tensor([0.4, 1e-8, 0.5, 0.1], dtype=torch.float64)

    actual = hellinger_reward_from_log_probs(student_p.log(), teacher_p.log())
    expected = (2.0 * ((teacher_p / student_p).sqrt() - 1.0)).to(torch.float32)

    torch.testing.assert_close(actual, expected)


def test_hellinger_is_zero_at_equality_and_sign_consistent():
    student_p = torch.tensor([0.3, 0.6, 0.01], dtype=torch.float64)
    teacher_p = torch.tensor([0.3, 0.2, 0.05], dtype=torch.float64)

    reward = hellinger_reward_from_log_probs(student_p.log(), teacher_p.log())

    assert reward[0].item() == pytest.approx(0.0, abs=1e-6)
    assert reward[1] < 0  # teacher assigns less probability -> suppress
    assert reward[2] > 0  # teacher assigns more probability -> reinforce


def test_hellinger_negative_side_is_bounded_by_minus_two():
    # Even a catastrophic student-only token cannot receive a penalty below -2.
    student_logp = torch.tensor([-0.1, -0.5])
    teacher_logp = torch.tensor([-1000.0, -40.0])

    reward = hellinger_reward_from_log_probs(student_logp, teacher_logp)

    assert torch.isfinite(reward).all()
    # Mathematically r > -2; in floating point the deep tail saturates at -2.
    assert (reward >= -2.0).all()


def test_hellinger_is_first_order_faithful_near_agreement():
    # h(a) = a + O(a^2), so small disagreements keep standard OPD behaviour.
    a = torch.tensor([1e-3, -1e-3, 5e-4])
    student_logp = torch.zeros_like(a)

    reward = hellinger_reward_from_log_probs(student_logp, a)

    torch.testing.assert_close(reward, a, atol=1e-6, rtol=1e-3)


def test_hellinger_second_moment_identity():
    # E_{v ~ q}[r(v)^2] = 4 sum_v (sqrt(p_v) - sqrt(q_v))^2 = 8 H^2(p, q).
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        q = torch.rand(64, generator=generator, dtype=torch.float64)
        p = torch.rand(64, generator=generator, dtype=torch.float64)
        q = q / q.sum()
        p = p / p.sum()

        reward = hellinger_reward_from_log_probs(q.log(), p.log())
        second_moment = (q * reward.double().square()).sum()
        squared_hellinger = ((p.sqrt() - q.sqrt()) ** 2).sum() / 2.0

        assert second_moment.item() == pytest.approx(8.0 * squared_hellinger.item(), rel=1e-5)
        assert second_moment.item() <= 8.0 + 1e-9


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        hellinger_reward_from_log_probs(torch.zeros(2, 3), torch.zeros(2, 4))


# --------------------------------------------------------------------------- #
# Quantile gates
# --------------------------------------------------------------------------- #
def test_gates_keep_the_requested_fraction():
    generator = torch.Generator().manual_seed(0)
    excess = torch.rand(4, 100, generator=generator)
    deficit = torch.rand(4, 100, generator=generator)
    valid = torch.ones(4, 100, dtype=torch.bool)

    excess_mask, deficit_mask, tau_neg, tau_pos = compute_tide_gates(
        excess, deficit, valid, keep_neg=0.2, keep_pos=0.1
    )

    assert excess_mask.sum().item() == pytest.approx(0.2 * 400, abs=2)
    assert deficit_mask.sum().item() == pytest.approx(0.1 * 400, abs=2)
    # Kept tokens are exactly those at or above the realized thresholds.
    assert bool(((excess >= tau_neg) == excess_mask.bool()).all())
    assert bool(((deficit >= tau_pos) == deficit_mask.bool()).all())


def test_gates_respect_the_valid_mask_and_edge_keep_rates():
    stat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    valid = torch.tensor([[True, True, False, False]])

    # Padding is never gated in, even at keep = 1.
    m_all, _, _, _ = compute_tide_gates(stat, stat, valid, keep_neg=1.0, keep_pos=1.0)
    torch.testing.assert_close(m_all, valid.float())

    # keep = 0 disables the branch.
    m_none, _, _, _ = compute_tide_gates(stat, stat, valid, keep_neg=0.0, keep_pos=0.0)
    assert m_none.sum().item() == 0.0

    with pytest.raises(ValueError, match="keep rates"):
        compute_tide_gates(stat, stat, valid, keep_neg=1.5, keep_pos=0.2)


# --------------------------------------------------------------------------- #
# Full objective
# --------------------------------------------------------------------------- #
def _make_inputs(seed=0, B=2, T=50, K=4):
    generator = torch.Generator().manual_seed(seed)
    student_logp = -torch.rand(B, T, generator=generator) * 5.0
    teacher_logp = -torch.rand(B, T, generator=generator) * 5.0
    teacher_topk_logp = -torch.sort(torch.rand(B, T, K, generator=generator) * 4.0, dim=-1).values
    student_on_teacher_logp = -torch.rand(B, T, K, generator=generator) * 8.0
    mask = torch.ones(B, T)
    mask[:, -5:] = 0.0
    return student_logp, teacher_logp, teacher_topk_logp, student_on_teacher_logp, mask


def test_scores_have_the_expected_structure_and_signs():
    student_logp, teacher_logp, t_topk, s_on_t, mask = _make_inputs()

    out = tide_scores(
        student_logp, teacher_logp, t_topk, s_on_t, mask,
        keep_neg=0.3, keep_pos=0.3, deficit_lambda=2.0,
    )

    excess, deficit = out["excess_scores"], out["deficit_scores"]
    assert excess.shape == student_logp.shape
    assert deficit.shape == t_topk.shape
    # The excess branch only suppresses, and Hellinger bounds it below by -2.
    assert (excess <= 0).all() and (excess >= -2.0).all()
    # The deficit branch only reinforces; its per-position weights are the
    # renormalized teacher, so they sum to lambda on gated positions.
    assert (deficit >= 0).all()
    gated = out["deficit_mask"].bool()
    torch.testing.assert_close(
        deficit.sum(-1)[gated], torch.full_like(deficit.sum(-1)[gated], 2.0)
    )
    # The matched band carries exactly zero on every slot.
    matched = (~out["excess_mask"].bool()) & (~gated)
    assert excess[matched].abs().sum().item() == 0.0
    assert deficit[matched].abs().sum().item() == 0.0
    # Padding carries nothing.
    assert excess[:, -5:].abs().sum().item() == 0.0
    assert deficit[:, -5:].abs().sum().item() == 0.0


def test_excess_set_contains_only_negative_log_ratios():
    student_logp, teacher_logp, t_topk, s_on_t, mask = _make_inputs(seed=1)

    out = tide_scores(
        student_logp, teacher_logp, t_topk, s_on_t, mask, keep_neg=0.9, keep_pos=0.1
    )

    # Even at an aggressive keep-rate, every gated token must have a_t < 0.
    gated = out["excess_mask"].bool()
    assert (out["sampled_residual"][gated] < 0).all()


def test_deficit_score_is_the_teacher_top_k_kl_without_student_renormalization():
    student_logp, teacher_logp, t_topk, s_on_t, mask = _make_inputs(seed=2)

    out = tide_scores(student_logp, teacher_logp, t_topk, s_on_t, mask)

    p_bar = torch.softmax(t_topk, dim=-1)
    expected = (p_bar * (torch.log(p_bar) - s_on_t)).sum(-1).clamp(min=0.0) * mask
    torch.testing.assert_close(out["deficit_stat"], expected, atol=1e-5, rtol=1e-5)


def test_deficit_score_vanishes_only_when_the_student_covers_the_support():
    # d_t = KL(p_bar || q^K) - log Q_t >= 0, with equality iff the student
    # matches the renormalized teacher AND puts all its mass on the top-K set.
    teacher_topk_logp = torch.log(torch.tensor([[[0.6, 0.4]]]))
    mask = torch.ones(1, 1)
    sampled = torch.log(torch.tensor([[0.5]]))

    matched = tide_scores(sampled, sampled, teacher_topk_logp, teacher_topk_logp, mask)
    assert matched["deficit_stat"].item() == pytest.approx(0.0, abs=1e-6)

    # Same conditional shape, but half the mass leaks outside the support.
    half = torch.log(torch.tensor(0.5))
    leaked = tide_scores(
        sampled, sampled, teacher_topk_logp, teacher_topk_logp + half, mask
    )
    assert leaked["deficit_stat"].item() == pytest.approx(-half.item(), abs=1e-5)


def test_deficit_gradient_does_not_vanish_with_the_student_probability():
    # The analytic cross-entropy update has logit gradient q(v) - p_bar(v),
    # which stays Theta(1) for a teacher-preferred token the student ignores.
    teacher_logits = torch.tensor([2.0, 1.0, 0.5])
    student_logits = torch.tensor([-8.0, 0.3, 1.2], requires_grad=True)
    p_bar = torch.softmax(teacher_logits, dim=-1)

    loss = -(p_bar * torch.log_softmax(student_logits, dim=-1)).sum()
    loss.backward()

    expected = torch.softmax(student_logits.detach(), dim=-1) - p_bar
    torch.testing.assert_close(student_logits.grad, expected)
    assert abs(student_logits.grad[0].item()) > 0.5


def test_branches_can_be_disabled_independently():
    student_logp, teacher_logp, t_topk, s_on_t, mask = _make_inputs(seed=3)

    deficit_only = tide_scores(
        student_logp, teacher_logp, t_topk, s_on_t, mask, keep_neg=0.0, keep_pos=0.2
    )
    assert deficit_only["excess_scores"].abs().sum().item() == 0.0
    assert deficit_only["deficit_scores"].abs().sum().item() > 0.0

    excess_only = tide_scores(
        student_logp, teacher_logp, t_topk, s_on_t, mask, keep_neg=0.2, deficit_lambda=0.0
    )
    assert excess_only["excess_scores"].abs().sum().item() > 0.0
    assert excess_only["deficit_scores"].abs().sum().item() == 0.0
