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

"""Core token-level objective of TIDE.

TIDE splits teacher--student disagreement at a student-visited state into two
directions and treats each with its own mechanism:

* **Excess branch.** The student over-weights the sampled token relative to the
  teacher (``a_t = log p(o_t | s_t) - log q(o_t | s_t) < 0``). The raw log-ratio
  is unbounded below, so suppression goes through the bounded Hellinger shaping
  ``h(a) = 2 (exp(a / 2) - 1) > -2``.
* **Deficit branch.** The teacher prefers tokens the student rarely samples.
  Such tokens are invisible to a sampled-token estimator, so the deficit score
  ``d_t`` is measured on the teacher's top-K support and the missing mass is
  recovered with an analytic cross-entropy update towards the renormalized
  teacher.

Both branches are restricted to the most informative positions by batch-adaptive
quantile gates with keep-rates ``rho^-`` and ``rho^+``. Positions selected by
neither branch (the matched band) receive exactly zero advantage.

This module is pure PyTorch and has no framework dependency; :func:`tide_scores`
is the single entry point used by the trainer.
"""

import torch


def hellinger_reward_from_log_probs(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the Hellinger-shaped reward ``h(a) = 2 (exp(a / 2) - 1)``.

    With ``a = log p - log q`` this is ``2 (sqrt(p / q) - 1)``, the exact
    negative-gradient reward of the squared Hellinger distance. Three properties
    matter for the excess branch:

    * **Bounded suppression.** ``h(a) > -2`` for every ``a < 0``, so a token the
      teacher rejects cannot produce an arbitrarily large update no matter how
      confident the student is.
    * **Local fidelity.** ``h(a) = a + O(a^2)``, so near agreement the update
      keeps the first-order behaviour of the standard OPD log-ratio reward.
    * **Proper divergence.** Treated as a detached coefficient, the resulting
      policy gradient is exactly ``4 grad H^2(p, q)``, so the saturation is not
      an arbitrary cutoff but the gradient of a divergence.

    The tensors may be sampled-token tensors ``(B, T)`` or candidate-token
    tensors ``(B, T, K)`` as long as their shapes match.
    """
    if student_log_prob.shape != teacher_log_prob.shape:
        raise ValueError(
            "student and teacher log-probability tensors must have the same shape, "
            f"got {student_log_prob.shape} and {teacher_log_prob.shape}"
        )

    delta = teacher_log_prob.float() - student_log_prob.float()
    # exp overflow guard: binds only above e^60, far beyond anything a softmax
    # pair produces in practice.
    scaled = (0.5 * delta).clamp(max=60.0)
    return 2.0 * torch.expm1(scaled)


def compute_tide_gates(
    excess_stat: torch.Tensor,
    deficit_stat: torch.Tensor,
    valid_mask: torch.Tensor,
    keep_neg: float,
    keep_pos: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Select the two active sets with batch-adaptive quantile thresholds.

    Args:
        excess_stat: ``(B, T)`` statistic whose LARGE values indicate excess
            (student confident, teacher disagrees). TIDE uses ``[-a_t]_+``.
        deficit_stat: ``(B, T)`` statistic whose LARGE values indicate deficit
            (teacher prefers tokens the student ignores). TIDE uses ``d_t``.
        valid_mask: ``(B, T)`` bool/int mask of valid response tokens.
        keep_neg: keep-rate ``rho^-`` of the excess gate, in ``[0, 1]``.
        keep_pos: keep-rate ``rho^+`` of the deficit gate, in ``[0, 1]``.

    Each threshold is the batch ``1 - keep`` quantile over valid tokens, so
    exactly a ``keep`` fraction stays active and the gates anneal themselves: as
    the residual distribution shrinks during training, the implied absolute
    thresholds shrink with it.

    Returns:
        ``(excess_mask, deficit_mask, tau_neg, tau_pos)`` where the masks are
        ``(B, T)`` float tensors in ``{0, 1}`` already multiplied by
        ``valid_mask``, and the taus are the realized thresholds.
    """
    if not 0.0 <= keep_neg <= 1.0 or not 0.0 <= keep_pos <= 1.0:
        raise ValueError(f"keep rates must lie in [0, 1], got {keep_neg}, {keep_pos}")

    valid = valid_mask.bool()

    def _gate(stat: torch.Tensor, keep: float) -> tuple[torch.Tensor, float]:
        stat = stat.float()
        if keep >= 1.0:
            return valid.float(), float("-inf")
        if keep <= 0.0:
            return torch.zeros_like(stat), float("inf")
        vals = stat[valid]
        if vals.numel() == 0:
            return torch.zeros_like(stat), float("inf")
        tau = torch.quantile(vals, 1.0 - keep)
        mask = (stat >= tau) & valid
        return mask.float(), tau.item()

    excess_mask, tau_neg = _gate(excess_stat, keep_neg)
    deficit_mask, tau_pos = _gate(deficit_stat, keep_pos)
    return excess_mask, deficit_mask, tau_neg, tau_pos


def tide_scores(
    sampled_student_logp: torch.Tensor,
    sampled_teacher_logp: torch.Tensor,
    teacher_topk_logp: torch.Tensor,
    student_on_teacher_logp: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    keep_neg: float = 0.2,
    keep_pos: float = 0.2,
    deficit_lambda: float = 1.0,
) -> dict:
    """Assemble the detached per-candidate advantages of the TIDE objective.

    The output is laid out for a ``(B, T, 1 + K)`` candidate policy gradient:
    slot 0 is the student's sampled token and carries the excess advantage,
    slots ``1..K`` are the teacher's top-K tokens and carry the deficit
    advantage. The corresponding loss is

        L = -(1 / N) sum_t [ excess_scores[t] * log q(o_t | s_t)
                             + sum_v deficit_scores[t, v] * log q(v | s_t) ].

    * **Excess branch.** ``excess_scores`` holds the Hellinger-shaped reward
      ``h(a_t) <= 0`` on the active set ``A^- = {a_t < 0, -a_t >= tau^-}``.
    * **Deficit branch.** ``deficit_scores`` holds ``lambda * p_bar_t(v)`` on the
      active set ``A^+ = {d_t >= tau^+}``, where ``p_bar_t`` is the teacher
      renormalized over its top-K support. This is the analytic cross-entropy
      update, whose logit gradient ``lambda (q(v) - p_bar_t(v))`` does not
      vanish with ``q(v)`` and therefore escapes the sampling barrier.
    * **Matched band.** Positions in neither active set carry exactly zero on
      every slot.

    Args:
        sampled_student_logp: ``(B, T)`` student log-prob of the sampled token.
        sampled_teacher_logp: ``(B, T)`` teacher log-prob of the sampled token.
        teacher_topk_logp: ``(B, T, K)`` teacher log-probs of its own top-K.
        student_on_teacher_logp: ``(B, T, K)`` student log-probs at those ids.
        response_mask: ``(B, T)`` valid-token mask.
        keep_neg / keep_pos: quantile keep-rates ``rho^-`` / ``rho^+``.
        deficit_lambda: weight ``lambda`` of the deficit branch.

    Returns:
        dict with ``excess_scores`` ``(B, T)``, ``deficit_scores`` ``(B, T, K)``,
        the two float gate masks ``excess_mask`` / ``deficit_mask`` ``(B, T)``,
        the raw statistics ``sampled_residual`` ``(B, T)`` and ``deficit_stat``
        ``(B, T)``, and the realized thresholds ``tau_neg`` / ``tau_pos``.
    """
    valid = response_mask.bool()

    # Sampled-token log-ratio a_t = log p - log q.
    sampled_residual = (sampled_teacher_logp.float() - sampled_student_logp.float()) * valid

    # Deficit score d_t: KL from the renormalized teacher top-K to the student
    # taken WITHOUT renormalizing the student, so d_t also charges the student
    # for probability mass leaked outside the teacher's support.
    teacher_bar_logp = teacher_topk_logp.float() - torch.logsumexp(
        teacher_topk_logp.float(), dim=-1, keepdim=True
    )
    teacher_bar_p = torch.exp(teacher_bar_logp)
    deficit_stat = (
        teacher_bar_p * (teacher_bar_logp - student_on_teacher_logp.float())
    ).sum(dim=-1)
    deficit_stat = deficit_stat.clamp(min=0.0) * valid

    excess_mask, deficit_mask, tau_neg, tau_pos = compute_tide_gates(
        excess_stat=(-sampled_residual).clamp(min=0.0),
        deficit_stat=deficit_stat,
        valid_mask=valid,
        keep_neg=keep_neg,
        keep_pos=keep_pos,
    )
    # Guard against a degenerate quantile tau^- <= 0 (fewer negative-residual
    # tokens than the keep rate): the excess set must only contain a_t < 0.
    excess_mask = excess_mask * (sampled_residual < 0).float()

    excess_reward = hellinger_reward_from_log_probs(
        sampled_student_logp, sampled_teacher_logp
    )

    excess_scores = excess_reward * excess_mask * valid
    deficit_scores = (
        deficit_lambda
        * teacher_bar_p
        * deficit_mask.unsqueeze(-1)
        * valid.unsqueeze(-1).float()
    )

    return {
        "excess_scores": excess_scores,
        "deficit_scores": deficit_scores,
        "excess_mask": excess_mask,
        "deficit_mask": deficit_mask,
        "sampled_residual": sampled_residual,
        "deficit_stat": deficit_stat,
        "tau_neg": tau_neg,
        "tau_pos": tau_pos,
    }
