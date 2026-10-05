import unittest

import torch

from legged_lab.perception.stair_skill_teacher import StairSkillTeacher
from legged_lab.perception.stair_step_controller import StepPhase


class StairSkillTeacherTest(unittest.TestCase):
    def setUp(self):
        self.teacher = StairSkillTeacher(torch.zeros(2, 3), .22, .32, .11, 1, torch.full((2,), 1000.))
        self.root = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
        self.feet = torch.tensor([[[0., .15, 0.], [0., -.15, 0.]]]).repeat(2, 1, 1)
        self.forces = torch.zeros(2, 2, 3)
        self.forces[:, :, 2] = 500.
        self.rotation = torch.eye(3).repeat(2, 1, 1)
        self.root_velocity = torch.zeros_like(self.root)
        self.corner_offsets = torch.tensor([[-.12, -.042, 0.], [.12, -.042, 0.],
                                           [.12, .042, 0.], [-.12, .042, 0.]])

    def tick(self, n=1):
        for _ in range(n):
            result = self.teacher.update(
                self.root, self.rotation, self.feet, torch.zeros_like(self.feet),
                self.feet[:, :, None]+self.corner_offsets, torch.tensor([0., 0., 1.]).expand(2, 2, 3),
                self.forces, self.root_velocity, torch.zeros_like(self.root), .02)
        return result

    def test_actor_has_no_load_or_support_truth_and_contract_matches(self):
        actor, critic, _ = self.tick()
        self.assertEqual(actor.shape, (2, 39))
        self.assertTrue(torch.equal(actor[:, 24:26], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(actor[:, 34:36], torch.zeros(2, 2)))
        torch.testing.assert_close(critic[:, 24:26], torch.full((2, 2), .5))

    def test_both_leads_need_measured_unloading(self):
        self.tick(6)
        self.assertTrue((self.teacher.phase == int(StepPhase.SHIFT_LEAD)).all())
        self.tick(30)
        self.assertTrue((self.teacher.phase == int(StepPhase.SHIFT_LEAD)).all())
        self.forces[self.teacher.rows, self.teacher.lead, 2] = 0
        self.forces[self.teacher.rows, 1-self.teacher.lead, 2] = 1000
        self.tick(6)
        self.assertTrue((self.teacher.phase == int(StepPhase.LIFT_LEAD)).all())

    def test_curriculum_start_does_not_award_unearned_progress(self):
        _, _, metrics = self.tick()
        self.assertEqual(self.teacher.phase[0], int(StepPhase.SHIFT_LEAD))
        self.assertEqual(self.teacher.phase[1], int(StepPhase.OBSERVE))
        self.assertEqual(float(metrics['progress'][0]), 0.)
        self.assertFalse(self.teacher.success.any())

    def test_unloading_while_body_moves_does_not_authorize_lift(self):
        self.tick(6)
        rows, lead = self.teacher.rows, self.teacher.lead
        self.forces[rows, lead, 2] = 0
        self.forces[rows, 1-lead, 2] = 1000
        self.root_velocity[:, 1] = .25
        self.tick(10)
        self.assertTrue((self.teacher.phase == int(StepPhase.SHIFT_LEAD)).all())
        self.root_velocity[:] = 0
        self.tick(6)
        self.assertTrue((self.teacher.phase == int(StepPhase.LIFT_LEAD)).all())

    def test_lift_timer_does_not_authorize_horizontal_crossing(self):
        self.tick()
        self.teacher.phase[:] = int(StepPhase.LIFT_LEAD)
        self.teacher.swing_start[:] = self.feet[self.teacher.rows, self.teacher.lead]
        self.forces[self.teacher.rows, self.teacher.lead, 2] = 0
        self.forces[self.teacher.rows, 1-self.teacher.lead, 2] = 1000
        self.tick(40)
        self.assertFalse(self.teacher.clearance.any())
        self.assertTrue((self.teacher.reference_feet[self.teacher.rows, self.teacher.lead, 0] == 0).all())
        self.assertTrue((self.teacher.phase == int(StepPhase.LIFT_LEAD)).all())

    def test_small_lift_curriculum_does_not_lower_full_clearance_guard(self):
        self.tick(6)
        rows, lead = self.teacher.rows, self.teacher.lead
        self.forces[rows, lead, 2] = 0
        self.forces[rows, 1-lead, 2] = 1000
        self.tick(6)
        self.feet[rows, lead, 2] = .02
        _, _, metrics = self.tick(6)
        torch.testing.assert_close(self.teacher.lift_limit, torch.full((2,), .04))
        self.assertFalse(self.teacher.clearance.any())
        self.assertTrue((self.teacher.phase == int(StepPhase.LIFT_LEAD)).all())
        self.assertFalse(self.teacher.success.any())
        self.assertTrue((metrics['lift_progress'] == 0).all())


    def test_full_physical_sequence_and_stable_double_support(self):
        self.tick(6)
        rows, lead = self.teacher.rows, self.teacher.lead
        trail = 1-lead
        self.forces[rows, lead, 2] = 0
        self.forces[rows, trail, 2] = 1000
        self.tick(6)
        self.feet[rows, lead, 2] = .17
        self.tick()
        self.feet[rows, lead, 0] = self.teacher.targets[rows, lead, 0]
        self.tick()
        self.assertTrue((self.teacher.phase == int(StepPhase.LOWER_LEAD)).all())
        self.feet[rows, lead, 2] = .11
        self.forces[rows, lead, 2] = 450
        self.forces[rows, trail, 2] = 550
        self.tick(6)
        self.assertTrue((self.teacher.phase == int(StepPhase.TRANSFER)).all())
        self.forces[rows, lead, 2] = 700
        self.forces[rows, trail, 2] = 300
        self.tick(6)
        self.forces[rows, lead, 2] = 1000
        self.forces[rows, trail, 2] = 0
        self.tick(6)
        self.feet[rows, trail, 2] = .17
        self.tick()
        self.feet[rows, trail, 0] = self.teacher.targets[rows, trail, 0]
        self.tick()
        self.feet[rows, trail, 2] = .11
        self.forces[:, :, 2] = 500
        self.tick(6)
        self.assertTrue((self.teacher.phase == int(StepPhase.SETTLE)).all())
        self.assertFalse(self.teacher.success.any())
        self.tick(10)
        self.assertTrue(self.teacher.success.all())

    def test_tread_edge_contact_never_counts_as_completion(self):
        self.tick()
        self.feet[:] = self.teacher.targets
        self.feet[:, :, 0] = self.teacher.far[:, None]-.05
        self.teacher.phase[:] = int(StepPhase.SETTLE)
        self.tick(20)
        self.assertFalse(self.teacher.success.any())

    def test_world_goal_stays_fixed_when_body_moves(self):
        actor, _, _ = self.tick()
        targets = self.teacher.targets.clone()
        self.root[:, 0] += .04
        self.feet[:, :, 0] += .04
        updated, _, _ = self.tick()
        torch.testing.assert_close(self.teacher.targets, targets)
        torch.testing.assert_close(updated[:, 12:18:3], actor[:, 12:18:3]-.04)

    def test_reset_does_not_clear_previous_outcome_before_logging(self):
        self.teacher.success[:] = True
        self.teacher.reset(torch.tensor([0]))
        self.assertTrue(self.teacher.success[0])
        self.tick()
        self.assertFalse(self.teacher.success.any())
