"""Pure grasp-system primitives (no ROS, fully unit-testable).

This module is the single source of truth for the grasping logic:

  * :class:`GraspGeometry` - where the fingertips sit relative to tool0
  * :func:`closing_schedule` - incremental jaw commands
  * :func:`detect_contact` - "jaws blocked by an object" test
  * :func:`classify_hold` - is the measured jaw width a real grasp?

Keeping the maths here (instead of inside the ROS node) means the grasp
behaviour can be tested on any machine and reused by other controllers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspGeometry:
    """Vertical geometry of a top-down grasp.

    All values are metres, expressed in the tool0 frame conventions used by
    this project: the fingers hang along -tool z, so the fingertip is
    ``finger_length`` below tool0.
    """

    finger_length: float = 0.104
    table_clearance: float = 0.015   # fingertips stay this far above the table
    approach_offset: float = 0.18    # pre-grasp height above the descend pose
    test_lift: float = 0.05          # small lift used to detect slipping

    @property
    def descend_z(self) -> float:
        """Tool0 height at the grasp pose."""
        return self.table_clearance + self.finger_length

    @property
    def approach_z(self) -> float:
        """Tool0 height above the object before descending."""
        return self.descend_z + self.approach_offset

    @property
    def test_lift_z(self) -> float:
        """Tool0 height of the short verification lift."""
        return self.descend_z + self.test_lift


def closing_schedule(stroke, steps=4):
    """Increasing per-finger commands from ``stroke/steps`` to ``stroke``.

    The first command is already a quarter of the stroke: the gazebo
    position loop is slow, so tiny first steps look like "no movement" and
    would be misread as contact.
    """
    steps = max(1, int(steps))
    return [float(stroke) * (index + 1) / steps for index in range(steps)]


def detect_contact(previous_width, current_width, expected_change, ratio=0.25):
    """Return ``(blocked, moved)`` for one closing step.

    ``expected_change`` is how much the jaw width should shrink if the jaws
    moved freely (twice the per-finger command). If they moved less than
    ``ratio`` of that, something is between them.
    """
    moved = float(previous_width) - float(current_width)
    return moved < float(ratio) * float(expected_change), moved


def classify_hold(width, open_gap, closed_gap, margin=0.004):
    """A grasp is real when the jaws stop between the closed and open gaps."""
    lower = float(closed_gap) + float(margin)
    upper = float(open_gap) - 0.005
    return lower < float(width) < upper
