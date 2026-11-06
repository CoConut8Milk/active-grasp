"""Tests for the rewritten grasp system (pure geometry / contact logic)."""

from ag_execution.grasp_system import (
    GraspGeometry,
    classify_hold,
    closing_schedule,
    detect_contact,
)


def test_geometry_places_fingertips_above_the_table():
    geometry = GraspGeometry(finger_length=0.104)
    assert abs(geometry.descend_z - 0.119) < 1e-9
    assert abs(geometry.approach_z - 0.299) < 1e-9
    assert abs(geometry.test_lift_z - 0.169) < 1e-9
    # fingertips = descend_z - finger_length
    assert abs((geometry.descend_z - geometry.finger_length) - 0.015) < 1e-9


def test_closing_schedule_is_incremental():
    schedule = closing_schedule(0.030, steps=4)
    assert schedule == [0.0075, 0.015, 0.0225, 0.030]
    assert closing_schedule(0.030, steps=1) == [0.030]


def test_contact_is_detected_from_blocked_movement():
    # Free jaws: they shrink by the expected 15 mm.
    blocked, moved = detect_contact(0.066, 0.051, expected_change=0.015)
    assert not blocked and abs(moved - 0.015) < 1e-9
    # Object between the jaws: they barely move.
    blocked, moved = detect_contact(0.040, 0.0395, expected_change=0.015)
    assert blocked and moved < 0.001
    # Hysteresis: a quarter of the expected movement still counts as free.
    blocked, _ = detect_contact(0.040, 0.0360, expected_change=0.015)
    assert not blocked


def test_hold_classification():
    assert classify_hold(0.030, open_gap=0.066, closed_gap=0.006)
    assert not classify_hold(0.006, open_gap=0.066, closed_gap=0.006)
    assert not classify_hold(0.062, open_gap=0.066, closed_gap=0.006)
