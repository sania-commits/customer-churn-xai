from src.api import assign_risk_segment


def test_low_risk_segment():
    assert assign_risk_segment(0.00) == "Low"
    assert assign_risk_segment(0.09) == "Low"


def test_medium_risk_segment():
    assert assign_risk_segment(0.10) == "Medium"
    assert assign_risk_segment(0.50) == "Medium"


def test_high_risk_segment():
    assert assign_risk_segment(0.51) == "High"
    assert assign_risk_segment(0.89) == "High"


def test_critical_risk_segment():
    assert assign_risk_segment(0.90) == "Critical"
    assert assign_risk_segment(1.00) == "Critical"